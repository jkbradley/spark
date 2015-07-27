/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.tree.impl

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.Logging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.TreeUtil._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.FeatureType.FeatureType
import org.apache.spark.mllib.tree.configuration.{FeatureType, Strategy}
import org.apache.spark.mllib.tree.impurity.{Variance, Gini, Entropy}
import org.apache.spark.mllib.tree.model.{Predict => OldPredict}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.collection.BitSet


/**
 * DecisionTree which partitions data by feature.
 *
 * Algorithm:
 *  - Repartition data, grouping by feature.
 *  - Prep data (sort continuous features).
 *  - On each partition, initialize instance--node map with each instance at root node.
 *  - Iterate, training 1 new level of the tree at a time:
 *     - On each partition, for each feature on the partition, select the best split for each node.
 *     - Aggregate best split for each node.
 *     - Aggregate bit vector (1 bit/instance) indicating whether each instance splits
 *       left or right.
 *     - Broadcast bit vector.  On each partition, update instance--node map.
 *
 * TODO: Update to use a sparse column store.
 */
private[ml] object AltDT extends Logging {

  private def myPrint(s: String = ""): Unit = return // print(s)
  private def myPrintln(s: String = ""): Unit = return // println(s)

  private[impl] def createImpurityAggregator(strategy: Strategy): ImpurityAggregatorSingle = {
    strategy.impurity match {
      case Entropy => new EntropyAggregatorSingle(strategy.numClasses)
      case Gini => new GiniAggregatorSingle(strategy.numClasses)
      case Variance => new VarianceAggregatorSingle
    }
  }

  /**
   * Method to train a decision tree model over an RDD.
   */
  def train(
      input: RDD[LabeledPoint],
      strategy: Strategy,
      parentUID: Option[String] = None): DecisionTreeModel = {
    // TODO: Check validity of params
    val rootNode = trainImpl(input, strategy)
    RandomForest.finalizeTree(rootNode, strategy.algo, parentUID)
  }

  private[impl] def trainImpl(input: RDD[LabeledPoint], strategy: Strategy): Node = {
    // The case with 1 node (depth = 0) is handled separately.
    // This allows all iterations in the depth > 0 case to use the same code.
    if (strategy.maxDepth == 0) {
      val impurityAggregator: ImpurityAggregatorSingle =
        input.aggregate(createImpurityAggregator(strategy))(
          (agg, lp) => agg.update(lp.label, 1.0),
          (agg1, agg2) => agg1.add(agg2))
      val impurityCalculator = impurityAggregator.getCalculator
      return new LeafNode(impurityCalculator.getPredict.predict, impurityCalculator.calculate())
    }

    // Prepare column store.
    //   Note: rowToColumnStoreDense checks to make sure numRows < Int.MaxValue.
    // TODO: Is this mapping from arrays to iterators to arrays (when constructing learningData)?
    //       Or is the mapping implicit (i.e., not costly)?
    val colStoreInit: RDD[(Int, Vector)] = rowToColumnStoreDense(input.map(_.features))
    val numRows: Int = colStoreInit.first()._2.size
    myPrintln(s"(D) numRows = $numRows")
    val labels = new Array[Double](numRows)
    input.map(_.label).zipWithIndex().collect().foreach { case (label: Double, rowIndex: Long) =>
      labels(rowIndex.toInt) = label
    }
    val labelsBc = input.sparkContext.broadcast(labels)
    // NOTE: Labels are not sorted with features since that would require 1 copy per feature,
    //       rather than 1 copy per worker.  This means a lot of random accesses.
    //       We could improve this by applying first-level sorting (by node) to labels.

    // Sort each column by feature values.
    val colStore: RDD[FeatureVector] = colStoreInit.map { case (featureIndex: Int, col: Vector) =>
      if (strategy.categoricalFeaturesInfo.contains(featureIndex)) {
        FeatureVector.fromOriginal(featureIndex, FeatureType.Categorical, col)
      } else {
        FeatureVector.fromOriginal(featureIndex, FeatureType.Continuous, col)
      }
    }
    myPrintln(s"colStore.numPartitions: ${colStore.partitions.length}")
    // Group columns together into one array of columns per partition.
    val groupedColStore: RDD[Array[FeatureVector]] = colStore.mapPartitions { iterator =>
      val groupedCols = new ArrayBuffer[FeatureVector]
      iterator.foreach(groupedCols += _)
      if (groupedCols.nonEmpty) Iterator(groupedCols.toArray) else Iterator()
    }
    groupedColStore.repartition(1).persist(StorageLevel.MEMORY_AND_DISK) // TODO: remove repartition

    // Initialize partitions with 1 node (each instance at the root node).
    var partitionInfosA: RDD[PartitionInfo] = groupedColStore.map { groupedCols =>
      val initActive = new BitSet(1)
      initActive.setUntil(1)
      new PartitionInfo(groupedCols, Array[Int](0, numRows), initActive)
    }

    // Initialize model.
    // Note: We do not use node indices.
    val rootNode = LearningNode.emptyNode(1) // TODO: remove node id
    // Active nodes (still being split), updated each iteration
    var activeNodePeriphery: Array[LearningNode] = Array(rootNode)
    var numNodeOffsets: Int = 2

    val partitionInfosDebug = new scala.collection.mutable.ArrayBuffer[RDD[PartitionInfo]]()
    partitionInfosDebug.append(partitionInfosA)

    // Iteratively learn, one level of the tree at a time.
    var currentLevel = 0
    var doneLearning = false
    while (currentLevel < strategy.maxDepth && !doneLearning) {
      myPrintln("\n========================================\n")
      myPrintln(s"(D) CURRENT LEVEL: $currentLevel")

      val partitionInfos = partitionInfosDebug.last
      myPrintln(s"(D) A: Current partitionInfos:\n")
      partitionInfos.collect().foreach(x => myPrintln(x.toString))
      myPrintln()

      // Compute best split for each active node.
      val bestSplitsAndGains: Array[(Split, InfoGainStats)] =
        computeBestSplits(partitionInfos, labelsBc, strategy)
      /*
      // NOTE: The actual active nodes (activeNodePeriphery) may be a subset of the nodes under
      //       bestSplitsAndGains since
      assert(activeNodePeriphery.length == bestSplitsAndGains.length,
        s"activeNodePeriphery.length=${activeNodePeriphery.length} does not equal" +
          s" bestSplitsAndGains.length=${bestSplitsAndGains.length}")
      */

      // Update current model and node periphery.
      // Note: This flatMap has side effects (on the model).
      activeNodePeriphery =
        computeActiveNodePeriphery(activeNodePeriphery, bestSplitsAndGains, strategy.getMinInfoGain)
      myPrintln(s"(D) activeNodePeriphery.length: ${activeNodePeriphery.length}")
      // We keep all old nodeOffsets and add one for each node split.
      // Each node split adds 2 nodes to activeNodePeriphery.
      // TODO: Should this be calculated after filtering for impurity??
      numNodeOffsets = numNodeOffsets + activeNodePeriphery.length / 2
      myPrintln(s"(D) numNodeOffsets: $numNodeOffsets")

      // Filter active node periphery by impurity.
      val estimatedRemainingActive = activeNodePeriphery.count(_.impurity > 0.0)

      // TODO: Check to make sure we split something, and stop otherwise.
      doneLearning = currentLevel + 1 >= strategy.maxDepth || estimatedRemainingActive == 0

      if (!doneLearning) {
        // Aggregate bit vector (1 bit/instance) indicating whether each instance goes left/right.
        val aggBitVectors: Array[BitSubvector] = collectBitVectors(partitionInfos, bestSplitsAndGains)

        myPrintln(s"(D) B: First partitionInfos' nodeOffsets: ${partitionInfos.first().nodeOffsets.mkString("(",",",")")}")

        // Broadcast aggregated bit vectors.  On each partition, update instance--node map.
        val aggBitVectorsBc = input.sparkContext.broadcast(aggBitVectors)
        // partitionInfos = partitionInfos.map { partitionInfo =>
        val partitionInfosB = partitionInfos.map { partitionInfo =>
          partitionInfo.update(aggBitVectorsBc.value, numNodeOffsets)
        }
        partitionInfosB.cache().count() // TODO: remove
        partitionInfosDebug.append(partitionInfosB)

        myPrintln(s"(D) C: First partitionInfos' nodeOffsets: ${partitionInfosB.first().nodeOffsets.mkString("(",",",")")}")
        // TODO: unpersist aggBitVectorsBc after action.
      }

      currentLevel += 1
    }

    // Done with learning
    groupedColStore.unpersist()
    labelsBc.unpersist()
    // TODO: return model
    rootNode.toNode
  }

  /**
   * Find the best splits for all active nodes.
   *  - On each partition, for each feature on the partition, select the best split for each node.
   *    Each worker returns: For each active node, best split + info gain
   *  - The splits across workers are aggregated to the driver.
   * @param partitionInfos
   * @param labelsBc
   * @param strategy
   * @return
   */
  private[impl] def computeBestSplits(
      partitionInfos: RDD[PartitionInfo],
      labelsBc: Broadcast[Array[Double]],
      strategy: Strategy): Array[(Split, InfoGainStats)] = {
    // On each partition, for each feature on the partition, select the best split for each node.
    // This will use:
    //  - groupedColStore (the features)
    //  - partitionInfos (the node -> instance mapping)
    //  - labelsBc (the labels column)
    // Each worker returns:
    //   for each active node, best split + info gain
    val partBestSplitsAndGains: RDD[Array[(Split, InfoGainStats)]] = partitionInfos.map {
      case PartitionInfo(columns: Array[FeatureVector], nodeOffsets: Array[Int], activeNodes: BitSet) =>
        myPrintln(s"(W) computeBestSplits(): activeNodes=${activeNodes.iterator.mkString(",")}")
        val localLabels = labelsBc.value
        // Iterate over the active nodes in the current level.
        activeNodes.iterator.map { nodeIndexInLevel: Int =>
          myPrintln(s"\t ~~> nodeIndexInLevel=$nodeIndexInLevel, nodeOffsets.length=${nodeOffsets.length}")
          val fromOffset = nodeOffsets(nodeIndexInLevel)
          val toOffset = nodeOffsets(nodeIndexInLevel + 1)
          val splitsAndStats =
            columns.map { col =>
              chooseSplit(col, localLabels, fromOffset, toOffset,
                createImpurityAggregator(strategy), strategy.minInfoGain)
            }
          // We use Iterator and flatMap to handle empty partitions.
          splitsAndStats.maxBy(_._2.gain)
        }.toArray
    }

    // TODO: treeReduce
    // Aggregate best split for each active node.
    partBestSplitsAndGains.reduce { case (splitsGains1, splitsGains2) =>
      splitsGains1.zip(splitsGains2).map { case ((split1, gain1), (split2, gain2)) =>
        if (gain1.gain >= gain2.gain) {
          (split1, gain1)
        } else {
          (split2, gain2)
        }
      }
    }
  }

  /**
   * On driver: Grow tree based on chosen splits, and compute new set of active nodes.
   * @param oldPeriphery  Old periphery of active nodes.
   * @param bestSplitsAndGains  Best (split, gain) pairs, which can be zipped with the old
   *                            periphery.
   * @param minInfoGain  Threshold for min info gain required to split a node.
   * @return  New active node periphery
   */
  private[impl] def computeActiveNodePeriphery(
      oldPeriphery: Array[LearningNode],
      bestSplitsAndGains: Array[(Split, InfoGainStats)],
      minInfoGain: Double): Array[LearningNode] = {
    bestSplitsAndGains.zipWithIndex.flatMap {
      case ((split, stats), nodeIdx) =>
        val node = oldPeriphery(nodeIdx)
        myPrintln(s"(D) nodeIdx: $nodeIdx, gain: ${stats.gain}")
        if (stats.gain > minInfoGain) {
          // TODO: Add prediction probability once that is added properly to trees
          node.predictionStats = new OldPredict(stats.prediction, -1)
          node.impurity = stats.impurity
          node.leftChild =
            Some(LearningNode(node.id * 2, stats.leftPredict, stats.leftImpurity, isLeaf = false)) // TODO: remove node id
          node.rightChild =
            Some(LearningNode(node.id * 2 + 1, stats.rightPredict, stats.rightImpurity, isLeaf = false)) // TODO: remove node id
          node.split = Some(split)
          node.stats = Some(stats.toOld)
          myPrintln(s"(D) DRIVER splitting node id=${node.id}: nodeIdx=$nodeIdx, gain=${stats.gain}")
          Iterator(node.leftChild.get, node.rightChild.get)
        } else {
          node.isLeaf = true
          Iterator()
        }
    }
  }

  /**
   * Aggregate bit vector (1 bit/instance) indicating whether each instance goes left/right.
   * - Send chosen splits to workers.
   * - Each worker creates part of the bit vector corresponding to the splits it created.
   * - Aggregate the partial bit vectors to create one vector (of length numRows).
   *   Correction: Aggregate only the pieces of that vector corresponding to instances at
   *   active nodes.
   * @param partitionInfos  RDD with feature data, plus current status metadata
   * @param bestSplitsAndGains  One (split, gain stats) pair per active node
   * @return Array of bit vectors, ordered by offset ranges
   */
  private[impl] def collectBitVectors(
      partitionInfos: RDD[PartitionInfo],
      bestSplitsAndGains: Array[(Split, InfoGainStats)]): Array[BitSubvector] = {
    val bestSplitsBc: Broadcast[Array[Split]] =
      partitionInfos.sparkContext.broadcast(bestSplitsAndGains.map(_._1))
    val workerBitSubvectors: RDD[Array[BitSubvector]] = partitionInfos.map {
      case PartitionInfo(columns: Array[FeatureVector], nodeOffsets: Array[Int],
                         activeNodes: BitSet) =>
        val localBestSplits: Array[Split] = bestSplitsBc.value
        // localFeatureIndex[feature index] = index into PartitionInfo.columns
        val localFeatureIndex: Map[Int, Int] = columns.map(_.featureIndex).zipWithIndex.toMap
        activeNodes.iterator.zip(localBestSplits.iterator).flatMap {
          case (nodeIndexInLevel: Int, split: Split) =>
            if (localFeatureIndex.contains(split.featureIndex)) {
              // This partition has the column (feature) used for this split.
              val fromOffset = nodeOffsets(nodeIndexInLevel)
              val toOffset = nodeOffsets(nodeIndexInLevel + 1)
              val colIndex: Int = localFeatureIndex(split.featureIndex)
              myPrintln(s"(W) collectBitVectors(): fromOffset=$fromOffset, toOffset=$toOffset, colIndex=$colIndex, nodeIndexInLevel=$nodeIndexInLevel")
              Iterator(bitSubvectorFromSplit(columns(colIndex), fromOffset, toOffset, split))
            } else {
              Iterator()
            }
        }.toArray
    }
    val aggBitVectors: Array[BitSubvector] = workerBitSubvectors.reduce(BitSubvector.merge)
    bestSplitsBc.unpersist()
    aggBitVectors
  }

  /**
   *
   * @param col
   * @param labels
   * @param fromOffset
   * @param toOffset
   * @return
   */
  private[impl] def chooseSplit(
      col: FeatureVector,
      labels: Array[Double],
      fromOffset: Int,
      toOffset: Int,
      impurityAgg: ImpurityAggregatorSingle,
      minInfoGain: Double): (Split, InfoGainStats) = {
    val featureIndex = col.featureIndex
    val valuesForNode = col.values.view.slice(fromOffset, toOffset)
    val labelsForNode = col.indices.view.slice(fromOffset, toOffset).map(labels.apply)
    myPrintln(s"(W) chooseSplit: feature=${col.featureIndex}, vals=${valuesForNode.mkString("(",",",")")}," +
      s" labels=${labelsForNode.mkString("(",",",")")}," +
      s" inds=${col.indices.view.slice(fromOffset, toOffset).mkString("(",",",")")}")
    impurityAgg.clear()
    val fullImpurityAgg = impurityAgg.deepCopy()
    labelsForNode.foreach(fullImpurityAgg.update(_, 1.0))
    col.featureType match {
      case FeatureType.Categorical =>
        chooseCategoricalSplit(col.featureIndex, valuesForNode, labelsForNode, impurityAgg,
          fullImpurityAgg, minInfoGain)
      case FeatureType.Continuous =>
        chooseContinuousSplit(col.featureIndex, valuesForNode, labelsForNode, impurityAgg,
          fullImpurityAgg, minInfoGain)
    }
  }

  private[impl] def chooseCategoricalSplit(
      featureIndex: Int,
      values: Seq[Double],
      labels: Seq[Double],
      leftImpurityAgg: ImpurityAggregatorSingle,
      rightImpurityAgg: ImpurityAggregatorSingle,
      minInfoGain: Double): (Split, InfoGainStats) = ???

  /**
   * Choose splitting rule: feature value <= threshold
   */
  private[impl] def chooseContinuousSplit(
      featureIndex: Int,
      values: Seq[Double],
      labels: Seq[Double],
      leftImpurityAgg: ImpurityAggregatorSingle,
      rightImpurityAgg: ImpurityAggregatorSingle,
      minInfoGain: Double): (Split, InfoGainStats) = {
    val prediction = leftImpurityAgg.getCalculator.getPredict

    var bestThreshold: Double = Double.NegativeInfinity
    var bestLeftImpurityAgg = leftImpurityAgg.deepCopy()
    var bestGain: Double = 0.0
    val fullImpurity = rightImpurityAgg.getCalculator.calculate()
    var leftCount: Double = 0.0
    var rightCount: Double = rightImpurityAgg.getCount
    val fullCount: Double = rightCount
    myPrintln(s"(W) chooseContinuousSplit(): featureIndex=$featureIndex, values=${values.mkString(",")}")
    var currentThreshold = values.headOption.getOrElse(bestThreshold)
    values.zip(labels).foreach { case (value, label) =>
      if (value != currentThreshold) {
        // Check gain
        val leftWeight = leftCount / fullCount
        val rightWeight = rightCount / fullCount
        val leftImpurity = leftImpurityAgg.getCalculator.calculate()
        val rightImpurity = rightImpurityAgg.getCalculator.calculate()
        val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
        myPrintln(s"\t --> gain=$gain, fullImpurity=$fullImpurity, leftWeight=$leftWeight," +
          s" leftImpurity=$leftImpurity, rightWeight=$rightWeight, rightImpurity=$rightImpurity")
        if (gain > bestGain && gain > minInfoGain) {
          bestThreshold = currentThreshold
          leftImpurityAgg.stats.copyToArray(bestLeftImpurityAgg.stats)
          bestGain = gain
          myPrintln(s"\t  -> best update: bestThreshold=$bestThreshold, bestGain=$bestGain")
        }
        currentThreshold = value
      }
      // Move this instance from right to left side of split.
      leftImpurityAgg.update(label, 1.0)
      rightImpurityAgg.update(label, -1.0)
      leftCount += 1.0
      rightCount -= 1.0
    }
    myPrintln()

    val leftImpurity = bestLeftImpurityAgg.getCalculator.calculate()
    val bestRightImpurityAgg =
      leftImpurityAgg.deepCopy().add(rightImpurityAgg).subtract(bestLeftImpurityAgg)
    val rightImpurity = bestRightImpurityAgg.getCalculator.calculate()
    val bestGainStats = new InfoGainStats(prediction.predict, bestGain, fullImpurity,
      leftImpurity, rightImpurity, bestLeftImpurityAgg.getCalculator.getPredict,
      bestRightImpurityAgg.getCalculator.getPredict)
    (new ContinuousSplit(featureIndex, bestThreshold), bestGainStats)
  }

  /**
   * Feature vector types are based on (feature type, representation).
   * The feature type can be continuous or categorical.
   *
   * Features are sorted by value, so we must store indices + values.
   * These values are currently stored in a dense representation only.
   * TODO: Support sparse storage (to optimize deeper levels of the tree), and maybe compressed
   *       storage (to optimize upper levels of the tree).
   */
  private[impl] class FeatureVector(
      val featureIndex: Int,
      val featureType: FeatureType,
      val values: Array[Double],
      val indices: Array[Int])
    extends Serializable {

    /** For debugging */
    override def toString: String = {
      "  FeatureVector(" +
        s"    featureIndex: $featureIndex,\n" +
        s"    featureType: $featureType,\n" +
        s"    values: ${values.mkString(", ")},\n" +
        s"    indices: ${indices.mkString(", ")},\n" +
        "  )"
    }

    def deepCopy(): FeatureVector =
      new FeatureVector(featureIndex, featureType, values.clone(), indices.clone())

    override def equals(other: Any): Boolean = {
      other match {
        case o: FeatureVector =>
          featureIndex == o.featureIndex && featureType == o.featureType &&
          values.sameElements(o.values) && indices.sameElements(o.indices)
        case _ => false
      }
    }
  }

  private[impl] object FeatureVector {
    /** Store column sorted by feature values. */
    def fromOriginal(
        featureIndex: Int,
        featureType: FeatureType,
        featureVector: Vector): FeatureVector = {
      val (values, indices) = featureVector.toArray.zipWithIndex.sorted.unzip
      new FeatureVector(featureIndex, featureType, values.toArray, indices.toArray)
    }
  }

  /**
   * For a given feature, for a given node, apply a split and return a bit vector indicating the
   * outcome of the split for each instance at that node.
   *
   * @param col  Column for feature
   * @param fromOffset  Start offset in col for the node
   * @param toOffset  End offset in col for the node
   * @param split  Split to apply to instances at this node.
   * @return  Bits indicating splits for instances at this node.
   *          These bits are sorted by the row indices, in order to guarantee an ordering
   *          understood by all workers.
   *          Thus, the bit indices used are based on 2-level sorting: first by node, and
   *          second by sorted row indices within the node's rows.
   *          bit[index in sorted array of row indices] = false for left, true for right
   */
  private[impl] def bitSubvectorFromSplit(
      col: FeatureVector,
      fromOffset: Int,
      toOffset: Int,
      split: Split): BitSubvector = {
    val nodeRowIndices = col.indices.view.slice(fromOffset, toOffset).toArray
    val nodeRowValues = col.values.view.slice(fromOffset, toOffset).toArray
    val nodeRowValuesSortedByIndices = nodeRowIndices.zip(nodeRowValues).sortBy(_._1).map(_._2)
    val bitv = new BitSubvector(fromOffset, toOffset)
    myPrint(s"(W)   bitSubvectorFromSplit(): nodeRowIndices=${nodeRowIndices.mkString("(",",",")")}," +
      s" nodeRowValues=${nodeRowValues.mkString("(",",",")")}")
    nodeRowValuesSortedByIndices.zipWithIndex.foreach { case (value, i) =>
      if (!split.shouldGoLeft(value)) {
        bitv.set(fromOffset + i)
        myPrint("R ")
      } else {
        myPrint("L ")
      }
    }
    myPrintln()
    bitv
  }

  /**
   * Intermediate data stored on each partition during learning.
   *
   * Node indexing for nodeOffsets, activeNodes:
   * Nodes are indexed left-to-right along the periphery of the tree, with 0-based indices.
   * The periphery is the set of leaf nodes (active and inactive).
   *
   * @param columns  Subset of columns (features) stored in this partition.
   *                 Each column is sorted first by nodes (left-to-right along the tree periphery);
   *                 all columns share this first level of sorting.
   *                 Within each node's group, each column is sorted based on feature value;
   *                 this second level of sorting differs across columns.
   * @param nodeOffsets  Offsets into the columns indicating the first level of sorting (by node).
   *                     The rows corresponding to node i are in the range
   *                     [nodeOffsets(i), nodeOffsets(i+1)).
   * @param activeNodes  Nodes which are active (still being split).
   *                     Inactive nodes are known to be leafs in the final tree.
   *                     TODO: Should this (and even nodeOffsets) not be stored in PartitionInfo,
   *                           but instead on the driver?
   */
  private[impl] case class PartitionInfo(
      columns: Array[FeatureVector],
      nodeOffsets: Array[Int],
      activeNodes: BitSet)
    extends Serializable {

    /** For debugging */
    override def toString: String = {
      "PartitionInfo(" +
        "  columns: {\n" +
        columns.mkString(",\n") +
        "  },\n" +
        s"  nodeOffsets: ${nodeOffsets.mkString(", ")},\n" +
        s"  activeNodes: ${activeNodes.iterator.mkString(", ")},\n" +
        ")\n"
    }

    /**
     * Update columns and nodeOffsets for the next level of the tree.
     *
     * Update columns:
     *   For each column,
     *     For each (previously) active node,
     *       Sort corresponding range of instances based on bit vector.
     * Update nodeOffsets, activeNodes:
     *   Split offsets for nodes which split (which can be identified using the bit vector).
     *
     * @param bitVectors  Bit vectors encoding splits for the next level of the tree.
     *                    These must follow a 2-level ordering, where the first level is by node
     *                    and the second level is by row index.
     *                    bitVector(i) = false iff instance i goes to the left child.
     *                    For instances at inactive (leaf) nodes, the value can be arbitrary.
     * @return Updated partition info
     */
    def update(bitVectors: Array[BitSubvector], newNumNodeOffsets: Int): PartitionInfo = {
      val newColumns = columns.map { oldCol =>
        val col = oldCol.deepCopy()
        var curBitVecIdx = 0
        activeNodes.iterator.foreach { nodeIdx =>
          val from = nodeOffsets(nodeIdx)
          val to = nodeOffsets(nodeIdx + 1)
          // Note: Each node is guaranteed to be covered within 1 bit vector.
          if (bitVectors(curBitVecIdx).to <= from) curBitVecIdx += 1
          val curBitVector = bitVectors(curBitVecIdx)
          // Sort range [from, to) based on indices.  This is required to match the bit vector
          // across all workers.  See [[bitSubvectorFromSplit]] for details.
          val rangeIndices = col.indices.view.slice(from, to).toArray
          val rangeValues = col.values.view.slice(from, to).toArray
          val sortedRange = rangeIndices.zip(rangeValues).sortBy(_._1)
          // Sort range [from, to) based on bit vector.
          sortedRange.zipWithIndex.map { case ((idx, value), i) =>
            val bit = curBitVector.get(from + i)
            // TODO: In-place merge, rather than general sort.
            (bit, value, idx)
          }.sorted.zipWithIndex.foreach { case ((bit, value, idx), i) =>
            col.values(from + i) = value
            col.indices(from + i) = idx
          }
        }
        col
      }

      // Create a 2-level representation of the new nodeOffsets (to be flattened).
      myPrintln(s"(W) initial nodeOffsets: ${nodeOffsets.mkString("(",",",")")}")
      val newNodeOffsets = nodeOffsets.map(Array(_))
      var curBitVecIdx = 0
      activeNodes.iterator.foreach { nodeIdx =>
        val from = nodeOffsets(nodeIdx)
        val to = nodeOffsets(nodeIdx + 1)
        if (bitVectors(curBitVecIdx).to <= from) curBitVecIdx += 1
        val curBitVector = bitVectors(curBitVecIdx)
        assert(curBitVector.from <= from && to <= curBitVector.to)
        // Count number of values splitting to left vs. right
        val numRight = Range(from, to).count(curBitVector.get)
        val numLeft = to - from - numRight
        if (numLeft != 0 && numRight != 0) {
          // node is split
          val oldOffset = newNodeOffsets(nodeIdx).head
          myPrintln(s"(W) WORKER splitting node: nodeIdx=$nodeIdx, oldOffset=$oldOffset, numLeft=$numLeft, numRight=$numRight")
          newNodeOffsets(nodeIdx) = Array(oldOffset, oldOffset + numLeft)
        } else {
          myPrintln(s"(W) WORKER NOT splitting node: nodeIdx=$nodeIdx, oldOffset=${newNodeOffsets(nodeIdx).head}, numLeft=$numLeft, numRight=$numRight")
        }
      }

      myPrintln(s"(W) newNodeOffsets: ${newNodeOffsets.map(_.mkString("(",",",")")).mkString("[", ", ", "]")}")
      assert(newNodeOffsets.map(_.length).sum == newNumNodeOffsets,
        s"(W) newNodeOffsets total size: ${newNodeOffsets.map(_.length).sum}," +
          s" newNumNodeOffsets: $newNumNodeOffsets")

      // Identify the new activeNodes based on the 2-level representation of the new nodeOffsets.
      val newActiveNodes = new BitSet(newNumNodeOffsets - 1)
      var newNodeOffsetsIdx = 0
      newNodeOffsets.foreach { offsets =>
        if (offsets.length == 2) {
          newActiveNodes.set(newNodeOffsetsIdx)
          newActiveNodes.set(newNodeOffsetsIdx + 1)
          newNodeOffsetsIdx += 2
        } else {
          newNodeOffsetsIdx += 1
        }
      }

      PartitionInfo(newColumns, newNodeOffsets.flatten, newActiveNodes)
    }
  }

}
