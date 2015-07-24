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
object AltDT extends Logging {

  private def createImpurityAggregator(strategy: Strategy): ImpurityAggregatorSingle = {
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

  def trainImpl(input: RDD[LabeledPoint], strategy: Strategy): Node = {
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
    // Group columns together into one array of columns per partition.
    val groupedColStore: RDD[Array[FeatureVector]] = colStore.mapPartitions { iterator =>
      val groupedCols = new ArrayBuffer[FeatureVector]
      iterator.foreach(groupedCols += _)
      Iterator(groupedCols.toArray)
    }
    groupedColStore.persist(StorageLevel.MEMORY_AND_DISK)

    // Initialize partitions with 1 partition (i.e., each instance at the root node).
    var partitionInfos: RDD[PartitionInfo] = groupedColStore.map { groupedCols =>
      val initActive = new BitSet(1)
      initActive.setUntil(1)
      new PartitionInfo(groupedCols, Array[Int](0, numRows), initActive)
    }

    // Initialize model.
    // Note: We do not use node indices.
    val rootNode = LearningNode.emptyNode(-1)
    // Active nodes (still being split), updated each iteration
    var activeNodePeriphery: Array[LearningNode] = Array(rootNode)
    var numNodeOffsets: Int = 2

    // Iteratively learn, one level of the tree at a time.
    var currentLevel = 0
    while (currentLevel < strategy.maxDepth) {

      // On each partition, for each feature on the partition, select the best split for each node.
      // This will use:
      //  - groupedColStore (the features)
      //  - partitionInfos (the node -> instance mapping)
      //  - labelsBc (the labels column)
      // Each worker returns:
      //   for each active node, best split + info gain
      val partBestSplitsAndGains: RDD[Array[(Split, InfoGainStats)]] = partitionInfos.map {
        case PartitionInfo(columns: Array[FeatureVector], nodeOffsets: Array[Int], activeNodes: BitSet) =>
          val localLabels = labelsBc.value
          // Iterate over the active nodes in the current level.
          activeNodes.iterator.map { nodeIndexInLevel: Int =>
            println(s"nodeIndexInLevel=$nodeIndexInLevel, nodeOffsets.length=${nodeOffsets.length}")
            val fromOffset = nodeOffsets(nodeIndexInLevel)
            val toOffset = nodeOffsets(nodeIndexInLevel + 1)
            val splitsAndStats =
              columns.map { col =>
                chooseSplit(col, localLabels, fromOffset, toOffset,
                  createImpurityAggregator(strategy), strategy.minInfoGain)
              }
            // nodeIndexInLevel -> splitsAndStats.maxBy(_._2.gain)
            splitsAndStats.maxBy(_._2.gain)
          }.toArray
      }

      // Aggregate best split for each active node.
      val bestSplitsAndGains: Array[(Split, InfoGainStats)] =
        partBestSplitsAndGains.reduce { case (splitsGains1, splitsGains2) =>
          splitsGains1.zip(splitsGains2).map { case ((split1, gain1), (split2, gain2)) =>
            if (gain1.gain >= gain2.gain) {
              (split1, gain1)
            } else {
              (split2, gain2)
            }
          }
        }

      // Update current model and node periphery.
      // Note: This flatMap has side effects (on the model).
      activeNodePeriphery = bestSplitsAndGains.zipWithIndex.flatMap {
        case ((split, stats), nodeIdx) =>
          val node = activeNodePeriphery(nodeIdx)
          node.predictionStats = new OldPredict(stats.prediction, -1)
          node.impurity = stats.impurity
          println(s"nodeIdx: $nodeIdx, gain: ${stats.gain}")
          if (stats.gain > strategy.getMinInfoGain) {
            // TODO: Add prediction probability once that is added properly to trees
            node.leftChild =
              Some(LearningNode(-1, stats.leftPredict, stats.leftImpurity, isLeaf = false))
            node.rightChild =
              Some(LearningNode(-1, stats.rightPredict, stats.rightImpurity, isLeaf = false))
            node.split = Some(split)
            node.stats = Some(stats.toOld)
            Iterator(node.leftChild.get, node.rightChild.get)
          } else {
            node.isLeaf = true
            Iterator()
          }
      }
      println(s"activeNodePeriphery.length: ${activeNodePeriphery.length}")
      // We keep all old nodeOffsets and add one for each node split.
      // Each node split adds 2 nodes to activeNodePeriphery.
      numNodeOffsets = numNodeOffsets + activeNodePeriphery.length / 2
      println(s"numNodeOffsets: $numNodeOffsets")

      // TODO: Check to make sure we split something, and stop otherwise.
      val doneLearning = currentLevel >= strategy.maxDepth

      if (!doneLearning) {
        // Aggregate bit vector (1 bit/instance) indicating whether each instance goes left/right.
        //  - Send chosen splits to workers.
        //  - Each worker creates part of the bit vector corresponding to the splits it created.
        //  - Aggregate the partial bit vectors to create one vector (of length numRows).
        //    Correction: Aggregate only the pieces of that vector corresponding to instances at
        //    active nodes.
        val bestSplitsBc: Broadcast[Array[Split]] =
          input.sparkContext.broadcast(bestSplitsAndGains.map(_._1))
        val workerBitSubvectors: RDD[Array[BitSubvector]] = partitionInfos.map {
          case PartitionInfo(columns: Array[FeatureVector], nodeOffsets: Array[Int], activeNodes: BitSet) =>
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
                  Iterator(bitSubvectorFromSplit(columns(colIndex), fromOffset, toOffset, split))
                } else {
                  Iterator()
                }
            }.toArray
        }
        val aggBitVectors: Array[BitSubvector] = workerBitSubvectors.reduce(BitSubvector.merge)
        bestSplitsBc.unpersist()

        // Broadcast aggregated bit vectors.  On each partition, update instance--node map.
        val aggBitVectorsBc = input.sparkContext.broadcast(aggBitVectors)
        partitionInfos = partitionInfos.map { partitionInfo =>
          partitionInfo.update(aggBitVectorsBc.value, numNodeOffsets)
        }
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

  // NOTE: The order of col does not correspond to the order of labels.  Use the index in col.
  /**
   *
   * @param col
   * @param labels
   * @param fromOffset
   * @param toOffset
   * @return
   */
  private def chooseSplit(
      col: FeatureVector,
      labels: Array[Double],
      fromOffset: Int,
      toOffset: Int,
      impurityAgg: ImpurityAggregatorSingle,
      minInfoGain: Double): (Split, InfoGainStats) = {
    val featureIndex = col.featureIndex
    val valuesForNode = col.values.view.slice(fromOffset, toOffset)
    val labelsForNode = col.indices.view.slice(fromOffset, toOffset).map(labels.apply)
    impurityAgg.clear()
    val fullImpurityAgg = impurityAgg.deepCopy()
    labels.foreach(fullImpurityAgg.update(_, 1.0))
    col.featureType match {
      case FeatureType.Categorical =>
        chooseCategoricalSplit(col.featureIndex, valuesForNode, labelsForNode, impurityAgg, fullImpurityAgg, minInfoGain)
      case FeatureType.Continuous =>
        chooseContinuousSplit(col.featureIndex, valuesForNode, labelsForNode, impurityAgg, fullImpurityAgg, minInfoGain)
    }
  }

  private def chooseCategoricalSplit(
      featureIndex: Int,
      values: Seq[Double],
      labels: Seq[Double],
      leftImpurityAgg: ImpurityAggregatorSingle,
      rightImpurityAgg: ImpurityAggregatorSingle,
      minInfoGain: Double): (Split, InfoGainStats) = ???

  /**
   * Choose splitting rule: feature value <= threshold
   */
  private def chooseContinuousSplit(
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
    println(s"\nfeatureIndex: $featureIndex")
    values.zip(labels).foreach { case (value, label) =>
      // Move this instance from right to left side of split.
      leftImpurityAgg.update(label, 1.0)
      rightImpurityAgg.update(label, -1.0)
      leftCount += 1.0
      rightCount -= 1.0
      val leftWeight = leftCount / fullCount
      val rightWeight = rightCount / fullCount
      // Check gain
      val leftImpurity = leftImpurityAgg.getCalculator.calculate()
      val rightImpurity = rightImpurityAgg.getCalculator.calculate()
      val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
      print(s" gain=$gain ")
      if (gain > bestGain && gain > minInfoGain) {
        bestThreshold = value
        leftImpurityAgg.stats.copyToArray(bestLeftImpurityAgg.stats)
        bestGain = gain
      }
    }
    println()

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
  private class FeatureVector(
      val featureIndex: Int,
      val featureType: FeatureType,
      val values: Array[Double],
      val indices: Array[Int])
    extends Serializable {

    def deepCopy(): FeatureVector =
      new FeatureVector(featureIndex, featureType, values.clone(), indices.clone())
  }

  private object FeatureVector {
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
  def bitSubvectorFromSplit(
      col: FeatureVector,
      fromOffset: Int,
      toOffset: Int,
      split: Split): BitSubvector = {
    val nodeRowIndices = col.indices.view.slice(fromOffset, toOffset).toArray
    val nodeRowValues = col.values.view.slice(fromOffset, toOffset).toArray
    val nodeRowValuesSortedByIndices = nodeRowIndices.zip(nodeRowValues).sortBy(_._1).map(_._2)
    val bitv = new BitSubvector(fromOffset, toOffset)
    nodeRowValuesSortedByIndices.zipWithIndex.foreach { case (value, i) =>
      if (!split.shouldGoLeft(value)) bitv.set(fromOffset + i)
    }
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
  private case class PartitionInfo(
      columns: Array[FeatureVector],
      nodeOffsets: Array[Int],
      activeNodes: BitSet)
    extends Serializable {

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
      val newNodeOffsets = nodeOffsets.map(Array(_))
      var curBitVecIdx = 0
      activeNodes.iterator.foreach { nodeIdx =>
        val from = nodeOffsets(nodeIdx)
        val to = nodeOffsets(nodeIdx + 1)
        if (bitVectors(curBitVecIdx).to <= from) curBitVecIdx += 1
        val curBitVector = bitVectors(curBitVecIdx)
        // Count number of values splitting to left vs. right
        val numRight = Range(from, to).count(curBitVector.get)
        val numLeft = to - from - numRight
        if (numRight != 0 && numRight != 0) {
          // node is split
          val oldOffset = newNodeOffsets(nodeIdx + 1).head
          newNodeOffsets(nodeIdx + 1) = Array(oldOffset, oldOffset + numLeft)
        }
      }

      assert(newNodeOffsets.map(_.length).sum == newNumNodeOffsets,
        s"newNodeOffsets total size: ${newNodeOffsets.map(_.length).sum}," +
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

      new PartitionInfo(newColumns, newNodeOffsets.flatten, newActiveNodes)
    }
  }

}
