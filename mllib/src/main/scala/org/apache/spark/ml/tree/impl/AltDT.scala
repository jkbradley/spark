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

import org.apache.spark.Logging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.TreeUtil._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impurity.{Entropy, Gini, Impurity, Variance}
import org.apache.spark.mllib.tree.model.ImpurityStats
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

  private[impl] class AltDTMetadata(
      val numClasses: Int,
      val maxBins: Int,
      val minInfoGain: Double,
      val impurity: Impurity) extends Serializable {

    private val maxCategoriesForUnorderedFeature =
      ((math.log(maxBins / 2 + 1) / math.log(2.0)) + 1).floor.toInt

    def isClassification: Boolean = numClasses >= 2

    def isMulticlass: Boolean = numClasses > 2

    /**
     * Indicates whether a categorical feature should be treated as unordered.
     *
     * TODO(SPARK-9957): If a categorical feature has only 1 category, we treat it as continuous.
     *                   Later, handle this properly by filtering out those features.
     */
    def isUnorderedFeature(numCategories: Int): Boolean = {
      // Decide if some categorical features should be treated as unordered features,
      //  which require 2 * ((1 << numCategories - 1) - 1) bins.
      // We do this check with log values to prevent overflows in case numCategories is large.
      // The last inequality is equivalent to: 2 * ((1 << numCategories - 1) - 1) <= maxBins
      isMulticlass && numCategories > 1 &&
        numCategories <= maxCategoriesForUnorderedFeature
    }

    def createImpurityAggregator(): ImpurityAggregatorSingle = {
      impurity match {
        case Entropy => new EntropyAggregatorSingle(numClasses)
        case Gini => new GiniAggregatorSingle(numClasses)
        case Variance => new VarianceAggregatorSingle
      }
    }
  }

  private[impl] object AltDTMetadata {
    def fromStrategy(strategy: Strategy): AltDTMetadata = new AltDTMetadata(strategy.numClasses,
      strategy.maxBins, strategy.minInfoGain, strategy.impurity)
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
    impl.RandomForest.finalizeTree(rootNode, strategy.algo, strategy.numClasses, parentUID)
  }

  private[impl] def trainImpl(input: RDD[LabeledPoint], strategy: Strategy): Node = {
    val metadata = AltDTMetadata.fromStrategy(strategy)

    // The case with 1 node (depth = 0) is handled separately.
    // This allows all iterations in the depth > 0 case to use the same code.
    // TODO: Check that learning works when maxDepth > 0 but learning stops at 1 node (because of
    //       other parameters).
    if (strategy.maxDepth == 0) {
      val impurityAggregator: ImpurityAggregatorSingle =
        input.aggregate(metadata.createImpurityAggregator())(
          (agg, lp) => agg.update(lp.label, 1.0),
          (agg1, agg2) => agg1.add(agg2))
      val impurityCalculator = impurityAggregator.getCalculator
      return new LeafNode(impurityCalculator.getPredict.predict, impurityCalculator.calculate(),
        impurityCalculator)
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
      val featureArity: Int = strategy.categoricalFeaturesInfo.getOrElse(featureIndex, 0)
      FeatureVector.fromOriginal(featureIndex, featureArity, col)
    }
    // Group columns together into one array of columns per partition.
    // TODO: Test avoiding this grouping, and see if it matters.
    val groupedColStore: RDD[Array[FeatureVector]] = colStore.mapPartitions {
      iterator: Iterator[FeatureVector] =>
        if (iterator.nonEmpty) Iterator(iterator.toArray) else Iterator()
    }
    groupedColStore.persist(StorageLevel.MEMORY_AND_DISK)

    // Initialize partitions with 1 node (each instance at the root node).
    var partitionInfos: RDD[PartitionInfo] = groupedColStore.map { groupedCols =>
      val initActive = new BitSet(1)
      initActive.set(0)
      new PartitionInfo(groupedCols, Array[Int](0, numRows), initActive)
    }

    // Initialize model.
    // Note: We do not use node indices.
    val rootNode = LearningNode.emptyNode(1) // TODO: remove node id
    // Active nodes (still being split), updated each iteration
    var activeNodePeriphery: Array[LearningNode] = Array(rootNode)
    var numNodeOffsets: Int = 2

    // Iteratively learn, one level of the tree at a time.
    var currentLevel = 0
    var doneLearning = false
    while (currentLevel < strategy.maxDepth && !doneLearning) {
      // Compute best split for each active node.
      val bestSplitsAndGains: Array[(Option[Split], ImpurityStats)] =
        computeBestSplits(partitionInfos, labelsBc, metadata)
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
      // We keep all old nodeOffsets and add one for each node split.
      // Each node split adds 2 nodes to activeNodePeriphery.
      // TODO: Should this be calculated after filtering for impurity??
      numNodeOffsets = numNodeOffsets + activeNodePeriphery.length / 2

      // Filter active node periphery by impurity.
      val estimatedRemainingActive = activeNodePeriphery.count(_.stats.impurity > 0.0)

      // TODO: Check to make sure we split something, and stop otherwise.
      doneLearning = currentLevel + 1 >= strategy.maxDepth || estimatedRemainingActive == 0

      if (!doneLearning) {
        // Aggregate bit vector (1 bit/instance) indicating whether each instance goes left/right.
        val aggBitVectors: Array[BitSubvector] =
          collectBitVectors(partitionInfos, bestSplitsAndGains.map(_._1))

        // Broadcast aggregated bit vectors.  On each partition, update instance--node map.
        val aggBitVectorsBc = input.sparkContext.broadcast(aggBitVectors)
        val newPartitionInfos = partitionInfos.map { partitionInfo =>
          partitionInfo.update(aggBitVectorsBc.value, numNodeOffsets)
        }
        // TODO: remove.  For some reason, this is needed to make things work.
        // Probably messing up somewhere above...
        newPartitionInfos.cache().count()
        partitionInfos = newPartitionInfos

        // TODO: unpersist aggBitVectorsBc after action.
      }

      currentLevel += 1
    }

    // Done with learning
    groupedColStore.unpersist()
    labelsBc.unpersist()
    rootNode.toNode
  }

  /**
   * Given the arity of a categorical feature (arity = number of categories),
   * return the number of bins for the feature if it is to be treated as an unordered feature.
   * There is 1 split for every partitioning of categories into 2 disjoint, non-empty sets;
   * there are math.pow(2, arity - 1) - 1 such splits.
   * Each split has 2 corresponding bins.
   */
  def numUnorderedBins(arity: Int): Int = 2 * ((1 << arity - 1) - 1)

  /**
   * Find the best splits for all active nodes.
   *  - On each partition, for each feature on the partition, select the best split for each node.
   *    Each worker returns: For each active node, best split + info gain
   *  - The splits across workers are aggregated to the driver.
   * @param partitionInfos
   * @param labelsBc
   * @param metadata
   * @return  Array over active nodes of (best split, impurity stats for split),
   *          where the split is None if no useful split exists
   */
  private[impl] def computeBestSplits(
      partitionInfos: RDD[PartitionInfo],
      labelsBc: Broadcast[Array[Double]],
      metadata: AltDTMetadata): Array[(Option[Split], ImpurityStats)] = {
    // On each partition, for each feature on the partition, select the best split for each node.
    // This will use:
    //  - groupedColStore (the features)
    //  - partitionInfos (the node -> instance mapping)
    //  - labelsBc (the labels column)
    // Each worker returns:
    //   for each active node, best split + info gain,
    //     where the best split is None if no useful split exists
    val partBestSplitsAndGains: RDD[Array[(Option[Split], ImpurityStats)]] = partitionInfos.map {
      case PartitionInfo(columns: Array[FeatureVector], nodeOffsets: Array[Int],
          activeNodes: BitSet) =>
        val localLabels = labelsBc.value
        // Iterate over the active nodes in the current level.
        val toReturn = new Array[(Option[Split], ImpurityStats)](activeNodes.cardinality())
        val iter: Iterator[Int] = activeNodes.iterator
        var i = 0
        while (iter.hasNext) {
          val nodeIndexInLevel = iter.next
          val fromOffset = nodeOffsets(nodeIndexInLevel)
          val toOffset = nodeOffsets(nodeIndexInLevel + 1)
          val splitsAndStats =
            columns.map { col =>
              chooseSplit(col, localLabels, fromOffset, toOffset, metadata)
            }
          toReturn(i) = splitsAndStats.maxBy(_._2.gain)
          i += 1
        }
        toReturn
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
   *                            periphery.  These stats will be used to replace the stats in
   *                            any nodes which are split.
   * @param minInfoGain  Threshold for min info gain required to split a node.
   * @return  New active node periphery.
   *          If a node is split, then this method will update its fields.
   */
  private[impl] def computeActiveNodePeriphery(
      oldPeriphery: Array[LearningNode],
      bestSplitsAndGains: Array[(Option[Split], ImpurityStats)],
      minInfoGain: Double): Array[LearningNode] = {
    bestSplitsAndGains.zipWithIndex.flatMap {
      case ((split, stats), nodeIdx) =>
        val node = oldPeriphery(nodeIdx)
        if (split.nonEmpty && stats.gain > minInfoGain) {
          // TODO: remove node id
          node.leftChild = Some(LearningNode(node.id * 2, isLeaf = false,
            ImpurityStats(stats.leftImpurity, stats.leftImpurityCalculator)))
          node.rightChild = Some(LearningNode(node.id * 2 + 1, isLeaf = false,
            ImpurityStats(stats.rightImpurity, stats.rightImpurityCalculator)))
          node.split = split
          node.isLeaf = false
          node.stats = stats
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
   * @param bestSplits  Split for each active node, or None if that node will not be split
   * @return Array of bit vectors, ordered by offset ranges
   */
  private[impl] def collectBitVectors(
      partitionInfos: RDD[PartitionInfo],
      bestSplits: Array[Option[Split]]): Array[BitSubvector] = {
    val bestSplitsBc: Broadcast[Array[Option[Split]]] =
      partitionInfos.sparkContext.broadcast(bestSplits)
    val workerBitSubvectors: RDD[Array[BitSubvector]] = partitionInfos.map {
      case PartitionInfo(columns: Array[FeatureVector], nodeOffsets: Array[Int],
                         activeNodes: BitSet) =>
        val localBestSplits: Array[Option[Split]] = bestSplitsBc.value
        // localFeatureIndex[feature index] = index into PartitionInfo.columns
        val localFeatureIndex: Map[Int, Int] = columns.map(_.featureIndex).zipWithIndex.toMap
        activeNodes.iterator.zip(localBestSplits.iterator).flatMap {
          case (nodeIndexInLevel: Int, Some(split: Split)) =>
            if (localFeatureIndex.contains(split.featureIndex)) {
              // This partition has the column (feature) used for this split.
              val fromOffset = nodeOffsets(nodeIndexInLevel)
              val toOffset = nodeOffsets(nodeIndexInLevel + 1)
              val colIndex: Int = localFeatureIndex(split.featureIndex)
              Iterator(bitSubvectorFromSplit(columns(colIndex), fromOffset, toOffset, split))
            } else {
              Iterator()
            }
          case (nodeIndexInLevel: Int, None) =>
            // Do not create a BitSubvector when there is no split.
            // This requires PartitionInfo.update to handle missing BitSubvectors.
            Iterator()
        }.toArray
    }
    val aggBitVectors: Array[BitSubvector] = workerBitSubvectors.reduce(BitSubvector.merge)
    bestSplitsBc.unpersist()
    aggBitVectors
  }

  /**
   * Choose the best split for a feature at a node.
   *
   * TODO: Return null or None when the split is invalid, such as putting all instances on one
   *       child node.
   *
   * @param col
   * @param labels
   * @param fromOffset
   * @param toOffset
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.
   */
  private[impl] def chooseSplit(
      col: FeatureVector,
      labels: Array[Double],
      fromOffset: Int,
      toOffset: Int,
      metadata: AltDTMetadata): (Option[Split], ImpurityStats) = {
    val valuesForNode = col.values.view.slice(fromOffset, toOffset)
    val labelsForNode = col.indices.view.slice(fromOffset, toOffset).map(labels.apply)
    if (col.isCategorical) {
      if (metadata.isUnorderedFeature(col.featureArity)) {
        chooseUnorderedCategoricalSplit(col.featureIndex, valuesForNode, labelsForNode, metadata,
          col.featureArity)
      } else {
        chooseOrderedCategoricalSplit(col.featureIndex, valuesForNode, labelsForNode, metadata,
          col.featureArity)
      }
    } else {
      chooseContinuousSplit(col.featureIndex, valuesForNode, labelsForNode, metadata)
    }
  }

  /**
   * Find the best split for an ordered categorical feature at a single node.
   *
   * Algorithm:
   *  - For each category, compute a "centroid."
   *     - For multiclass classification, the centroid is the label impurity.
   *     - For binary classification and regression, the centroid is the average label.
   *  - Sort the centroids, and consider splits anywhere in this order.
   *    Thus, with K categories, we consider K - 1 possible splits.
   *
   * @param featureIndex  Index of feature being split.
   * @param values  Feature values at this node.  Sorted in increasing order.
   * @param labels  Labels corresponding to values, in the same order.
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.  The impurity stats maybe still be
   *          useful, so they are returned.
   */
  private[impl] def chooseOrderedCategoricalSplit(
      featureIndex: Int,
      values: Seq[Double],
      labels: Seq[Double],
      metadata: AltDTMetadata,
      featureArity: Int): (Option[Split], ImpurityStats) = {
    // TODO: Support high-arity features by using a single array to hold the stats.

    // aggStats(category) = label statistics for category
    val aggStats = Array.tabulate[ImpurityAggregatorSingle](featureArity)(
      _ => metadata.createImpurityAggregator())
    var i = 0
    val len = values.length
    while (i < len) {
      val cat = values(i)
      val label = labels(i)
      aggStats(cat.toInt).update(label)
      i += 1
    }

    // Compute centroids.  centroidsForCategories is a list: (category, centroid)
    val centroidsForCategories: Seq[(Int, Double)] = if (metadata.isMulticlass) {
      // For categorical variables in multiclass classification,
      // the bins are ordered by the impurity of their corresponding labels.
      Range(0, featureArity).map { case featureValue =>
        val categoryStats = aggStats(featureValue)
        val centroid = if (categoryStats.getCount != 0) {
          categoryStats.getCalculator.calculate()
        } else {
          Double.MaxValue
        }
        (featureValue, centroid)
      }
    } else if (metadata.isClassification) { // binary classification
      // For categorical variables in binary classification,
      // the bins are ordered by the centroid of their corresponding labels.
      Range(0, featureArity).map { case featureValue =>
        val categoryStats = aggStats(featureValue)
        val centroid = if (categoryStats.getCount != 0) {
          assert(categoryStats.stats.length == 2)
          (categoryStats.stats(1) - categoryStats.stats(0)) / categoryStats.getCount
        } else {
          Double.MaxValue
        }
        (featureValue, centroid)
      }
    } else { // regression
      // For categorical variables in regression,
      // the bins are ordered by the centroid of their corresponding labels.
      Range(0, featureArity).map { case featureValue =>
        val categoryStats = aggStats(featureValue)
        val centroid = if (categoryStats.getCount != 0) {
          categoryStats.getCalculator.predict
        } else {
          Double.MaxValue
        }
        (featureValue, centroid)
      }
    }

    logDebug("Centroids for categorical variable: " + centroidsForCategories.mkString(","))

    val categoriesSortedByCentroid: List[Int] = centroidsForCategories.toList.sortBy(_._2).map(_._1)

    // Cumulative sums of bin statistics for left, right parts of split.
    val leftImpurityAgg = metadata.createImpurityAggregator()
    val rightImpurityAgg = metadata.createImpurityAggregator()
    var j = 0
    val length = aggStats.length
    while (j < length) {
      rightImpurityAgg.add(aggStats(j))
      j += 1
    }

    var bestSplitIndex: Int = -1  // index into categoriesSortedByCentroid
    val bestLeftImpurityAgg = leftImpurityAgg.deepCopy()
    var bestGain: Double = 0.0
    val fullImpurity = rightImpurityAgg.getCalculator.calculate()
    var leftCount: Double = 0.0
    var rightCount: Double = rightImpurityAgg.getCount
    val fullCount: Double = rightCount

    // Consider all splits. These only cover valid splits, with at least one category on each side.
    val numSplits = categoriesSortedByCentroid.length - 1
    var sortedCatIndex = 0
    while (sortedCatIndex < numSplits) {
      val cat = categoriesSortedByCentroid(sortedCatIndex)
      // Update left, right stats
      val catStats = aggStats(cat)
      leftImpurityAgg.add(catStats)
      rightImpurityAgg.subtract(catStats)
      leftCount += catStats.getCount
      rightCount -= catStats.getCount
      // Compute impurity
      val leftWeight = leftCount / fullCount
      val rightWeight = rightCount / fullCount
      val leftImpurity = leftImpurityAgg.getCalculator.calculate()
      val rightImpurity = rightImpurityAgg.getCalculator.calculate()
      val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
      if (gain > bestGain && gain > metadata.minInfoGain) {
        bestSplitIndex = sortedCatIndex
        leftImpurityAgg.stats.copyToArray(bestLeftImpurityAgg.stats)
        bestGain = gain
      }
      sortedCatIndex += 1
    }

    val categoriesForSplit =
      categoriesSortedByCentroid.slice(0, bestSplitIndex + 1).map(_.toDouble)
    val bestFeatureSplit =
      new CategoricalSplit(featureIndex, categoriesForSplit.toArray, featureArity)
    val fullImpurityAgg = leftImpurityAgg.deepCopy().add(rightImpurityAgg)
    val bestRightImpurityAgg = fullImpurityAgg.deepCopy().subtract(bestLeftImpurityAgg)
    val bestImpurityStats = new ImpurityStats(bestGain, fullImpurity, fullImpurityAgg.getCalculator,
      bestLeftImpurityAgg.getCalculator, bestRightImpurityAgg.getCalculator)

    if (bestSplitIndex == -1 || bestGain == 0.0) {
      (None, bestImpurityStats)
    } else {
      (Some(bestFeatureSplit), bestImpurityStats)
    }
  }

  /**
   * Find the best split for an unordered categorical feature at a single node.
   *
   * Algorithm:
   *  - Considers all possible subsets (exponentially many)
   *
   * @param featureIndex  Index of feature being split.
   * @param values  Feature values at this node.  Sorted in increasing order.
   * @param labels  Labels corresponding to values, in the same order.
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.  The impurity stats maybe still be
   *          useful, so they are returned.
   */
  private[impl] def chooseUnorderedCategoricalSplit(
      featureIndex: Int,
      values: Seq[Double],
      labels: Seq[Double],
      metadata: AltDTMetadata,
      featureArity: Int): (Option[Split], ImpurityStats) = {

    // Label stats for each category
    val aggStats = Array.tabulate[ImpurityAggregatorSingle](featureArity)(
      _ => metadata.createImpurityAggregator())
    values.zip(labels).foreach { case (cat, label) =>
      // NOTE: we assume the values for categorical features are Ints in [0,featureArity)
      aggStats(cat.toInt).update(label)
    }

    // Aggregated statistics for left part of split and entire split.
    val leftImpurityAgg = metadata.createImpurityAggregator()
    val fullImpurityAgg = metadata.createImpurityAggregator()
    aggStats.foreach(fullImpurityAgg.add)
    val fullImpurity = fullImpurityAgg.getCalculator.calculate()

    if (featureArity == 1) {
      // All instances go right
      val impurityStats = new ImpurityStats(0.0, fullImpurityAgg.getCalculator.calculate(),
        fullImpurityAgg.getCalculator, leftImpurityAgg.getCalculator,
        fullImpurityAgg.getCalculator)
      (None, impurityStats)
    } else {
      //  TODO: We currently add and remove the stats for all categories for each split.
      //  A better way to do it would be to consider splits in an order such that each iteration
      //  only requires addition/removal of a single category and a single add/subtract to
      //  leftCount and rightCount.
      //  TODO: Use more efficient encoding such as gray codes
      val splits: Array[CategoricalSplit] = findSplits(featureIndex, featureArity, metadata)
      var bestSplit: Option[CategoricalSplit] = None
      val bestLeftImpurityAgg = leftImpurityAgg.deepCopy()
      var bestGain: Double = -1.0
      val fullCount: Double = values.size
      for (split <- splits) {
        // Update left, right impurity stats
        split.leftCategories.foreach(c => leftImpurityAgg.add(aggStats(c.toInt)))
        val rightImpurityAgg = fullImpurityAgg.deepCopy().subtract(leftImpurityAgg)
        val leftCount = leftImpurityAgg.getCount
        val rightCount = rightImpurityAgg.getCount
        // Compute impurity
        val leftWeight = leftCount / fullCount
        val rightWeight = rightCount / fullCount
        val leftImpurity = leftImpurityAgg.getCalculator.calculate()
        val rightImpurity = rightImpurityAgg.getCalculator.calculate()
        val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
        if (gain > bestGain && gain > metadata.minInfoGain) {
          bestSplit = Some(split)
          leftImpurityAgg.stats.copyToArray(bestLeftImpurityAgg.stats)
          bestGain = gain
        }
        // Reset left impurity stats
        leftImpurityAgg.clear()
      }

      val bestFeatureSplit = bestSplit match {
        case Some(split) => Some(
          new CategoricalSplit(featureIndex, split.leftCategories, featureArity))
        case None => None

      }
      val bestRightImpurityAgg = fullImpurityAgg.deepCopy().subtract(bestLeftImpurityAgg)
      val bestImpurityStats = new ImpurityStats(bestGain, fullImpurity,
        fullImpurityAgg.getCalculator, bestLeftImpurityAgg.getCalculator,
        bestRightImpurityAgg.getCalculator)
      (bestFeatureSplit, bestImpurityStats)
    }
  }

  /**
   * Returns all possible subsets of features for categorical splits.
   */
  private def findSplits(
      featureIndex: Int,
      featureArity: Int,
      metadata: AltDTMetadata): Array[CategoricalSplit] = {
    // Unordered features
    // 2^(featureArity - 1) - 1 combinations
    val numSplits = (1 << (featureArity - 1)) - 1
    val splits = new Array[CategoricalSplit](numSplits)

    var splitIndex = 0
    while (splitIndex < numSplits) {
      val categories: List[Double] =
        RandomForest.extractMultiClassCategories(splitIndex + 1, featureArity)
      splits(splitIndex) =
        new CategoricalSplit(featureIndex, categories.toArray, featureArity)
      splitIndex += 1
    }
    splits
  }

  /**
   * Choose splitting rule: feature value <= threshold
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.  The impurity stats maybe still be
   *          useful, so they are returned.
   */
  private[impl] def chooseContinuousSplit(
      featureIndex: Int,
      values: Seq[Double],
      labels: Seq[Double],
      metadata: AltDTMetadata): (Option[Split], ImpurityStats) = {

    val leftImpurityAgg = metadata.createImpurityAggregator()
    val rightImpurityAgg = metadata.createImpurityAggregator()
    var i = 0
    val len = labels.length
    while (i < len) {
      rightImpurityAgg.update(labels(i), 1.0)
      i += 1
    }

    var bestThreshold: Double = Double.NegativeInfinity
    val bestLeftImpurityAgg = leftImpurityAgg.deepCopy()
    var bestGain: Double = 0.0
    val fullImpurity = rightImpurityAgg.getCalculator.calculate()
    var leftCount: Double = 0.0
    var rightCount: Double = rightImpurityAgg.getCount
    val fullCount: Double = rightCount
    var currentThreshold = values.headOption.getOrElse(bestThreshold)
    var j = 0
    val length = values.length
    while (j < length) {
      val value = values(j)
      val label = labels(j)
      if (value != currentThreshold) {
        // Check gain
        val leftWeight = leftCount / fullCount
        val rightWeight = rightCount / fullCount
        val leftImpurity = leftImpurityAgg.getCalculator.calculate()
        val rightImpurity = rightImpurityAgg.getCalculator.calculate()
        val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
        if (gain > bestGain && gain > metadata.minInfoGain) {
          bestThreshold = currentThreshold
          leftImpurityAgg.stats.copyToArray(bestLeftImpurityAgg.stats)
          bestGain = gain
        }
        currentThreshold = value
      }
      // Move this instance from right to left side of split.
      leftImpurityAgg.update(label, 1.0)
      rightImpurityAgg.update(label, -1.0)
      leftCount += 1.0
      rightCount -= 1.0
      j += 1
    }

    val fullImpurityAgg = leftImpurityAgg.deepCopy().add(rightImpurityAgg)
    val bestRightImpurityAgg = fullImpurityAgg.deepCopy().subtract(bestLeftImpurityAgg)
    val bestImpurityStats = new ImpurityStats(bestGain, fullImpurity, fullImpurityAgg.getCalculator,
      bestLeftImpurityAgg.getCalculator, bestRightImpurityAgg.getCalculator)
    val split = if (bestThreshold != Double.NegativeInfinity && bestThreshold != values.last) {
      Some(new ContinuousSplit(featureIndex, bestThreshold))
    } else {
      None
    }
    (split, bestImpurityStats)
  }

  /**
   * Feature vector types are based on (feature type, representation).
   * The feature type can be continuous or categorical.
   *
   * Features are sorted by value, so we must store indices + values.
   * These values are currently stored in a dense representation only.
   * TODO: Support sparse storage (to optimize deeper levels of the tree), and maybe compressed
   *       storage (to optimize upper levels of the tree).
   * @param featureArity  For categorical features, this gives the number of categories.
   *                      For continuous features, this should be set to 0.
   */
  private[impl] class FeatureVector(
      val featureIndex: Int,
      val featureArity: Int,
      val values: Array[Double],
      val indices: Array[Int])
    extends Serializable {

    def isCategorical: Boolean = featureArity > 0

    /** For debugging */
    override def toString: String = {
      "  FeatureVector(" +
        s"    featureIndex: $featureIndex,\n" +
        s"    featureType: ${if (featureArity == 0) "Continuous" else "Categorical"},\n" +
        s"    featureArity: $featureArity,\n" +
        s"    values: ${values.mkString(", ")},\n" +
        s"    indices: ${indices.mkString(", ")},\n" +
        "  )"
    }

    def deepCopy(): FeatureVector =
      new FeatureVector(featureIndex, featureArity, values.clone(), indices.clone())

    override def equals(other: Any): Boolean = {
      other match {
        case o: FeatureVector =>
          featureIndex == o.featureIndex && featureArity == o.featureArity &&
            values.sameElements(o.values) && indices.sameElements(o.indices)
        case _ => false
      }
    }
  }

  private[impl] object FeatureVector {
    /** Store column sorted by feature values. */
    def fromOriginal(
        featureIndex: Int,
        featureArity: Int,
        featureVector: Vector): FeatureVector = {
      val (values, indices) = featureVector.toArray.zipWithIndex.sorted.unzip
      new FeatureVector(featureIndex, featureArity, values.toArray, indices.toArray)
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
    val nodeRowIndices = col.indices.slice(fromOffset, toOffset)
    val nodeRowValues = col.values.slice(fromOffset, toOffset)
    val nodeRowValuesSortedByIndices = nodeRowIndices.zip(nodeRowValues).sortBy(_._1).map(_._2)
    val bitv = new BitSubvector(fromOffset, toOffset)
    var i = 0
    while (i < nodeRowValuesSortedByIndices.length) {
      val value = nodeRowValuesSortedByIndices(i)
      if (!split.shouldGoLeft(value)) {
        bitv.set(fromOffset + i)
      }
      i += 1
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
     *                    When an active node is not split (e.g., because no good split was found),
     *                    then the corresponding BitSubvector can be missing.
     * @return Updated partition info
     */
    def update(bitVectors: Array[BitSubvector], newNumNodeOffsets: Int): PartitionInfo = {
      val newColumns = columns.map { oldCol =>
        val col = oldCol.deepCopy()
        var curBitVecIdx = 0
        activeNodes.iterator.foreach { nodeIdx =>
          val from = nodeOffsets(nodeIdx)
          val to = nodeOffsets(nodeIdx + 1)
          if (curBitVecIdx + 1 < bitVectors.length && bitVectors(curBitVecIdx).to <= from) {
            // If there are no more BitVectors, curBitVecIdx stays at the last bitVector,
            // which is acceptable (since it will not cover further nodes which were not split).
            curBitVecIdx += 1
          }
          val curBitVector = bitVectors(curBitVecIdx)
          // If the current BitVector does not cover this node, then this node was not split,
          // so we do not need to update its part of the column.  Otherwise, we update it.
          if (curBitVector.from <= from && to <= curBitVector.to) {
            // Sort range [from, to) based on indices.  This is required to match the bit vector
            // across all workers.  See [[bitSubvectorFromSplit]] for details.
            val rangeIndices = col.indices.view.slice(from, to).toArray
            val rangeValues = col.values.view.slice(from, to).toArray
            val sortedRange = rangeIndices.zip(rangeValues).sortBy(_._1)
            // Sort range [from, to) based on bit vector.
            sortedRange.zipWithIndex.map { case ((idx, value), i) =>
              val bit = curBitVector.get(from + i)
              // TODO: In-place merge, rather than general sort.
              // TODO: We don't actually need to sort the categorical features using our approach.
              (bit, value, idx)
            }.sorted.zipWithIndex.foreach { case ((bit, value, idx), i) =>
              col.values(from + i) = value
              col.indices(from + i) = idx
            }
          }
        }
        col
      }

      // Create a 2-level representation of the new nodeOffsets (to be flattened).
      // These 2 levels correspond to original nodes and their children (if split).
      val newNodeOffsets = nodeOffsets.map(Array(_))
      var curBitVecIdx = 0
      activeNodes.iterator.foreach { nodeIdx =>
        val from = nodeOffsets(nodeIdx)
        val to = nodeOffsets(nodeIdx + 1)
        if (curBitVecIdx + 1 < bitVectors.length && bitVectors(curBitVecIdx).to <= from) {
          // If there are no more BitVectors, curBitVecIdx stays at the last bitVector,
          // which is acceptable (since it will not cover further nodes which were not split).
          curBitVecIdx += 1
        }
        val curBitVector = bitVectors(curBitVecIdx)
        // If the current BitVector does not cover this node, then this node was not split,
        // so we do not need to create a new node offset.  Otherwise, we create an offset.
        if (curBitVector.from <= from && to <= curBitVector.to) {
          // Count number of values splitting to left vs. right
          val numRight = Range(from, to).count(curBitVector.get)
          val numLeft = to - from - numRight
          if (numLeft != 0 && numRight != 0) {
            // node is split
            val oldOffset = newNodeOffsets(nodeIdx).head
            newNodeOffsets(nodeIdx) = Array(oldOffset, oldOffset + numLeft)
          }
        }
      }

      assert(newNodeOffsets.map(_.length).sum == newNumNodeOffsets,
        s"(W) newNodeOffsets total size: ${newNodeOffsets.map(_.length).sum}," +
          s" newNumNodeOffsets: $newNumNodeOffsets")

      // Identify the new activeNodes based on the 2-level representation of the new nodeOffsets.
      val newActiveNodes = new BitSet(newNumNodeOffsets - 1)
      var newNodeOffsetsIdx = 0
      var i = 0
      while (i < newNodeOffsets.length) {
        val offsets = newNodeOffsets(i)
        if (offsets.length == 2) {
          newActiveNodes.set(newNodeOffsetsIdx)
          newActiveNodes.set(newNodeOffsetsIdx + 1)
          newNodeOffsetsIdx += 2
        } else {
          newNodeOffsetsIdx += 1
        }
        i += 1
      }
      PartitionInfo(newColumns, newNodeOffsets.flatten, newActiveNodes)
    }
  }
}
