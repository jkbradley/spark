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

package org.apache.spark.mllib.tree

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.tree.configuration.FeatureType.FeatureType
import org.apache.spark.mllib.tree.configuration.FeatureType.FeatureType
import org.apache.spark.mllib.tree.impurity._
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.Logging
import org.apache.spark.SparkContext._
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector, Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.{FeatureType, Strategy}
import org.apache.spark.mllib.tree.impl.Util._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd.RDD


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
@Experimental
class AltDT (private val strategy: Strategy) extends Serializable with Logging {

  strategy.assertValid()

  /**
   * Feature vector types are based on (feature type, representation).
   * The feature type can be continuous or categorical.
   *
   * Features are sorted by value, so we must store indices + values.
   * These values are currently stored in a dense representation only.
   * TODO: Support sparse storage.
   */
  private class FeatureVector(
      val featureIndex: Int,
      val featureType: FeatureType,
      val values: Array[Double],
      val indices: Array[Int])
    extends Serializable

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
   * Intermediate data stored on each partition during learning.
   * @param columns  Subset of columns (features) stored in this partition.
   * @param nodeOffsets  Offsets into the rows (instances) indicating how they are split among
   *                     nodes.  The rows corresponding to node i are in the range
   *                     [nodeOffsets(i), nodeOffsets(i+1)).
   */
  private case class PartitionInfo(columns: Array[FeatureVector], nodeOffsets: Array[Int])
    extends Serializable {

    /**
     * Update columns and nodeOffsets for the next level of the tree..
     * @param bitVector  Bit vector encoding splits for the next level of the tree.
     *                   bitVector(i) = false iff instance i goes to the left child.
     * @return Updated partition info
     */
    def update(bitVector: Array[Boolean]): PartitionInfo = ???
  }

  private def getImpurityAggregator: ImpurityAggregatorSingle = {
    strategy.impurity match {
      case Entropy => new EntropyAggregatorSingle(strategy.numClasses)
      case Gini => new GiniAggregatorSingle(strategy.numClasses)
      case Variance => new VarianceAggregatorSingle
    }
  }

  /*
  private class NodeIndex private (val splits: Array[Boolean]) extends Serializable {
    import NodeIndex._

    def leftChild: Array[Boolean] = splits :+ LEFT

    def rightChild: Array[Boolean] = splits :+ RIGHT

  }

  private object NodeIndex extends Serializable {
    final val LEFT: Boolean = false
    final val RIGHT: Boolean = true
    def root: NodeIndex = new NodeIndex(Array.emptyBooleanArray)
  }

  private class SplitInfo() extends Serializable
  */

  /**
   * Method to train a decision tree model over an RDD
   * @param input Training data: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]]
   * @return DecisionTreeModel that can be used for prediction
   */
  def train(input: RDD[LabeledPoint]): DecisionTreeModel = {
    // TODO: Check validity of params

    // The case with 1 node (depth = 0) is handled separately.
    // This allows all iterations in the depth > 0 case to use the same code.
    if (strategy.maxDepth == 0) {
      val impurityAggregator: ImpurityAggregatorSingle =
        input.aggregate(getImpurityAggregator)(
          (agg, lp) => agg.update(lp.label, 1.0),
          (agg1, agg2) => agg1.merge(agg2))
      val impurityCalculator = impurityAggregator.getCalculator
      val node = Node(nodeIndex = 0, predict = impurityCalculator.getPredict,
        impurity = impurityCalculator.calculate(), isLeaf = true)
      return new DecisionTreeModel(node, strategy.algo)
    }

    // Prepare column store.
    //   Note: rowToColumnStoreDense checks to make sure numRows < Int.MaxValue.
    // TODO: Is this mapping from arrays to iterators to arrays (when constructing learningData)?
    //       Or is the mapping implicit (i.e., not costly)?
    val colStoreInit: RDD[(Int, Vector)] = rowToColumnStoreDense(input.map(_.features))
    val numRows: Int = colStoreInit.take(1)(0)._2.size
    val labels = new Array[Double](numRows)
    input.map(_.label).zipWithIndex().collect().foreach { case (label: Double, rowIndex: Long) =>
      labels(rowIndex.toInt) = label
    }
    val labelsBroadcast = input.sparkContext.broadcast(labels)

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
      new PartitionInfo(groupedCols, Array[Int](0, numRows))
    }

    // Iteratively learn, one level of the tree at a time.
    var currentLevel = 0
    while (currentLevel < strategy.maxDepth) {

      val numNodesInLevel: Int = ???

      // On each partition, for each feature on the partition, select the best split for each node.
      // This will use:
      //  - groupedColStore (the features)
      //  - partitionInfos (the node -> instance mapping)
      //  - labelsBroadcast (the labels column)
      // Each worker returns:
      //   for each node, best split + info gain
      val workerBestSplitsAndGains: RDD[Array[(Split, InformationGainStats)]] = partitionInfos.map {
        case PartitionInfo(columns: Array[FeatureVector], nodeOffsets: Array[Int]) =>
          val localLabels = labelsBroadcast.value
          val localBestSplitsAndGains = new Array[(Split, InformationGainStats)](numNodesInLevel)
          // Iterate over the nodes in the current level.
          var nodeIndexInLevel = 0
          while (nodeIndexInLevel < numNodesInLevel) {
            val fromOffset = nodeOffsets(nodeIndexInLevel)
            val toOffset = nodeOffsets(nodeIndexInLevel + 1)
            val splitsAndStats =
              columns.map(col => chooseSplit(col, localLabels, fromOffset, toOffset))
            localBestSplitsAndGains(nodeIndexInLevel) = splitsAndStats.maxBy(_._2.gain)
            nodeIndexInLevel += 1
          }
          localBestSplitsAndGains
      }

      // Aggregate best split for each node.
      // TODO: verify that initialization gives info gain 0
      val bestSplitsAndGains: Array[(Split, InformationGainStats)] =
        workerBestSplitsAndGains.fold(new Array[(Split, InformationGainStats)](numNodesInLevel)){
          case (splitsGains1, splitsGains2) =>
            splitsGains1.zip(splitsGains2).map { case ((split1, gain1), (split2, gain2)) =>
              if (gain1.gain >= gain2.gain) {
                (split1, gain1)
              } else {
                (split2, gain2)
              }
            }
        }

      // Update current model.
      // TODO

      // Aggregate bit vector (1 bit/instance) indicating whether each instance goes left or right.
      //  - Send chosen splits to workers.
      //  - Each worker creates part of the bit vector corresponding to the splits it created.
      //  - Aggregate the partial bit vectors to create one vector (of length numRows).
      val broadcastBestSplits: Broadcast[Array[Split]] =
        input.sparkContext.broadcast(bestSplitsAndGains.map(_._1))
      val workerPartialBitVectors: RDD[Array[PartialBitVector]] = partitionInfos.map {
        case PartitionInfo(columns: Array[FeatureVector], nodeOffsets: Array[Int]) =>
          val localBestSplits = broadcastBestSplits.value
          // localFeatureIndex[feature index] = index into groupedCols
          val localFeatureIndex: Map[Int, Int] = columns.map(_.featureIndex).zipWithIndex.toMap
          localBestSplits.zipWithIndex.flatMap { case (split: Split, nodeIndexInLevel: Int) =>
            if (localFeatureIndex.contains(split.feature)) {
              val fromOffset = nodeOffsets(nodeIndexInLevel)
              val toOffset = nodeOffsets(nodeIndexInLevel + 1)
              val colIndex: Int = localFeatureIndex(split.feature)
              Iterator(PartialBitVector.fromSplit(
                columns(colIndex), fromOffset, toOffset, split, nodeIndexInLevel))
            } else {
              Iterator()
            }
          }.toArray
      }
      val fullBitVector: Array[Boolean] = {
        val aggBitVector: Array[PartialBitVector] =
          workerPartialBitVectors.fold(Array.empty[PartialBitVector])(PartialBitVector.merge)
        assert(aggBitVector.size == 1)
        assert(aggBitVector(0).fromNode == 0 && aggBitVector(0).toNode == numNodesInLevel)
        aggBitVector(0).bits
      }

      // Broadcast bit vector.  On each partition, update instance--node map.
      val broadcastBitVector = input.sparkContext.broadcast(fullBitVector)
      partitionInfos = partitionInfos.map {
        case partitionInfo: PartitionInfo =>
          val localBitVector: Array[Boolean] = broadcastBitVector.value
          partitionInfo.update(localBitVector)
      }

      currentLevel += 1
    }

    // Done with learning
    groupedColStore.unpersist()
    labelsBroadcast.unpersist()
    // TODO: return model
  }

  // TODO
  // NOTE: The order of col does not correspond to the order of labels.  Use the index in col.
  private def chooseSplit(
      col: FeatureVector,
      labels: Array[Double],
      fromOffset: Int,
      toOffset: Int): (Split, InformationGainStats) = ???

  private class PartialBitVector(val fromNode: Int, val toNode: Int, val bits: Array[Boolean])

  private object PartialBitVector {
    def merge(
        parts1: Array[PartialBitVector],
        parts2: Array[PartialBitVector]): Array[PartialBitVector] = {
      // Merge sorted parts1, parts2
      val sortedParts = Array.fill[PartialBitVector](parts1.size + parts2.size)(null)
      var i1 = 0 // indexing parts1
      var i2 = 0 // indexing parts2
      while (i1 < parts1.size || i2 < parts2.size) {
        val i = i1 + i2 // index into sortedParts
        if (i1 < parts1.size) {
          if (i2 < parts2.size) {
            // Choose between parts1,2
            if (parts1(i1).fromNode < parts2(i2).fromNode) {
              sortedParts(i) = parts1(i1)
              i1 += 1
            } else {
              sortedParts(i) = parts2(i2)
              i2 += 1
            }
          } else {
            // Take remaining items from parts1
            parts1.view.slice(i1, parts1.size).copyToArray(sortedParts, i)
            i1 = parts1.size
          }
        } else {
          // Take remaining items from parts2
          parts2.view.slice(i2, parts2.size).copyToArray(sortedParts, i)
          i2 = parts2.size
        }
      }
      // Merge adjacent PartialBitVectors (for adjacent node ranges)
      val newParts = new ArrayBuffer[PartialBitVector]()
      var j = 0 // indexing sortedParts
      if (sortedParts.size > 0) {
        newParts += sortedParts(0)
        j += 1
      }
      while (j < sortedParts.size) {
        // Check to see if the next PartialBitVector can be merged with the previous one.
        if (newParts.last.toNode == sortedParts(j).fromNode) {
          // TODO: RIGHT HERE NOW
        } else {

        }
        j += 1
      }
      newParts.toArray
    }

    def fromSplit(
        col: FeatureVector,
        fromOffset: Int,
        toOffset: Int,
        split: Split,
        nodeIndexInLevel: Int) = ???
  }

}

object AltDT extends Serializable with Logging {
}
