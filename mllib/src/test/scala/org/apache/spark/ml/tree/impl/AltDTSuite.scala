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

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.AltDT.{AltDTMetadata, FeatureVector, PartitionInfo}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.impurity._
import org.apache.spark.mllib.tree.model.ImpurityStats
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.util.collection.BitSet
import org.roaringbitmap.RoaringBitmap

import scala.util.Random

/**
 * Test suite for [[AltDT]].
 */
class AltDTSuite extends SparkFunSuite with MLlibTestSparkContext  {

  /* * * * * * * * * * * Integration tests * * * * * * * * * * */

  test("run deep example") {
    val data = Range(0, 3).map(x => LabeledPoint(math.pow(x, 3), Vectors.dense(x)))
    val df = sqlContext.createDataFrame(data)
    val dt = new DecisionTreeRegressor()
      .setFeaturesCol("features") // indexedFeatures
      .setLabelCol("label")
      .setMaxDepth(10)
      .setAlgorithm("byCol")
    val model = dt.fit(df)
    assert(model.rootNode.isInstanceOf[InternalNode])
    val root = model.rootNode.asInstanceOf[InternalNode]
    assert(root.leftChild.isInstanceOf[InternalNode] && root.rightChild.isInstanceOf[LeafNode])
    val left = root.leftChild.asInstanceOf[InternalNode]
    assert(left.leftChild.isInstanceOf[LeafNode], left.rightChild.isInstanceOf[LeafNode])
  }

  test("run example") {
    val data = Range(0, 8).map(x => LabeledPoint(x, Vectors.dense(x)))
    val df = sqlContext.createDataFrame(data)
    val dt = new DecisionTreeRegressor()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxDepth(10)
      .setAlgorithm("byCol")
    val model = dt.fit(df)
    assert(model.rootNode.isInstanceOf[InternalNode])
    val root = model.rootNode.asInstanceOf[InternalNode]
    assert(root.leftChild.isInstanceOf[InternalNode] && root.rightChild.isInstanceOf[InternalNode])
    val left = root.leftChild.asInstanceOf[InternalNode]
    val right = root.rightChild.asInstanceOf[InternalNode]
    val grandkids = Array(left.leftChild, left.rightChild, right.leftChild, right.rightChild)
    assert(grandkids.forall(_.isInstanceOf[InternalNode]))
  }

  test("example with imbalanced tree") {
    val data = Seq(
      (0.0, Vectors.dense(0.0, 0.0)),
      (0.0, Vectors.dense(0.0, 0.0)),
      (1.0, Vectors.dense(0.0, 1.0)),
      (0.0, Vectors.dense(0.0, 1.0)),
      (1.0, Vectors.dense(1.0, 0.0)),
      (1.0, Vectors.dense(1.0, 0.0)),
      (1.0, Vectors.dense(1.0, 1.0)),
      (1.0, Vectors.dense(1.0, 1.0))
    ).map { case (l, p) => LabeledPoint(l, p) }
    val df = sqlContext.createDataFrame(data)
    val dt = new DecisionTreeRegressor()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxDepth(5)
      .setAlgorithm("byCol")
    val model = dt.fit(df)
    assert(model.depth === 2)
    assert(model.numNodes === 5)
  }

  test("example providing transposed dataset") {
    val data = Range(0, 8).map(x => LabeledPoint(x, Vectors.dense(x)))
    val df = sqlContext.createDataFrame(data)
    val dt = new DecisionTreeRegressor()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxDepth(10)
      .setAlgorithm("byCol")
    val (columns, labels) = dt.transposeDataset(df)
    val model = dt.fit(df, columns, labels)
    assert(model.rootNode.isInstanceOf[InternalNode])
    val root = model.rootNode.asInstanceOf[InternalNode]
    assert(root.leftChild.isInstanceOf[InternalNode] && root.rightChild.isInstanceOf[InternalNode])
    val left = root.leftChild.asInstanceOf[InternalNode]
    val right = root.rightChild.asInstanceOf[InternalNode]
    val grandkids = Array(left.leftChild, left.rightChild, right.leftChild, right.rightChild)
    assert(grandkids.forall(_.isInstanceOf[InternalNode]))
  }

  /* * * * * * * * * * * Helper classes * * * * * * * * * * */

  test("FeatureVector") {
    val v = new FeatureVector(1, 0, Array(0.1, 0.3, 0.7), Array(1, 2, 0))

    val vCopy = v.deepCopy()
    vCopy.values(0) = 1000
    assert(v.values(0) !== vCopy.values(0))

    val original = Vectors.dense(0.7, 0.1, 0.3)
    val v2 = FeatureVector.fromOriginal(1, 0, original)
    assert(v === v2)
  }

  test("FeatureVectorSortByValue") {
    val values = Array(0.1, 0.2, 0.4, 0.6, 0.7, 0.9, 1.5, 1.55)
    val col = Random.shuffle(values.toIterator).toArray
    val unsortedIndices = col.indices
    val sortedIndices = unsortedIndices.sortBy(x => col(x)).toArray
    val fvUnsorted = Vectors.dense(col)
    val featureIndex = 3
    val featureArity = 0
    val fvSorted =
      FeatureVector.fromOriginal(featureIndex, featureArity, fvUnsorted)
    assert(fvSorted.featureIndex === featureIndex)
    assert(fvSorted.featureArity === featureArity)
    assert(fvSorted.values.deep === values.deep)
    assert(fvSorted.indices.deep === sortedIndices.deep)
  }

  test("PartitionInfo") {
    val numRows = 4
    val col1 =
      FeatureVector.fromOriginal(0, 0, Vectors.dense(0.8, 0.2, 0.1, 0.6))
    val col2 =
      FeatureVector.fromOriginal(1, 3, Vectors.dense(0, 1, 0, 2))
    assert(col1.values.length === numRows)
    assert(col2.values.length === numRows)
    val nodeOffsets = Array(0, numRows)
    val activeNodes = new BitSet(1)
    activeNodes.set(0)

    val info = PartitionInfo(Array(col1, col2), nodeOffsets, activeNodes)

    // Create bitVector for splitting the 4 rows: L, R, L, R
    // New groups are {0, 2}, {1, 3}
    val bitVector = new RoaringBitmap()
    bitVector.add(1)
    bitVector.add(3)

    // for these tests, use the activeNodes for nodeSplitBitVector
    val newInfo = info.update(bitVector, newNumNodeOffsets = 3)

    assert(newInfo.columns.length === 2)
    val expectedCol1a =
      new FeatureVector(0, 0, Array(0.1, 0.8, 0.2, 0.6), Array(2, 0, 1, 3))
    val expectedCol1b =
      new FeatureVector(1, 3, Array(0, 0, 1, 2), Array(0, 2, 1, 3))
    assert(newInfo.columns(0) === expectedCol1a)
    assert(newInfo.columns(1) === expectedCol1b)
    assert(newInfo.nodeOffsets === Array(0, 2, 4))
    assert(newInfo.activeNodes.iterator.toSet === Set(0, 1))

    // Create 2 bitVectors for splitting into: 0, 2, 1, 3
    val bitVector2 = new RoaringBitmap()
    bitVector2.add(2) // 2 goes to the right
    bitVector2.add(3) // 3 goes to the right

    val newInfo2 = newInfo.update(bitVector2, newNumNodeOffsets = 5)

    assert(newInfo2.columns.length === 2)
    val expectedCol2a =
      new FeatureVector(0, 0, Array(0.8, 0.1, 0.2, 0.6), Array(0, 2, 1, 3))
    val expectedCol2b =
      new FeatureVector(1, 3, Array(0, 0, 1, 2), Array(0, 2, 1, 3))
    assert(newInfo2.columns(0) === expectedCol2a)
    assert(newInfo2.columns(1) === expectedCol2b)
    assert(newInfo2.nodeOffsets === Array(0, 1, 2, 3, 4))
    assert(newInfo2.activeNodes.iterator.toSet === Set(0, 1, 2, 3))
  }

  /* * * * * * * * * * * Misc  * * * * * * * * * * */

  test("numUnorderedBins") {
    // Note: We have duplicate bins (the inverse) for unordered features.  This should be fixed!
    assert(AltDT.numUnorderedBins(2) === 2)  // 2 categories => 2 bins
    assert(AltDT.numUnorderedBins(3) === 6)  // 3 categories => 6 bins
  }

  /* * * * * * * * * * * Choosing Splits  * * * * * * * * * * */

  test("computeBestSplits") {
    // TODO
  }

  test("chooseSplit: choose correct type of split") {
    val labels = Seq(0, 0, 0, 1, 1, 1, 1).map(_.toDouble).toArray
    val fromOffset = 1
    val toOffset = 4
    val impurity = Entropy
    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, impurity)

    val col1 = FeatureVector.fromOriginal(featureIndex = 0, featureArity = 0,
      featureVector = Vectors.dense(0.8, 0.1, 0.1, 0.2, 0.3, 0.5, 0.6))
    val (split1, _) = AltDT.chooseSplit(col1, labels, fromOffset, toOffset, metadata)
    assert(split1.nonEmpty && split1.get.isInstanceOf[ContinuousSplit])

    val col2 = FeatureVector.fromOriginal(featureIndex = 0, featureArity = 3,
      featureVector = Vectors.dense(0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0))
    val (split2, _) = AltDT.chooseSplit(col2, labels, fromOffset, toOffset, metadata)
    assert(split2.nonEmpty && split2.get.isInstanceOf[CategoricalSplit])
  }

  test("chooseOrderedCategoricalSplit: basic case") {
    val featureIndex = 0
    val values = Seq(0, 0, 1, 2, 2, 2, 2).map(_.toDouble)
    val featureArity = values.max.toInt + 1

    def testHelper(
        labels: Seq[Double],
        expectedLeftCategories: Array[Double],
        expectedLeftStats: Array[Double],
        expectedRightStats: Array[Double]): Unit = {
      val expectedRightCategories = Range(0, featureArity)
        .filter(c => !expectedLeftCategories.contains(c)).map(_.toDouble).toArray
      val impurity = Entropy
      val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, impurity)
      val (split, stats) =
        AltDT.chooseOrderedCategoricalSplit(featureIndex, values, labels, metadata, featureArity)
      split match {
        case Some(s: CategoricalSplit) =>
          assert(s.featureIndex === featureIndex)
          assert(s.leftCategories === expectedLeftCategories)
          assert(s.rightCategories === expectedRightCategories)
        case _ =>
          throw new AssertionError(
            s"Expected CategoricalSplit but got ${split.getClass.getSimpleName}")
      }
      val fullImpurityStatsArray =
        Array(labels.count(_ == 0.0).toDouble, labels.count(_ == 1.0).toDouble)
      val fullImpurity = impurity.calculate(fullImpurityStatsArray, labels.length)
      assert(stats.gain === fullImpurity)
      assert(stats.impurity === fullImpurity)
      assert(stats.impurityCalculator.stats === fullImpurityStatsArray)
      assert(stats.leftImpurityCalculator.stats === expectedLeftStats)
      assert(stats.rightImpurityCalculator.stats === expectedRightStats)
      assert(stats.valid)
    }

    val labels1 = Seq(0, 0, 1, 1, 1, 1, 1).map(_.toDouble)
    testHelper(labels1, Array(0.0), Array(2.0, 0.0), Array(0.0, 5.0))

    val labels2 = Seq(0, 0, 0, 1, 1, 1, 1).map(_.toDouble)
    testHelper(labels2, Array(0.0, 1.0), Array(3.0, 0.0), Array(0.0, 4.0))
  }

  test("chooseOrderedCategoricalSplit: return bad split if we should not split") {
    val featureIndex = 0
    val values = Seq(0, 0, 1, 2, 2, 2, 2).map(_.toDouble)
    val featureArity = values.max.toInt + 1

    val labels = Seq(1, 1, 1, 1, 1, 1, 1).map(_.toDouble)

    val impurity = Entropy
    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, impurity)
    val (split, stats) =
      AltDT.chooseOrderedCategoricalSplit(featureIndex, values, labels, metadata, featureArity)
    assert(split.isEmpty)
    val fullImpurityStatsArray =
      Array(labels.count(_ == 0.0).toDouble, labels.count(_ == 1.0).toDouble)
    val fullImpurity = impurity.calculate(fullImpurityStatsArray, labels.length)
    assert(stats.gain === 0.0)
    assert(stats.impurity === fullImpurity)
    assert(stats.impurityCalculator.stats === fullImpurityStatsArray)
    assert(stats.valid)
  }

  test("chooseUnorderedCategoricalSplit: basic case") {
    val featureIndex = 0
    val featureArity = 4
    val values = Seq(3.0, 1.0, 0.0, 2.0, 2.0)
    val labels = Seq(0.0, 0.0, 1.0, 1.0, 1.0)
    val impurity = Entropy
    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, impurity)
    val (split, _) = AltDT.chooseUnorderedCategoricalSplit(
      featureIndex, values, labels, metadata, featureArity)
    split match {
      case Some(s: CategoricalSplit) =>
        assert(s.featureIndex === featureIndex)
        assert(s.leftCategories.toSet === Set(0.0, 2.0))
        assert(s.rightCategories.toSet === Set(1.0, 3.0))
        // TODO: test correctness of stats
      case _ =>
        throw new AssertionError(
          s"Expected CategoricalSplit but got ${split.getClass.getSimpleName}")
    }
  }

  test("chooseUnorderedCategoricalSplit: return bad split if we should not split") {
    val featureIndex = 0
    val featureArity = 4
    val values = Seq(3.0, 1.0, 0.0, 2.0, 2.0)
    val labels = Seq(1.0, 1.0, 1.0, 1.0, 1.0)
    val impurity = Entropy
    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, impurity)
    val (split, stats) =
      AltDT.chooseOrderedCategoricalSplit(featureIndex, values, labels, metadata, featureArity)
    assert(split.isEmpty)
    val fullImpurityStatsArray =
      Array(labels.count(_ == 0.0).toDouble, labels.count(_ == 1.0).toDouble)
    val fullImpurity = impurity.calculate(fullImpurityStatsArray, labels.length)
    assert(stats.gain === 0.0)
    assert(stats.impurity === fullImpurity)
    assert(stats.impurityCalculator.stats === fullImpurityStatsArray)
    assert(stats.valid)
  }

  test("chooseContinuousSplit: basic case") {
    val featureIndex = 0
    val values = Seq(0.1, 0.2, 0.3, 0.4, 0.5)
    val labels = Seq(0.0, 0.0, 1.0, 1.0, 1.0)
    val impurity = Entropy
    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, impurity)
    val (split, stats) = AltDT.chooseContinuousSplit(featureIndex, values, labels, metadata)
    split match {
      case Some(s: ContinuousSplit) =>
        assert(s.featureIndex === featureIndex)
        assert(s.threshold === 0.2)
      case _ =>
        throw new AssertionError(
          s"Expected ContinuousSplit but got ${split.getClass.getSimpleName}")
    }
    val fullImpurityStatsArray =
      Array(labels.count(_ == 0.0).toDouble, labels.count(_ == 1.0).toDouble)
    val fullImpurity = impurity.calculate(fullImpurityStatsArray, labels.length)
    assert(stats.gain === fullImpurity)
    assert(stats.impurity === fullImpurity)
    assert(stats.impurityCalculator.stats === fullImpurityStatsArray)
    assert(stats.leftImpurityCalculator.stats === Array(2.0, 0.0))
    assert(stats.rightImpurityCalculator.stats === Array(0.0, 3.0))
    assert(stats.valid)
  }

  test("chooseContinuousSplit: return bad split if we should not split") {
    val featureIndex = 0
    val values = Seq(0.1, 0.2, 0.3, 0.4, 0.5)
    val labels = Seq(0.0, 0.0, 0.0, 0.0, 0.0)
    val impurity = Entropy
    val metadata = new AltDTMetadata(numClasses = 2, maxBins = 4, minInfoGain = 0.0, impurity)
    val (split, stats) = AltDT.chooseContinuousSplit(featureIndex, values, labels, metadata)
    // split should be None
    assert(split.isEmpty)
    // stats for parent node should be correct
    val fullImpurityStatsArray =
      Array(labels.count(_ == 0.0).toDouble, labels.count(_ == 1.0).toDouble)
    val fullImpurity = impurity.calculate(fullImpurityStatsArray, labels.length)
    assert(stats.gain === 0.0)
    assert(stats.impurity === fullImpurity)
    assert(stats.impurityCalculator.stats === fullImpurityStatsArray)
  }

  /* * * * * * * * * * * Bit subvectors * * * * * * * * * * */

  test("bitSubvectorFromSplit: 1 node") {
    val col =
      FeatureVector.fromOriginal(0, 0, Vectors.dense(0.1, 0.2, 0.4, 0.6, 0.7))
    val fromOffset = 0
    val toOffset = col.values.length
    val numRows = toOffset
    val split = new ContinuousSplit(0, threshold = 0.5)
    val bitv = AltDT.bitVectorFromSplit(col, fromOffset, toOffset, split, numRows)
    assert(bitv.toArray.toSet === Set(3, 4))
  }

  test("bitSubvectorFromSplit: 2 nodes") {
    // Initially, 1 split: (0, 2, 4) | (1, 3)
    val col = new FeatureVector(0, 0, Array(0.1, 0.2, 0.4, 0.6, 0.7),
      Array(4, 2, 0, 1, 3))
    def checkSplit(fromOffset: Int, toOffset: Int, threshold: Double,
      expectedRight: Set[Int]): Unit = {
        val split = new ContinuousSplit(0, threshold)
        val numRows = col.values.length
        val bitv = AltDT.bitVectorFromSplit(col, fromOffset, toOffset, split, numRows)
        assert(bitv.toArray.toSet === expectedRight)
    }
    // Left child node
    checkSplit(0, 3, 0.05, Set(0, 2, 4))
    checkSplit(0, 3, 0.15, Set(0, 2))
    checkSplit(0, 3, 0.2, Set(0))
    checkSplit(0, 3, 0.5, Set())
    // Right child node
    checkSplit(3, 5, 0.1, Set(1, 3))
    checkSplit(3, 5, 0.65, Set(3))
    checkSplit(3, 5, 0.8, Set())
  }

  test("collectBitVectors with 1 vector") {
    val col =
      FeatureVector.fromOriginal(0, 0, Vectors.dense(0.1, 0.2, 0.4, 0.6, 0.7))
    val numRows = col.values.length
    val activeNodes = new BitSet(1)
    activeNodes.set(0)
    val info = PartitionInfo(Array(col), Array(0, numRows), activeNodes)
    val partitionInfos = sc.parallelize(Seq(info))
    val bestSplit = new ContinuousSplit(0, threshold = 0.5)
    val bitVector = AltDT.aggregateBitVector(partitionInfos, Array(Some(bestSplit)), numRows)
    assert(bitVector.toArray.toSet === Set(3, 4))
  }

  test("collectBitVectors with 1 vector, with tied threshold") {
    val col = new FeatureVector(0, 0,
      Array(-4.0, -4.0, -2.0, -2.0, -1.0, -1.0, 1.0, 1.0),
      Array(3, 7, 2, 6, 1, 5, 0, 4))
    val numRows = col.values.length
    val activeNodes = new BitSet(1)
    activeNodes.set(0)
    val info = PartitionInfo(Array(col), Array(0, numRows), activeNodes)
    val partitionInfos = sc.parallelize(Seq(info))
    val bestSplit = new ContinuousSplit(0, threshold = -2.0)
    val bitVector = AltDT.aggregateBitVector(partitionInfos, Array(Some(bestSplit)), numRows)
    assert(bitVector.toArray.toSet === Set(0, 1, 4, 5))
  }

  /* * * * * * * * * * * Active nodes * * * * * * * * * * */

  test("computeActiveNodePeriphery") {
    // old periphery: 2 nodes
    val left = LearningNode.emptyNode(id = 1)
    val right = LearningNode.emptyNode(id = 2)
    val oldPeriphery: Array[LearningNode] = Array(left, right)
    // bestSplitsAndGains: Do not split left, but split right node.
    val lCalc = new EntropyCalculator(Array(8.0, 1.0))
    val lStats = new ImpurityStats(0.0, lCalc.calculate(),
      lCalc, lCalc, new EntropyCalculator(Array(0.0, 0.0)))

    val rSplit = new ContinuousSplit(featureIndex = 1, threshold = 0.6)
    val rCalc = new EntropyCalculator(Array(5.0, 7.0))
    val rRightChildCalc = new EntropyCalculator(Array(1.0, 5.0))
    val rLeftChildCalc = new EntropyCalculator(Array(
      rCalc.stats(0) - rRightChildCalc.stats(0),
      rCalc.stats(1) - rRightChildCalc.stats(1)))
    val rGain = {
      val rightWeight = rRightChildCalc.stats.sum / rCalc.stats.sum
      val leftWeight = rLeftChildCalc.stats.sum / rCalc.stats.sum
      rCalc.calculate() -
        rightWeight * rRightChildCalc.calculate() - leftWeight * rLeftChildCalc.calculate()
    }
    val rStats =
      new ImpurityStats(rGain, rCalc.calculate(), rCalc, rLeftChildCalc, rRightChildCalc)

    val bestSplitsAndGains: Array[(Option[Split], ImpurityStats)] =
      Array((None, lStats), (Some(rSplit), rStats))

    // Test A: Split right node
    val newPeriphery1: Array[LearningNode] =
      AltDT.computeActiveNodePeriphery(oldPeriphery, bestSplitsAndGains, minInfoGain = 0.0)
    // Expect 2 active nodes
    assert(newPeriphery1.length === 2)
    // Confirm right node was updated
    assert(right.split.get === rSplit)
    assert(!right.isLeaf)
    assert(right.stats.exactlyEquals(rStats))
    assert(right.leftChild.nonEmpty && right.leftChild.get === newPeriphery1(0))
    assert(right.rightChild.nonEmpty && right.rightChild.get === newPeriphery1(1))
    // Confirm new active nodes have stats but no children
    assert(newPeriphery1(0).leftChild.isEmpty && newPeriphery1(0).rightChild.isEmpty &&
      newPeriphery1(0).split.isEmpty &&
      newPeriphery1(0).stats.impurityCalculator.exactlyEquals(rLeftChildCalc))
    assert(newPeriphery1(1).leftChild.isEmpty && newPeriphery1(1).rightChild.isEmpty &&
      newPeriphery1(1).split.isEmpty &&
      newPeriphery1(1).stats.impurityCalculator.exactlyEquals(rRightChildCalc))

    // Test B: Increase minInfoGain, so split nothing
    val newPeriphery2: Array[LearningNode] =
      AltDT.computeActiveNodePeriphery(oldPeriphery, bestSplitsAndGains, minInfoGain = 1000.0)
    assert(newPeriphery2.isEmpty)
  }
}
