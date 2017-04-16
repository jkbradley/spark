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

package org.apache.spark.ml.regression

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.{Instance, LabeledPoint}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tree.impl.TreeTests
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTestingUtils}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.regression.{LabeledPoint => OldLabeledPoint}
import org.apache.spark.mllib.tree.{DecisionTree => OldDecisionTree, DecisionTreeSuite => OldDecisionTreeSuite}
import org.apache.spark.mllib.util.{LinearDataGenerator, MLlibTestSparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}

class DecisionTreeRegressorSuite
  extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  import DecisionTreeRegressorSuite.compareAPIs
  import testImplicits._

  private var categoricalDataPointsRDD: RDD[LabeledPoint] = _
  private var linearRegressionData: DataFrame = _

  private val seed = 42

  override def beforeAll() {
    super.beforeAll()
    categoricalDataPointsRDD =
      sc.parallelize(OldDecisionTreeSuite.generateCategoricalDataPoints().map(_.asML))
    linearRegressionData = sc.parallelize(LinearDataGenerator.generateLinearInput(
      intercept = 6.3, weights = Array(4.7, 7.2), xMean = Array(0.9, -1.3),
      xVariance = Array(0.7, 1.2), nPoints = 1000, seed, eps = 0.5), 2).map(_.asML).toDF()
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests calling train()
  /////////////////////////////////////////////////////////////////////////////

  test("Regression stump with 3-ary (ordered) categorical features") {
    val dt = new DecisionTreeRegressor()
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(100)
      .setSeed(1)
    val categoricalFeatures = Map(0 -> 3, 1 -> 3)
    compareAPIs(categoricalDataPointsRDD, dt, categoricalFeatures)
  }

  test("Regression stump with binary (ordered) categorical features") {
    val dt = new DecisionTreeRegressor()
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(100)
    val categoricalFeatures = Map(0 -> 2, 1 -> 2)
    compareAPIs(categoricalDataPointsRDD, dt, categoricalFeatures)
  }

  test("copied model must have the same parent") {
    val categoricalFeatures = Map(0 -> 2, 1 -> 2)
    val df = TreeTests.setMetadata(categoricalDataPointsRDD.map(_.toInstance(1.0)),
      categoricalFeatures, numClasses = 0)
    val dtr = new DecisionTreeRegressor()
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(8)
    val model = dtr.fit(df)
    MLTestingUtils.checkCopyAndUids(dtr, model)
  }

  test("predictVariance") {
    val dt = new DecisionTreeRegressor()
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(100)
      .setPredictionCol("")
      .setVarianceCol("variance")
    val categoricalFeatures = Map(0 -> 2, 1 -> 2)

    val df = TreeTests.setMetadata(categoricalDataPointsRDD.map(_.toInstance(1.0)),
      categoricalFeatures, numClasses = 0)
    val model = dt.fit(df)

    val predictions = model.transform(df)
      .select(model.getFeaturesCol, model.getVarianceCol)
      .collect()

    predictions.foreach { case Row(features: Vector, variance: Double) =>
      val expectedVariance = model.rootNode.predictImpl(features).impurityStats.calculate()
      assert(variance === expectedVariance,
        s"Expected variance $expectedVariance but got $variance.")
    }

    val varianceData: RDD[Instance] = TreeTests.varianceData(sc).map(_.toInstance(1.0))
    val varianceDF = TreeTests.setMetadata(varianceData, Map.empty[Int, Int], 0)
    dt.setMaxDepth(1)
      .setMaxBins(6)
      .setSeed(0)
    val transformVarDF = dt.fit(varianceDF).transform(varianceDF)
    val calculatedVariances = transformVarDF.select(dt.getVarianceCol).collect().map {
      case Row(variance: Double) => variance
    }

    // Since max depth is set to 1, the best split point is that which splits the data
    // into (0.0, 1.0, 2.0) and (10.0, 12.0, 14.0). The predicted variance for each
    // data point in the left node is 0.667 and for each data point in the right node
    // is 2.667
    val expectedVariances = Array(0.667, 0.667, 0.667, 2.667, 2.667, 2.667)
    calculatedVariances.zip(expectedVariances).foreach { case (actual, expected) =>
      assert(actual ~== expected absTol 1e-3)
    }
  }

  test("Feature importance with toy data") {
    val dt = new DecisionTreeRegressor()
      .setImpurity("variance")
      .setMaxDepth(3)
      .setSeed(123)

    // In this data, feature 1 is very important.
    val data: RDD[Instance] = TreeTests.featureImportanceData(sc).map(_.toInstance(1.0))
    val categoricalFeatures = Map.empty[Int, Int]
    val df: DataFrame = TreeTests.setMetadata(data, categoricalFeatures, 0)

    val model = dt.fit(df)

    val importances = model.featureImportances
    val mostImportantFeature = importances.argmax
    assert(mostImportantFeature === 1)
    assert(importances.toArray.sum === 1.0)
    assert(importances.toArray.forall(_ >= 0.0))
  }

  test("should support all NumericType labels and not support other types") {
    val dt = new DecisionTreeRegressor().setMaxDepth(1)
    MLTestingUtils.checkNumericTypes[DecisionTreeRegressionModel, DecisionTreeRegressor](
      dt, spark, isClassification = false) { (expected, actual) =>
        TreeTests.checkEqual(expected, actual)
      }
  }

  test("training with sample weights") {
    val df = linearRegressionData
    val numClasses = 0
    val predEquals = (x: Double, y: Double) => x ~== y relTol 0.05
    val testParams = Seq(1)  //  Seq(5, 10)
    for (maxDepth <- testParams) {
      val estimator = new DecisionTreeRegressor()
        .setMaxDepth(maxDepth)
        .setSeed(1234)
        // .setMinWeightFractionPerNode(0.05)
      MLTestingUtils.testArbitrarilyScaledWeights[DecisionTreeRegressionModel,
        DecisionTreeRegressor](df.as[LabeledPoint], estimator,
        MLTestingUtils.modelPredictionEquals(df, predEquals, 0.9))
      /*
      MLTestingUtils.testOutliersWithSmallWeights[DecisionTreeRegressionModel,
        DecisionTreeRegressor](df.as[LabeledPoint], estimator, numClasses,
        MLTestingUtils.modelPredictionEquals(df, predEquals, 0.8),
        outlierRatio = 2)
      MLTestingUtils.testOversamplingVsWeighting[DecisionTreeRegressionModel,
        DecisionTreeRegressor](df.as[LabeledPoint], estimator,
        MLTestingUtils.modelPredictionEquals(df, predEquals, 1.0), seed)
      */
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests of model save/load
  /////////////////////////////////////////////////////////////////////////////

  test("read/write") {
    def checkModelData(
        model: DecisionTreeRegressionModel,
        model2: DecisionTreeRegressionModel): Unit = {
      TreeTests.checkEqual(model, model2)
      assert(model.numFeatures === model2.numFeatures)
    }

    val dt = new DecisionTreeRegressor()
    val rdd = TreeTests.getTreeReadWriteData(sc).map(_.toInstance(1.0))

    // Categorical splits with tree depth 2
    val categoricalData: DataFrame =
      TreeTests.setMetadata(rdd, Map(0 -> 2, 1 -> 3), numClasses = 0)
    testEstimatorAndModelReadWrite(dt, categoricalData,
      TreeTests.allParamSettings, TreeTests.allParamSettings, checkModelData)

    // Continuous splits with tree depth 2
    val continuousData: DataFrame =
      TreeTests.setMetadata(rdd, Map.empty[Int, Int], numClasses = 0)
    testEstimatorAndModelReadWrite(dt, continuousData,
      TreeTests.allParamSettings, TreeTests.allParamSettings, checkModelData)

    // Continuous splits with tree depth 0
    testEstimatorAndModelReadWrite(dt, continuousData,
      TreeTests.allParamSettings ++ Map("maxDepth" -> 0),
      TreeTests.allParamSettings ++ Map("maxDepth" -> 0), checkModelData)
  }
}

private[ml] object DecisionTreeRegressorSuite extends SparkFunSuite {

  /**
   * Train 2 decision trees on the given dataset, one using the old API and one using the new API.
   * Convert the old tree to the new format, compare them, and fail if they are not exactly equal.
   */
  def compareAPIs(
      data: RDD[LabeledPoint],
      dt: DecisionTreeRegressor,
      categoricalFeatures: Map[Int, Int]): Unit = {
    val numFeatures = data.first().features.size
    val oldStrategy = dt.getOldStrategy(categoricalFeatures)
    val oldTree = OldDecisionTree.train(data.map(OldLabeledPoint.fromML), oldStrategy)
    val newData: DataFrame =
      TreeTests.setMetadata(data.map(_.toInstance(1.0)), categoricalFeatures, numClasses = 0)
    val newTree = dt.fit(newData)
    // Use parent from newTree since this is not checked anyways.
    val oldTreeAsNew = DecisionTreeRegressionModel.fromOld(
      oldTree, newTree.parent.asInstanceOf[DecisionTreeRegressor], categoricalFeatures)
    TreeTests.checkEqual(oldTreeAsNew, newTree)
    assert(newTree.numFeatures === numFeatures)
  }
}
