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

package org.apache.spark.ml.classification

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.impl.TreeTests
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.tree.LeafNode
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{EnsembleTestHelper, RandomForest => OldRandomForest}
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}

/**
 * Test suite for [[RandomForestClassifier]].
 */
class RandomForestClassifierSuite extends SparkFunSuite with MLlibTestSparkContext {

  import RandomForestClassifierSuite.compareAPIs

  private var orderedLabeledPoints50_1000: RDD[LabeledPoint] = _
  private var orderedLabeledPoints5_20: RDD[LabeledPoint] = _

  override def beforeAll() {
    super.beforeAll()
    orderedLabeledPoints50_1000 =
      sc.parallelize(EnsembleTestHelper.generateOrderedLabeledPoints(numFeatures = 50, 1000))
    orderedLabeledPoints5_20 =
      sc.parallelize(EnsembleTestHelper.generateOrderedLabeledPoints(numFeatures = 5, 20))
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests calling train()
  /////////////////////////////////////////////////////////////////////////////

  def binaryClassificationTestWithContinuousFeatures(rf: RandomForestClassifier) {
    val categoricalFeatures = Map.empty[Int, Int]
    val numClasses = 2
    val newRF = rf
      .setImpurity("Gini")
      .setMaxDepth(2)
      .setNumTrees(1)
      .setFeatureSubsetStrategy("auto")
      .setSeed(123)
    compareAPIs(orderedLabeledPoints50_1000, newRF, categoricalFeatures, numClasses)
  }

  test("params") {
    ParamsSuite.checkParams(new RandomForestClassifier)
    val model = new RandomForestClassificationModel("rfc",
      Array(new DecisionTreeClassificationModel("dtc", new LeafNode(0.0, 0.0, null), 2)), 2)
    ParamsSuite.checkParams(model)
  }

  test("Binary classification with continuous features:" +
    " comparing DecisionTree vs. RandomForest(numTrees = 1)") {
    val rf = new RandomForestClassifier()
    binaryClassificationTestWithContinuousFeatures(rf)
  }

  test("Binary classification with continuous features and node Id cache:" +
    " comparing DecisionTree vs. RandomForest(numTrees = 1)") {
    val rf = new RandomForestClassifier()
      .setCacheNodeIds(true)
    binaryClassificationTestWithContinuousFeatures(rf)
  }

  test("alternating categorical and continuous features with multiclass labels to test indexing") {
    val arr = Array(
      LabeledPoint(0.0, Vectors.dense(1.0, 0.0, 0.0, 3.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(0.0, 1.0, 1.0, 1.0, 2.0)),
      LabeledPoint(0.0, Vectors.dense(2.0, 0.0, 0.0, 6.0, 3.0)),
      LabeledPoint(2.0, Vectors.dense(0.0, 2.0, 1.0, 3.0, 2.0))
    )
    val rdd = sc.parallelize(arr)
    val categoricalFeatures = Map(0 -> 3, 2 -> 2, 4 -> 4)
    val numClasses = 3

    val rf = new RandomForestClassifier()
      .setImpurity("Gini")
      .setMaxDepth(5)
      .setNumTrees(2)
      .setFeatureSubsetStrategy("sqrt")
      .setSeed(12345)
    compareAPIs(rdd, rf, categoricalFeatures, numClasses)
  }

  test("subsampling rate in RandomForest"){
    val rdd = orderedLabeledPoints5_20
    val categoricalFeatures = Map.empty[Int, Int]
    val numClasses = 2

    val rf1 = new RandomForestClassifier()
      .setImpurity("Gini")
      .setMaxDepth(2)
      .setCacheNodeIds(true)
      .setNumTrees(3)
      .setFeatureSubsetStrategy("auto")
      .setSeed(123)
    compareAPIs(rdd, rf1, categoricalFeatures, numClasses)

    val rf2 = rf1.setSubsamplingRate(0.5)
    compareAPIs(rdd, rf2, categoricalFeatures, numClasses)
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests of feature importance
  /////////////////////////////////////////////////////////////////////////////
  test("Regression feature importance with toy data") {
    val numClasses = 2
    val newRF = new RandomForestClassifier()
      .setImpurity("Gini")
      .setMaxDepth(2)
      .setNumTrees(100)
      .setFeatureSubsetStrategy("auto")
      .setSeed(123)

    /* Verify results using SKLearn:

       from sklearn.ensemble import RandomForestClassifier
       X = np.array([
               [1, 0, 0, 0, 1],
               [0, 0, 0, 1, 0],
               [0, 0, 1, 0, 1],
               [1, 0, 0, 0, 0],
               [1, 1, 1, 0, 0]
           ])
       y = np.array([
               0,
               1,
               1,
               0,
               1
           ])
       classifier = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=2).fit(X,y)
       importances = classifier.feature_importances_
       std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
       indices = np.argsort(importances)[::-1]
       print("Feature importance:")
       for f in range(5):
       print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

       Feature importance:
       1. feature 2 (0.330000)
       2. feature 0 (0.304583)
       3. feature 3 (0.119167)
       4. feature 1 (0.111389)
       5. feature 4 (0.044861)
     */

    val data: RDD[LabeledPoint] = sc.parallelize(Seq(
      new LabeledPoint(0, Vectors.dense(1, 0, 0, 0, 1)),
      new LabeledPoint(1, Vectors.dense(0, 0, 0, 1, 0)),
      new LabeledPoint(1, Vectors.dense(0, 0, 1, 0, 1)),
      new LabeledPoint(0, Vectors.dense(1, 0, 0, 0, 0)),
      new LabeledPoint(1, Vectors.dense(1, 1, 1, 0, 0))
    ))
    val categoricalFeatures = Map.empty[Int, Int]
    val df: DataFrame = TreeTests.setMetadata(data, categoricalFeatures, numClasses)

    val result =  {
      val importance = newRF.fit(df).featureImportances.toArray
      Vectors.dense(importance.sortBy(_._1).map(_._2))
    }
    val expected = Vectors.dense(0.304583, 0.111389, 0.33, 0.119167, 0.044861)

    assert(result ~== expected absTol 0.02)
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests of model save/load
  /////////////////////////////////////////////////////////////////////////////

  // TODO: Reinstate test once save/load are implemented  SPARK-6725
  /*
  test("model save/load") {
    val tempDir = Utils.createTempDir()
    val path = tempDir.toURI.toString

    val trees =
      Range(0, 3).map(_ => OldDecisionTreeSuite.createModel(OldAlgo.Classification)).toArray
    val oldModel = new OldRandomForestModel(OldAlgo.Classification, trees)
    val newModel = RandomForestClassificationModel.fromOld(oldModel)

    // Save model, load it back, and compare.
    try {
      newModel.save(sc, path)
      val sameNewModel = RandomForestClassificationModel.load(sc, path)
      TreeTests.checkEqual(newModel, sameNewModel)
    } finally {
      Utils.deleteRecursively(tempDir)
    }
  }
  */
}

private object RandomForestClassifierSuite {

  /**
   * Train 2 models on the given dataset, one using the old API and one using the new API.
   * Convert the old model to the new format, compare them, and fail if they are not exactly equal.
   */
  def compareAPIs(
      data: RDD[LabeledPoint],
      rf: RandomForestClassifier,
      categoricalFeatures: Map[Int, Int],
      numClasses: Int): Unit = {
    val oldStrategy =
      rf.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, rf.getOldImpurity)
    val oldModel = OldRandomForest.trainClassifier(
      data, oldStrategy, rf.getNumTrees, rf.getFeatureSubsetStrategy, rf.getSeed.toInt)
    val newData: DataFrame = TreeTests.setMetadata(data, categoricalFeatures, numClasses)
    val newModel = rf.fit(newData)
    // Use parent from newTree since this is not checked anyways.
    val oldModelAsNew = RandomForestClassificationModel.fromOld(
      oldModel, newModel.parent.asInstanceOf[RandomForestClassifier], categoricalFeatures,
      numClasses)
    TreeTests.checkEqual(oldModelAsNew, newModel)
    assert(newModel.hasParent)
    assert(!newModel.trees.head.asInstanceOf[DecisionTreeClassificationModel].hasParent)
    assert(newModel.numClasses == numClasses)
    val results = newModel.transform(newData)
    results.select("rawPrediction", "prediction").collect().foreach {
      case Row(raw: Vector, prediction: Double) => {
        assert(raw.size == numClasses)
        val predFromRaw = raw.toArray.zipWithIndex.maxBy(_._1)._2
        assert(predFromRaw == prediction)
      }
    }
  }
}
