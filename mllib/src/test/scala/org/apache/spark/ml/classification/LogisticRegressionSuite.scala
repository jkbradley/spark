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

import scala.language.existentials
import scala.util.Random

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.util.{DefaultReadWriteTest, Identifiable, MLTestingUtils}
import org.apache.spark.mllib.classification.LogisticRegressionSuite._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.sql.{DataFrame, Row}

class LogisticRegressionSuite
  extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  @transient var dataset: DataFrame = _
  @transient var binaryDataset: DataFrame = _
  private val eps: Double = 1e-5

  override def beforeAll(): Unit = {
    super.beforeAll()

    dataset = sqlContext.createDataFrame(generateLogisticInput(1.0, 1.0, nPoints = 100, seed = 42))

    /*
       Here is the instruction describing how to export the test data into CSV format
       so we can validate the training accuracy compared with R's glmnet package.

       import org.apache.spark.mllib.classification.LogisticRegressionSuite
       val nPoints = 10000
       val coefficients = Array(-0.57997, 0.912083, -0.371077, -0.819866, 2.688191)
       val xMean = Array(5.843, 3.057, 3.758, 1.199)
       val xVariance = Array(0.6856, 0.1899, 3.116, 0.581)
       val data = sc.parallelize(LogisticRegressionSuite.generateMultinomialLogisticInput(
         coefficients, xMean, xVariance, true, nPoints, 42), 1)
       data.map(x=> x.label + ", " + x.features(0) + ", " + x.features(1) + ", "
         + x.features(2) + ", " + x.features(3)).saveAsTextFile("path")
     */
    binaryDataset = {
      val nPoints = 10000
      val coefficients = Array(-0.57997, 0.912083, -0.371077, -0.819866, 2.688191)
      val xMean = Array(5.843, 3.057, 3.758, 1.199)
      val xVariance = Array(0.6856, 0.1899, 3.116, 0.581)

      val testData =
        generateMultinomialLogisticInput(coefficients, xMean, xVariance, true, nPoints, 42)

      sqlContext.createDataFrame(sc.parallelize(testData, 4))
    }
  }

  test("params") {
    ParamsSuite.checkParams(new LogisticRegression)
    val model = new LogisticRegressionModel("logReg", Vectors.dense(0.0), 0.0)
    ParamsSuite.checkParams(model)
  }

  test("logistic regression: default params") {
    val lr = new LogisticRegression
    assert(lr.getLabelCol === "label")
    assert(lr.getFeaturesCol === "features")
    assert(lr.getPredictionCol === "prediction")
    assert(lr.getRawPredictionCol === "rawPrediction")
    assert(lr.getProbabilityCol === "probability")
    assert(lr.getWeightCol === "")
    assert(lr.getFitIntercept)
    assert(lr.getStandardization)
    val model = lr.fit(dataset)
    model.transform(dataset)
      .select("label", "probability", "prediction", "rawPrediction")
      .collect()
    assert(model.getThreshold === 0.5)
    assert(model.getFeaturesCol === "features")
    assert(model.getPredictionCol === "prediction")
    assert(model.getRawPredictionCol === "rawPrediction")
    assert(model.getProbabilityCol === "probability")
    assert(model.intercept !== 0.0)
    assert(model.hasParent)
  }

  test("empty probabilityCol") {
    val lr = new LogisticRegression().setProbabilityCol("")
    val model = lr.fit(dataset)
    assert(model.hasSummary)
    // Validate that we re-insert a probability column for evaluation
    val fieldNames = model.summary.predictions.schema.fieldNames
    assert((dataset.schema.fieldNames.toSet).subsetOf(
      fieldNames.toSet))
    assert(fieldNames.exists(s => s.startsWith("probability_")))
  }

  test("setThreshold, getThreshold") {
    val lr = new LogisticRegression
    // default
    assert(lr.getThreshold === 0.5, "LogisticRegression.threshold should default to 0.5")
    withClue("LogisticRegression should not have thresholds set by default.") {
      intercept[java.util.NoSuchElementException] { // Note: The exception type may change in future
        lr.getThresholds
      }
    }
    // Set via threshold.
    // Intuition: Large threshold or large thresholds(1) makes class 0 more likely.
    lr.setThreshold(1.0)
    assert(lr.getThresholds === Array(0.0, 1.0))
    lr.setThreshold(0.0)
    assert(lr.getThresholds === Array(1.0, 0.0))
    lr.setThreshold(0.5)
    assert(lr.getThresholds === Array(0.5, 0.5))
    // Set via thresholds
    val lr2 = new LogisticRegression
    lr2.setThresholds(Array(0.3, 0.7))
    val expectedThreshold = 1.0 / (1.0 + 0.3 / 0.7)
    assert(lr2.getThreshold ~== expectedThreshold relTol 1E-7)
    // thresholds and threshold must be consistent
    lr2.setThresholds(Array(0.1, 0.2, 0.3))
    withClue("getThreshold should throw error if thresholds has length != 2.") {
      intercept[IllegalArgumentException] {
        lr2.getThreshold
      }
    }
    // thresholds and threshold must be consistent: values
    withClue("fit with ParamMap should throw error if threshold, thresholds do not match.") {
      intercept[IllegalArgumentException] {
        val lr2model = lr2.fit(dataset,
          lr2.thresholds -> Array(0.3, 0.7), lr2.threshold -> (expectedThreshold / 2.0))
        lr2model.getThreshold
      }
    }
  }

  test("logistic regression doesn't fit intercept when fitIntercept is off") {
    val lr = new LogisticRegression
    lr.setFitIntercept(false)
    val model = lr.fit(dataset)
    assert(model.intercept === 0.0)

    // copied model must have the same parent.
    MLTestingUtils.checkCopy(model)
  }

  test("logistic regression with setters") {
    // Set params, train, and check as many params as we can.
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(1.0)
      .setThreshold(0.6)
      .setProbabilityCol("myProbability")
    val model = lr.fit(dataset)
    val parent = model.parent.asInstanceOf[LogisticRegression]
    assert(parent.getMaxIter === 10)
    assert(parent.getRegParam === 1.0)
    assert(parent.getThreshold === 0.6)
    assert(model.getThreshold === 0.6)

    // Modify model params, and check that the params worked.
    model.setThreshold(1.0)
    val predAllZero = model.transform(dataset)
      .select("prediction", "myProbability")
      .collect()
      .map { case Row(pred: Double, prob: Vector) => pred }
    assert(predAllZero.forall(_ === 0),
      s"With threshold=1.0, expected predictions to be all 0, but only" +
      s" ${predAllZero.count(_ === 0)} of ${dataset.count()} were 0.")
    // Call transform with params, and check that the params worked.
    val predNotAllZero =
      model.transform(dataset, model.threshold -> 0.0,
        model.probabilityCol -> "myProb")
        .select("prediction", "myProb")
        .collect()
        .map { case Row(pred: Double, prob: Vector) => pred }
    assert(predNotAllZero.exists(_ !== 0.0))

    // Call fit() with new params, and check as many params as we can.
    lr.setThresholds(Array(0.6, 0.4))
    val model2 = lr.fit(dataset, lr.maxIter -> 5, lr.regParam -> 0.1,
      lr.probabilityCol -> "theProb")
    val parent2 = model2.parent.asInstanceOf[LogisticRegression]
    assert(parent2.getMaxIter === 5)
    assert(parent2.getRegParam === 0.1)
    assert(parent2.getThreshold === 0.4)
    assert(model2.getThreshold === 0.4)
    assert(model2.getProbabilityCol === "theProb")
  }

  test("logistic regression: Predictor, Classifier methods") {
    val sqlContext = this.sqlContext
    val lr = new LogisticRegression

    val model = lr.fit(dataset)
    assert(model.numClasses === 2)
    val numFeatures = dataset.select("features").first().getAs[Vector](0).size
    assert(model.numFeatures === numFeatures)

    val threshold = model.getThreshold
    val results = model.transform(dataset)

    // Compare rawPrediction with probability
    results.select("rawPrediction", "probability").collect().foreach {
      case Row(raw: Vector, prob: Vector) =>
        assert(raw.size === 2)
        assert(prob.size === 2)
        val probFromRaw1 = 1.0 / (1.0 + math.exp(-raw(1)))
        assert(prob(1) ~== probFromRaw1 relTol eps)
        assert(prob(0) ~== 1.0 - probFromRaw1 relTol eps)
    }

    // Compare prediction with probability
    results.select("prediction", "probability").collect().foreach {
      case Row(pred: Double, prob: Vector) =>
        val predFromProb = prob.toArray.zipWithIndex.maxBy(_._1)._2
        assert(pred == predFromProb)
    }
  }

  test("MultiClassSummarizer") {
    val summarizer1 = (new MultiClassSummarizer)
      .add(0.0).add(3.0).add(4.0).add(3.0).add(6.0)
    assert(summarizer1.histogram === Array[Double](1, 0, 0, 2, 1, 0, 1))
    assert(summarizer1.countInvalid === 0)
    assert(summarizer1.numClasses === 7)

    val summarizer2 = (new MultiClassSummarizer)
      .add(1.0).add(5.0).add(3.0).add(0.0).add(4.0).add(1.0)
    assert(summarizer2.histogram === Array[Double](1, 2, 0, 1, 1, 1))
    assert(summarizer2.countInvalid === 0)
    assert(summarizer2.numClasses === 6)

    val summarizer3 = (new MultiClassSummarizer)
      .add(0.0).add(1.3).add(5.2).add(2.5).add(2.0).add(4.0).add(4.0).add(4.0).add(1.0)
    assert(summarizer3.histogram === Array[Double](1, 1, 1, 0, 3))
    assert(summarizer3.countInvalid === 3)
    assert(summarizer3.numClasses === 5)

    val summarizer4 = (new MultiClassSummarizer)
      .add(3.1).add(4.3).add(2.0).add(1.0).add(3.0)
    assert(summarizer4.histogram === Array[Double](0, 1, 1, 1))
    assert(summarizer4.countInvalid === 2)
    assert(summarizer4.numClasses === 4)

    // small map merges large one
    val summarizerA = summarizer1.merge(summarizer2)
    assert(summarizerA.hashCode() === summarizer2.hashCode())
    assert(summarizerA.histogram === Array[Double](2, 2, 0, 3, 2, 1, 1))
    assert(summarizerA.countInvalid === 0)
    assert(summarizerA.numClasses === 7)

    // large map merges small one
    val summarizerB = summarizer3.merge(summarizer4)
    assert(summarizerB.hashCode() === summarizer3.hashCode())
    assert(summarizerB.histogram === Array[Double](1, 2, 2, 1, 3))
    assert(summarizerB.countInvalid === 5)
    assert(summarizerB.numClasses === 5)
  }

  test("MultiClassSummarizer with weighted samples") {
    val summarizer1 = (new MultiClassSummarizer)
      .add(label = 0.0, weight = 0.2).add(3.0, 0.8).add(4.0, 3.2).add(3.0, 1.3).add(6.0, 3.1)
    assert(Vectors.dense(summarizer1.histogram) ~==
      Vectors.dense(Array(0.2, 0, 0, 2.1, 3.2, 0, 3.1)) absTol 1E-10)
    assert(summarizer1.countInvalid === 0)
    assert(summarizer1.numClasses === 7)

    val summarizer2 = (new MultiClassSummarizer)
      .add(1.0, 1.1).add(5.0, 2.3).add(3.0).add(0.0).add(4.0).add(1.0).add(2, 0.0)
    assert(Vectors.dense(summarizer2.histogram) ~==
      Vectors.dense(Array[Double](1.0, 2.1, 0.0, 1, 1, 2.3)) absTol 1E-10)
    assert(summarizer2.countInvalid === 0)
    assert(summarizer2.numClasses === 6)

    val summarizer = summarizer1.merge(summarizer2)
    assert(Vectors.dense(summarizer.histogram) ~==
      Vectors.dense(Array(1.2, 2.1, 0.0, 3.1, 4.2, 2.3, 3.1)) absTol 1E-10)
    assert(summarizer.countInvalid === 0)
    assert(summarizer.numClasses === 7)
  }

  test("read/write") {
    def checkModelData(model: LogisticRegressionModel, model2: LogisticRegressionModel): Unit = {
      assert(model.intercept === model2.intercept)
      assert(model.coefficients.toArray === model2.coefficients.toArray)
      assert(model.numClasses === model2.numClasses)
      assert(model.numFeatures === model2.numFeatures)
    }
    val lr = new LogisticRegression()
    testEstimatorAndModelReadWrite(lr, dataset, LogisticRegressionSuite.allParamSettings,
      checkModelData)
  }
}

object LogisticRegressionSuite {

  /**
   * Mapping from all Params to valid settings which differ from the defaults.
   * This is useful for tests which need to exercise all Params, such as save/load.
   * This excludes input columns to simplify some tests.
   */
  val allParamSettings: Map[String, Any] = ProbabilisticClassifierSuite.allParamSettings ++ Map(
    "probabilityCol" -> "myProbability",
    "thresholds" -> Array(0.4, 0.6),
    "regParam" -> 0.01,
    "elasticNetParam" -> 0.1,
    "maxIter" -> 2,  // intentionally small
    "fitIntercept" -> true,
    "tol" -> 0.8,
    "standardization" -> false,
    "threshold" -> 0.6
  )
}
