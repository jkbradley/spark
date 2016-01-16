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

package org.apache.spark.mllib.classification

import scala.collection.JavaConverters._
import scala.util.Random
import scala.util.control.Breaks._

import org.scalatest.Matchers

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.{LocalClusterSparkContext, MLlibTestSparkContext}
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.util.Utils


object LogisticRegressionSuite {

  def generateLogisticInputAsList(
    offset: Double,
    scale: Double,
    nPoints: Int,
    seed: Int): java.util.List[LabeledPoint] = {
    generateLogisticInput(offset, scale, nPoints, seed).asJava
  }

  // Generate input of the form Y = logistic(offset + scale*X)
  def generateLogisticInput(
      offset: Double,
      scale: Double,
      nPoints: Int,
      seed: Int): Seq[LabeledPoint] = {
    val rnd = new Random(seed)
    val x1 = Array.fill[Double](nPoints)(rnd.nextGaussian())

    val y = (0 until nPoints).map { i =>
      val p = 1.0 / (1.0 + math.exp(-(offset + scale * x1(i))))
      if (rnd.nextDouble() < p) 1.0 else 0.0
    }

    val testData = (0 until nPoints).map(i => LabeledPoint(y(i), Vectors.dense(Array(x1(i)))))
    testData
  }

  /**
   * Generates `k` classes multinomial synthetic logistic input in `n` dimensional space given the
   * model weights and mean/variance of the features. The synthetic data will be drawn from
   * the probability distribution constructed by weights using the following formula.
   *
   * P(y = 0 | x) = 1 / norm
   * P(y = 1 | x) = exp(x * w_1) / norm
   * P(y = 2 | x) = exp(x * w_2) / norm
   * ...
   * P(y = k-1 | x) = exp(x * w_{k-1}) / norm
   * where norm = 1 + exp(x * w_1) + exp(x * w_2) + ... + exp(x * w_{k-1})
   *
   * @param weights matrix is flatten into a vector; as a result, the dimension of weights vector
   *                will be (k - 1) * (n + 1) if `addIntercept == true`, and
   *                if `addIntercept != true`, the dimension will be (k - 1) * n.
   * @param xMean the mean of the generated features. Lots of time, if the features are not properly
   *              standardized, the algorithm with poor implementation will have difficulty
   *              to converge.
   * @param xVariance the variance of the generated features.
   * @param addIntercept whether to add intercept.
   * @param nPoints the number of instance of generated data.
   * @param seed the seed for random generator. For consistent testing result, it will be fixed.
   */
  def generateMultinomialLogisticInput(
      weights: Array[Double],
      xMean: Array[Double],
      xVariance: Array[Double],
      addIntercept: Boolean,
      nPoints: Int,
      seed: Int): Seq[LabeledPoint] = {
    val rnd = new Random(seed)

    val xDim = xMean.length
    val xWithInterceptsDim = if (addIntercept) xDim + 1 else xDim
    val nClasses = weights.length / xWithInterceptsDim + 1

    val x = Array.fill[Vector](nPoints)(Vectors.dense(Array.fill[Double](xDim)(rnd.nextGaussian())))

    x.foreach { vector =>
      // This doesn't work if `vector` is a sparse vector.
      val vectorArray = vector.toArray
      var i = 0
      val len = vectorArray.length
      while (i < len) {
        vectorArray(i) = vectorArray(i) * math.sqrt(xVariance(i)) + xMean(i)
        i += 1
      }
    }

    val y = (0 until nPoints).map { idx =>
      val xArray = x(idx).toArray
      val margins = Array.ofDim[Double](nClasses)
      val probs = Array.ofDim[Double](nClasses)

      for (i <- 0 until nClasses - 1) {
        for (j <- 0 until xDim) margins(i + 1) += weights(i * xWithInterceptsDim + j) * xArray(j)
        if (addIntercept) margins(i + 1) += weights((i + 1) * xWithInterceptsDim - 1)
      }
      // Preventing the overflow when we compute the probability
      val maxMargin = margins.max
      if (maxMargin > 0) for (i <- 0 until nClasses) margins(i) -= maxMargin

      // Computing the probabilities for each class from the margins.
      val norm = {
        var temp = 0.0
        for (i <- 0 until nClasses) {
          probs(i) = math.exp(margins(i))
          temp += probs(i)
        }
        temp
      }
      for (i <- 0 until nClasses) probs(i) /= norm

      // Compute the cumulative probability so we can generate a random number and assign a label.
      for (i <- 1 until nClasses) probs(i) += probs(i - 1)
      val p = rnd.nextDouble()
      var y = 0
      breakable {
        for (i <- 0 until nClasses) {
          if (p < probs(i)) {
            y = i
            break
          }
        }
      }
      y
    }

    val testData = (0 until nPoints).map(i => LabeledPoint(y(i), x(i)))
    testData
  }

  /** Binary labels, 3 features */
  private val binaryModel = new LogisticRegressionModel(
    weights = Vectors.dense(0.1, 0.2, 0.3), intercept = 0.5, numFeatures = 3, numClasses = 2)

  /** 3 classes, 2 features */
  private val multiclassModel = new LogisticRegressionModel(
    weights = Vectors.dense(0.1, 0.2, 0.3, 0.4), intercept = 1.0, numFeatures = 2, numClasses = 3)

  private def checkModelsEqual(a: LogisticRegressionModel, b: LogisticRegressionModel): Unit = {
    assert(a.weights == b.weights)
    assert(a.intercept == b.intercept)
    assert(a.numClasses == b.numClasses)
    assert(a.numFeatures == b.numFeatures)
    assert(a.getThreshold == b.getThreshold)
  }
}
