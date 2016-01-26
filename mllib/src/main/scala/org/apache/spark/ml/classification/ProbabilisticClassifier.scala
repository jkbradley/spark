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

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors, VectorUDT}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/**
 * (private[classification])  Params for probabilistic classification.
 */
private[classification] trait ProbabilisticClassifierParams
  extends ClassifierParams with HasProbabilityCol with HasThresholds


/**
 * :: DeveloperApi ::
 *
 * Single-label binary or multiclass classifier which can output class conditional probabilities.
 *
 * @tparam E  Concrete Estimator type
 * @tparam M  Concrete Model type
 */
@DeveloperApi
abstract class ProbabilisticClassifier[
    E <: ProbabilisticClassifier[E, M],
    M <: ProbabilisticClassificationModel[M]]
  extends Classifier[E, M] with ProbabilisticClassifierParams {

  override protected def transformSchemaImpl(schema: StructType): StructType = {
    val schema2 = super.transformSchemaImpl(schema)
    if (isDefined(probabilityCol)) {
      SchemaUtils.appendColumn(schema2, $(probabilityCol), new VectorUDT)
    } else {
      schema2
    }
  }

  /** @group setParam */
  def setProbabilityCol(value: String): E = set(probabilityCol, value).asInstanceOf[E]

  /** @group setParam */
  def setThresholds(value: Array[Double]): E = set(thresholds, value).asInstanceOf[E]
}


/**
 * :: DeveloperApi ::
 *
 * Model produced by a [[ProbabilisticClassifier]].
 * Classes are indexed {0, 1, ..., numClasses - 1}.
 *
 * @tparam M  Concrete Model type
 */
@DeveloperApi
abstract class ProbabilisticClassificationModel[
    M <: ProbabilisticClassificationModel[M]]
  extends ClassificationModel[M] with ProbabilisticClassifierParams {

  /** @group setParam */
  def setProbabilityCol(value: String): M = set(probabilityCol, value).asInstanceOf[M]

  /** @group setParam */
  def setThresholds(value: Array[Double]): M = set(thresholds, value).asInstanceOf[M]

  override def validateParams(): Unit = {
    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".transform() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }
  }

  override protected def transformSchemaImpl(schema: StructType): StructType = {
    val schema2 = super.transformSchemaImpl(schema)
    if (isDefined(probabilityCol)) {
      SchemaUtils.appendColumn(schema2, $(probabilityCol), new VectorUDT)
    } else {
      schema2
    }
  }

  /**
   * Transforms dataset by reading from [[featuresCol]], and appending new columns as specified by
   * parameters:
   *  - predicted labels as [[predictionCol]] of type [[Double]]
   *  - raw predictions (confidences) as [[rawPredictionCol]] of type [[Vector]]
   *  - probability of each class as [[probabilityCol]] of type [[Vector]].
   *
   * @param dataset input dataset
   * @return transformed dataset
   */
  override protected def transformImpl(dataset: DataFrame): DataFrame = {
    // Output selected columns only.
    // This is a bit complicated since it tries to avoid repeated computation.
    var outputData = dataset
    var numColsOutput = 0
    if (isDefined(rawPredictionCol)) {
      val predictRawUDF = udf { (features: Vector) => predictRaw(features) }
      outputData = outputData.withColumn(getRawPredictionCol, predictRawUDF(col(getFeaturesCol)))
      numColsOutput += 1
    }
    if (isDefined(probabilityCol)) {
      val probUDF = if (isDefined(rawPredictionCol)) {
        udf(raw2probability _).apply(col($(rawPredictionCol)))
      } else {
        val probabilityUDF = udf { (features: Vector) => predictProbability(features) }
        probabilityUDF(col($(featuresCol)))
      }
      outputData = outputData.withColumn($(probabilityCol), probUDF)
      numColsOutput += 1
    }
    if (isDefined(predictionCol)) {
      val predUDF = if (isDefined(rawPredictionCol)) {
        udf(raw2prediction _).apply(col($(rawPredictionCol)))
      } else if (isDefined(probabilityCol)) {
        udf(probability2prediction _).apply(col($(probabilityCol)))
      } else {
        val predictUDF = udf { (features: Vector) => predict(features) }
        predictUDF(col($(featuresCol)))
      }
      outputData = outputData.withColumn($(predictionCol), predUDF)
      numColsOutput += 1
    }

    if (numColsOutput == 0) {
      this.logWarning(s"$uid: ProbabilisticClassificationModel.transform() was called as NOOP" +
        " since no output columns were set.")
    }
    outputData
  }

  /**
   * Estimate the probability of each class given the raw prediction,
   * doing the computation in-place.
   * These predictions are also called class conditional probabilities.
   *
   * This internal method is used to implement [[transform()]] and output [[probabilityCol]].
   *
   * @return Estimated class conditional probabilities (modified input vector)
   */
  protected def raw2probabilityInPlace(rawPrediction: Vector): Vector

  /** Non-in-place version of [[raw2probabilityInPlace()]] */
  protected def raw2probability(rawPrediction: Vector): Vector = {
    val probs = rawPrediction.copy
    raw2probabilityInPlace(probs)
  }

  override protected def raw2prediction(rawPrediction: Vector): Double = {
    if (!isDefined(thresholds)) {
      rawPrediction.argmax
    } else {
      probability2prediction(raw2probability(rawPrediction))
    }
  }

  /**
   * Predict the probability of each class given the features.
   * These predictions are also called class conditional probabilities.
   *
   * This internal method is used to implement [[transform()]] and output [[probabilityCol]].
   *
   * @return Estimated class conditional probabilities
   */
  protected def predictProbability(features: Vector): Vector = {
    val rawPreds = predictRaw(features)
    raw2probabilityInPlace(rawPreds)
  }

  /**
   * Given a vector of class conditional probabilities, select the predicted label.
   * This supports thresholds which favor particular labels.
   * @return  predicted label
   */
  protected def probability2prediction(probability: Vector): Double = {
    if (!isDefined(thresholds)) {
      probability.argmax
    } else {
      val thresholds: Array[Double] = getThresholds
      val scaledProbability: Array[Double] =
        probability.toArray.zip(thresholds).map { case (p, t) =>
          if (t == 0.0) Double.PositiveInfinity else p / t
        }
      Vectors.dense(scaledProbability).argmax
    }
  }
}

private[ml] object ProbabilisticClassificationModel {

  /**
   * Normalize a vector of raw predictions to be a multinomial probability vector, in place.
   *
   * The input raw predictions should be >= 0.
   * The output vector sums to 1, unless the input vector is all-0 (in which case the output is
   * all-0 too).
   *
   * NOTE: This is NOT applicable to all models, only ones which effectively use class
   *       instance counts for raw predictions.
   */
  def normalizeToProbabilitiesInPlace(v: DenseVector): Unit = {
    val sum = v.values.sum
    if (sum != 0) {
      var i = 0
      val size = v.size
      while (i < size) {
        v.values(i) /= sum
        i += 1
      }
    }
  }
}
