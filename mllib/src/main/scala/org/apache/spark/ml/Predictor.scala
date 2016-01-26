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

package org.apache.spark.ml

import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.mllib.linalg.{Vector, VectorUDT}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataTypes, DoubleType, StructType}

/**
 * (private[ml])  Trait for parameters for prediction (regression and classification).
 */
private[ml] trait PredictorParams extends PipelineStage
  with HasLabelCol with HasFeaturesCol with HasPredictionCol {

  // TODO: Support casting Array[Double] and Array[Float] to Vector when FeaturesType = Vector
  setInputColDataType(labelCol, Seq(DataTypes.DoubleType))
  setUseForFitOnly(labelCol)
  setInputColDataType(featuresCol, Seq(new VectorUDT))
}

/**
 * :: DeveloperApi ::
 * Abstraction for prediction problems (regression and classification).
 *
 * @tparam Learner  Specialization of this class.  If you subclass this type, use this type
 *                  parameter to specify the concrete type.
 * @tparam M  Specialization of [[PredictionModel]].  If you subclass this type, use this type
 *            parameter to specify the concrete type for the corresponding model.
 */
@DeveloperApi
abstract class Predictor[
    Learner <: Predictor[Learner, M],
    M <: PredictionModel[M]]
  extends Estimator[M] with PredictorParams {

  /** @group setParam */
  def setLabelCol(value: String): Learner = set(labelCol, value).asInstanceOf[Learner]

  /** @group setParam */
  def setFeaturesCol(value: String): Learner = set(featuresCol, value).asInstanceOf[Learner]

  /** @group setParam */
  def setPredictionCol(value: String): Learner = set(predictionCol, value).asInstanceOf[Learner]

  override def copy(extra: ParamMap): Learner

  override protected def transformSchemaImpl(schema: StructType): StructType = {
    if (isDefined(predictionCol)) {
      SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
    } else {
      schema
    }
  }

  /**
   * Extract [[labelCol]] and [[featuresCol]] from the given dataset,
   * and put it in an RDD with strong types.
   */
  protected def extractLabeledPoints(dataset: DataFrame): RDD[LabeledPoint] = {
    dataset.select($(labelCol), $(featuresCol))
      .map { case Row(label: Double, features: Vector) => LabeledPoint(label, features) }
  }
}

/**
 * :: DeveloperApi ::
 * Abstraction for a model for prediction tasks (regression and classification).
 *
 * @tparam M  Specialization of [[PredictionModel]].  If you subclass this type, use this type
 *            parameter to specify the concrete type for the corresponding model.
 */
@DeveloperApi
abstract class PredictionModel[M <: PredictionModel[M]]
  extends Model[M] with PredictorParams {

  /** @group setParam */
  def setFeaturesCol(value: String): M = set(featuresCol, value).asInstanceOf[M]

  /** @group setParam */
  def setPredictionCol(value: String): M = set(predictionCol, value).asInstanceOf[M]

  /** Returns the number of features the model was trained on. If unknown, returns -1 */
  @Since("1.6.0")
  def numFeatures: Int = -1

  override protected def transformSchemaImpl(schema: StructType): StructType = {
    if (isDefined(predictionCol)) {
      SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
    } else {
      schema
    }
  }

  /**
   * Transforms dataset by reading from [[featuresCol]], calling [[predict()]], and storing
   * the predictions as a new column [[predictionCol]].
   *
   * @param dataset input dataset
   * @return transformed dataset with [[predictionCol]] of type [[Double]]
   */
  override protected def transformImpl(dataset: DataFrame): DataFrame = {
    if (isDefined(predictionCol)) {
      val predictUDF = udf { (features: Vector) => predict(features) }
      dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
    } else {
      dataset
    }
  }

  /**
   * Predict label for the given features.
   * This internal method is used to implement [[transform()]] and output [[predictionCol]].
   */
  protected def predict(features: Vector): Double
}
