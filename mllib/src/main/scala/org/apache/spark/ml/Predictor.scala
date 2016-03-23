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
import org.apache.spark.mllib.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

/**
 * (private[ml])  Trait for parameters for prediction (regression and classification).
 */
private[ml] trait PredictorParams extends Params
  with HasLabelCol with HasFeaturesCol with HasPredictionCol {

  /**
   * Validates and transforms the input schema with the provided param map.
   * @param schema input schema
   * @param fitting whether this is in fitting
   * @param featuresDataType  SQL DataType for FeaturesType.
   *                          E.g., [[org.apache.spark.mllib.linalg.VectorUDT]] for vector features.
   * @return output schema
   */
  protected def validateAndTransformSchema(
      schema: StructType,
      fitting: Boolean,
      featuresDataType: DataType): StructType = {
    // TODO: Support casting Array[Double] and Array[Float] to Vector when FeaturesType = Vector
    SchemaUtils.checkColumnType(schema, $(featuresCol), featuresDataType)
    if (fitting) {
      // TODO: Allow other numeric types
      SchemaUtils.checkColumnType(schema, $(labelCol), DoubleType)
    }
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }
}

/**
 * :: DeveloperApi ::
 * Abstraction for prediction problems (regression and classification).
 *
 * @tparam FeaturesType  Type of features.
 *                       E.g., [[org.apache.spark.mllib.linalg.VectorUDT]] for vector features.
 * @tparam Learner  Specialization of this class.  If you subclass this type, use this type
 *                  parameter to specify the concrete type.
 * @tparam M  Specialization of [[PredictionModel]].  If you subclass this type, use this type
 *            parameter to specify the concrete type for the corresponding model.
 */
@DeveloperApi
abstract class Predictor[
    FeaturesType,
    Learner <: Predictor[FeaturesType, Learner, M],
    M <: PredictionModel[FeaturesType, M]]
  extends Estimator[M] with PredictorParams {

  /** @group setParam */
  def setLabelCol(value: String): Learner = set(labelCol, value).asInstanceOf[Learner]

  /** @group setParam */
  def setFeaturesCol(value: String): Learner = set(featuresCol, value).asInstanceOf[Learner]

  /** @group setParam */
  def setPredictionCol(value: String): Learner = set(predictionCol, value).asInstanceOf[Learner]

  override def fit(dataset: DataFrame): M = {
    // This handles a few items such as schema validation.
    // Developers only need to implement train().
    transformSchema(dataset.schema, logging = true)
    copyValues(train(dataset).setParent(this))
  }

  override def copy(extra: ParamMap): Learner

  /**
   * Train a model using the given dataset and parameters.
   * Developers can implement this instead of [[fit()]] to avoid dealing with schema validation
   * and copying parameters into the model.
   *
   * @param dataset  Training dataset
   * @return  Fitted model
   */
  protected def train(dataset: DataFrame): M

  /**
   * Returns the SQL DataType corresponding to the FeaturesType type parameter.
   *
   * This is used by [[validateAndTransformSchema()]].
   * This workaround is needed since SQL has different APIs for Scala and Java.
   *
   * The default value is VectorUDT, but it may be overridden if FeaturesType is not Vector.
   */
  private[ml] def featuresDataType: DataType = new VectorUDT

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema, fitting = true, featuresDataType)
  }

  /**
   * Extract [[labelCol]] and [[featuresCol]] from the given dataset,
   * and put it in an RDD with strong types.
   */
  protected def extractLabeledPoints(dataset: DataFrame): RDD[LabeledPoint] = {
    dataset.select($(labelCol), $(featuresCol))
      .map { case Row(label: Double, features: Vector) => LabeledPoint(label, features) }
  }

  /**
    * Extract [[labelCol]] and [[featuresCol]] from the given dataset,
    * and transpose [[featuresCol]] to store data by column, instead of
    * by row. Returns a 2-tuple of (columns, labels).
    */
  protected def transpose(df: DataFrame): (RDD[Vector], RDD[Double]) = {
    df.sqlContext.setConf("spark.sql.retainGroupColumns", "false")

    // UDF that maps (row, rowIdx) to (colIdx -> (row, rowIdx))
    val transpose = udf { (features: Vector, rowIndex: Long) =>
      features.toArray.zipWithIndex.map { case(rowValue, colIndex) =>
        colIndex -> (rowValue, rowIndex)
      }
    }

    val aggUDF = new SortArraysUDAF

    val labels = df.select(col($(labelCol))).map(r => r.getDouble(0))

    val columns = df.withColumn("rowIdx", monotonicallyIncreasingId()) // add rowIdx to sort by row
      // map to Array[(colIdx, (rowValue, rowIdx))]
      .select(transpose(col($(featuresCol)), col("rowIdx")).as($(featuresCol)))
      // flatten to (colIdx, (rowValue, rowIdx))
      .select(explode(col($(featuresCol))).as($(featuresCol)))
      // split colIdx and (rowValue, rowIdx) into separate DataFrame columns
      .select(col(s"${$(featuresCol)}._1").as("colIdx"), col(s"${$(featuresCol)}._2").as("rows"))
      // group by column index, aggregate (rowValue, rowIdx) into a sorted Array of rowValues
      .groupBy("colIdx").agg(aggUDF(col("rows")).as($(featuresCol)))
      // convert from Array to Vector
      .select(col($(featuresCol))).map { row =>
        Vectors.dense(row.getSeq[Double](0).toArray)
      }

    (columns, labels)
  }

  class SortArraysUDAF extends UserDefinedAggregateFunction {
    // |-- rows: struct (nullable = true)
    // |    |-- _1: double (nullable = false)
    // |    |-- _2: long (nullable = false)
    def inputSchema: org.apache.spark.sql.types.StructType =
      StructType(StructField("rows",
          StructType(
            StructType(StructField("_1", DoubleType) :: StructField("_2", LongType)
              :: Nil)
          )
        )
      :: Nil)

    def bufferSchema: StructType = StructType(
      StructField("values", ArrayType(DoubleType)) ::
        StructField("rowIndexes", ArrayType(LongType)) :: Nil
    )

    def dataType: DataType = ArrayType(DoubleType, containsNull = false)

    def deterministic: Boolean = true

    def initialize(buffer: MutableAggregationBuffer): Unit = {
      buffer(0) = Array[Double]()
      buffer(1) = Array[Long]()
    }

    def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
      val row = input.getStruct(0)
      buffer(0) = buffer.getSeq[Double](0) ++ Array(row.getDouble(0))
      buffer(1) = buffer.getSeq[Long](1) ++ Array(row.getLong(1))
    }

    def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
      buffer1(0) = buffer1.getSeq[Double](0) ++ buffer2.getSeq[Double](0)
      buffer1(1) = buffer1.getSeq[Long](1) ++ buffer2.getSeq[Long](1)
    }

    def evaluate(buffer: Row): Any = {
      val arr = buffer.getSeq[Double](0)
      val rowIndexes = buffer.getSeq[Long](1)
      arr.zip(rowIndexes).sortBy(_._2).unzip._1
    }
  }
}

/**
 * :: DeveloperApi ::
 * Abstraction for a model for prediction tasks (regression and classification).
 *
 * @tparam FeaturesType  Type of features.
 *                       E.g., [[org.apache.spark.mllib.linalg.VectorUDT]] for vector features.
 * @tparam M  Specialization of [[PredictionModel]].  If you subclass this type, use this type
 *            parameter to specify the concrete type for the corresponding model.
 */
@DeveloperApi
abstract class PredictionModel[FeaturesType, M <: PredictionModel[FeaturesType, M]]
  extends Model[M] with PredictorParams {

  /** @group setParam */
  def setFeaturesCol(value: String): M = set(featuresCol, value).asInstanceOf[M]

  /** @group setParam */
  def setPredictionCol(value: String): M = set(predictionCol, value).asInstanceOf[M]

  /** Returns the number of features the model was trained on. If unknown, returns -1 */
  @Since("1.6.0")
  def numFeatures: Int = -1

  /**
   * Returns the SQL DataType corresponding to the FeaturesType type parameter.
   *
   * This is used by [[validateAndTransformSchema()]].
   * This workaround is needed since SQL has different APIs for Scala and Java.
   *
   * The default value is VectorUDT, but it may be overridden if FeaturesType is not Vector.
   */
  protected def featuresDataType: DataType = new VectorUDT

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema, fitting = false, featuresDataType)
  }

  /**
   * Transforms dataset by reading from [[featuresCol]], calling [[predict()]], and storing
   * the predictions as a new column [[predictionCol]].
   *
   * @param dataset input dataset
   * @return transformed dataset with [[predictionCol]] of type [[Double]]
   */
  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    if ($(predictionCol).nonEmpty) {
      transformImpl(dataset)
    } else {
      this.logWarning(s"$uid: Predictor.transform() was called as NOOP" +
        " since no output columns were set.")
      dataset
    }
  }

  protected def transformImpl(dataset: DataFrame): DataFrame = {
    val predictUDF = udf { (features: Any) =>
      predict(features.asInstanceOf[FeaturesType])
    }
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  /**
   * Predict label for the given features.
   * This internal method is used to implement [[transform()]] and output [[predictionCol]].
   */
  protected def predict(features: FeaturesType): Double
}
