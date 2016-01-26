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

import java.{util => ju}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

import org.apache.hadoop.fs.Path
import org.json4s._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.{Logging, SparkContext}
import org.apache.spark.annotation.{DeveloperApi, Experimental, Since}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{AbstractDataType, StructType}

/**
 * :: DeveloperApi ::
 * A stage in a pipeline, either an [[Estimator]] or a [[Transformer]].
 */
@DeveloperApi
abstract class PipelineStage extends Params with Logging {

  private val inputColDataTypes: mutable.Map[Param[_], Seq[AbstractDataType]] =
    mutable.Map.empty[Param[_], Seq[AbstractDataType]]

  /** list of (inputCol param, valid data types). Note some are optional (weightCol), and some depend on fit/transform */
  final def getInputCols: Map[String, Seq[AbstractDataType]] = {
    inputColDataTypes.flatMap {
      case (colParam: InputColParam, types: Seq[AbstractDataType]) =>
        Seq($(colParam) -> types)
      case (colsParam: InputColsParam, types: Seq[AbstractDataType]) =>
        $(colsParam).map(_ -> types)
    }.toMap
  }

  protected final def setInputColDataType(
      inputCol: InputColParam,
      dataTypes: Seq[AbstractDataType]): Unit = {
    inputColDataTypes(inputCol) = dataTypes
  }

  protected final def setInputColDataType(
      inputCols: InputColsParam,
      dataTypes: Seq[AbstractDataType]): Unit = {
    inputColDataTypes(inputCols) = dataTypes
  }

  // If set to false, then transform only
  private val inputColFitOnly: mutable.Map[Param[_], Boolean] =
    mutable.Map.empty[Param[_], Boolean]

  final def useForFit(param: Param[_]): Boolean = inputColFitOnly.get(param) match {
    case Some(fitOnly) => fitOnly
    case None => true
  }

  final def useForTransform(param: Param[_]): Boolean = inputColFitOnly.get(param) match {
    case Some(fitOnly) => !fitOnly
    case None => true
  }

  final def useNow(param: Param[_], fitting: Boolean): Boolean = {
    if (fitting) useForFit(param) else useForTransform(param)
  }

  protected final def setUseForFitOnly(inputCol: InputColParam): Unit = {
    inputColFitOnly(inputCol) = true
  }

  protected final def setUseForTransformOnly(inputCol: InputColParam): Unit = {
    inputColFitOnly(inputCol) = false
  }

  /**
   * :: DeveloperApi ::
   *
   * Derives the output schema from the input schema.
    * This also calls [[validateParams()]].
   */
  @DeveloperApi
  final def transformSchema(schema: StructType, fitting: Boolean): StructType =
    transformSchema(schema, fitting, logging = true)

  /**
   * :: DeveloperApi ::
   *
   * Derives the output schema from the input schema and parameters, optionally with logging.
   *
   * This should be optimistic.  If it is unclear whether the schema will be valid, then it should
   * be assumed valid until proven otherwise.
   */
  @DeveloperApi
  final def transformSchema(
      schema: StructType,
      fitting: Boolean,
      logging: Boolean): StructType = {
    if (logging) {
      logDebug(s"Input schema: ${schema.json}")
    }
    validateParams()
    checkInputColDataTypes(schema, fitting)
    val outputSchema = transformSchemaImpl(schema, fitting)
    if (logging) {
      logDebug(s"Expected output schema: ${outputSchema.json}")
    }
    outputSchema
  }

  /**
   * Called by [[transformSchema()]]
   * @param inputSchema
   */
  protected def checkInputColDataTypes(inputSchema: StructType, fitting: Boolean): Unit = {
    params.filter(p => useNow(p, fitting)).foreach {
      // If input column appears in inputColDataTypes, check the type.  Otherwise, ignore it.
      case colParam: InputColParam =>
        inputColDataTypes.get(colParam).foreach { expectedTypes =>
          SchemaUtils.checkColumnTypes(inputSchema, $(colParam), expectedTypes)
        }
      case colsParam: InputColsParam =>
        inputColDataTypes.get(colsParam).foreach { expectedTypes =>
          $(colsParam).foreach { col =>
            SchemaUtils.checkColumnTypes(inputSchema, col, expectedTypes)
          }
        }
      case _ => // do nothing
    }
  }

  /**
   * In implementing classes, this method should check the following items, as needed:
   *  - input column metadata
   *  - interacting input columns' schema
   * This method does not need to check individual input columns' schema;
   * that is handled by [[inputColDataTypes]].
   * This should also compute the output schema.
   */
  protected def transformSchemaImpl(schema: StructType, fitting: Boolean): StructType

  override def copy(extra: ParamMap): PipelineStage
}

/**
 * :: Experimental ::
 * A simple pipeline, which acts as an estimator. A Pipeline consists of a sequence of stages, each
 * of which is either an [[Estimator]] or a [[Transformer]]. When [[Pipeline#fit]] is called, the
 * stages are executed in order. If a stage is an [[Estimator]], its [[Estimator#fit]] method will
 * be called on the input dataset to fit a model. Then the model, which is a transformer, will be
 * used to transform the dataset as the input to the next stage. If a stage is a [[Transformer]],
 * its [[Transformer#transform]] method will be called to produce the dataset for the next stage.
 * The fitted model from a [[Pipeline]] is an [[PipelineModel]], which consists of fitted models and
 * transformers, corresponding to the pipeline stages. If there are no stages, the pipeline acts as
 * an identity transformer.
 */
@Experimental
class Pipeline(override val uid: String) extends Estimator[PipelineModel] with MLWritable {

  def this() = this(Identifiable.randomUID("pipeline"))

  /**
   * param for pipeline stages
   * @group param
   */
  val stages: Param[Array[PipelineStage]] = new Param(this, "stages",
    "stages of the pipeline, which should have unique UIDs",
    isValid = (s: Array[PipelineStage]) => s.toSet.size == s.length)

  /** @group setParam */
  def setStages(value: Array[PipelineStage]): this.type = { set(stages, value); this }

  // Below, we clone stages so that modifications to the list of stages will not change
  // the Param value in the Pipeline.
  /** @group getParam */
  def getStages: Array[PipelineStage] = $(stages).clone()

  override def validateParams(): Unit = {
    super.validateParams()
    $(stages).foreach(_.validateParams())
  }

  /**
   * Fits the pipeline to the input dataset with additional parameters. If a stage is an
   * [[Estimator]], its [[Estimator#fit]] method will be called on the input dataset to fit a model.
   * Then the model, which is a transformer, will be used to transform the dataset as the input to
   * the next stage. If a stage is a [[Transformer]], its [[Transformer#transform]] method will be
   * called to produce the dataset for the next stage. The fitted model from a [[Pipeline]] is an
   * [[PipelineModel]], which consists of fitted models and transformers, corresponding to the
   * pipeline stages. If there are no stages, the output model acts as an identity transformer.
   *
   * @param dataset input dataset
   * @return fitted pipeline
   */
  override protected def fitImpl(dataset: DataFrame): PipelineModel = {
    val theStages = $(stages)
    // Search for the last estimator.
    var indexOfLastEstimator = -1
    theStages.view.zipWithIndex.foreach { case (stage, index) =>
      stage match {
        case _: Estimator[_] =>
          indexOfLastEstimator = index
        case _ =>
      }
    }
    var curDataset = dataset
    val transformers = ListBuffer.empty[Transformer]
    theStages.view.zipWithIndex.foreach { case (stage, index) =>
      if (index <= indexOfLastEstimator) {
        val transformer = stage match {
          case estimator: Estimator[_] =>
            estimator.fit(curDataset)
          case t: Transformer =>
            t
          case _ =>
            throw new IllegalArgumentException(
              s"Do not support stage $stage of type ${stage.getClass}")
        }
        if (index < indexOfLastEstimator) {
          curDataset = transformer.transform(curDataset)
        }
        transformers += transformer
      } else {
        transformers += stage.asInstanceOf[Transformer]
      }
    }

    new PipelineModel(uid, transformers.toArray)
  }

  override def copy(extra: ParamMap): Pipeline = {
    val map = extractParamMap(extra)
    val newStages = map(stages).map(_.copy(extra))
    new Pipeline(uid).setStages(newStages)
  }

  override protected def transformSchemaImpl(schema: StructType): StructType = {
    $(stages).foldLeft(schema) { (cur, stage) =>
      stage.transformSchema(cur, fitting = true, logging = false)
    }
  }

  @Since("1.6.0")
  override def write: MLWriter = new Pipeline.PipelineWriter(this)
}

@Since("1.6.0")
object Pipeline extends MLReadable[Pipeline] {

  @Since("1.6.0")
  override def read: MLReader[Pipeline] = new PipelineReader

  @Since("1.6.0")
  override def load(path: String): Pipeline = super.load(path)

  private[Pipeline] class PipelineWriter(instance: Pipeline) extends MLWriter {

    SharedReadWrite.validateStages(instance.getStages)

    override protected def saveImpl(path: String): Unit =
      SharedReadWrite.saveImpl(instance, instance.getStages, sc, path)
  }

  private class PipelineReader extends MLReader[Pipeline] {

    /** Checked against metadata when loading model */
    private val className = classOf[Pipeline].getName

    override def load(path: String): Pipeline = {
      val (uid: String, stages: Array[PipelineStage]) = SharedReadWrite.load(className, sc, path)
      new Pipeline(uid).setStages(stages)
    }
  }

  /** Methods for [[MLReader]] and [[MLWriter]] shared between [[Pipeline]] and [[PipelineModel]] */
  private[ml] object SharedReadWrite {

    import org.json4s.JsonDSL._

    /** Check that all stages are Writable */
    def validateStages(stages: Array[PipelineStage]): Unit = {
      stages.foreach {
        case stage: MLWritable => // good
        case other =>
          throw new UnsupportedOperationException("Pipeline write will fail on this Pipeline" +
            s" because it contains a stage which does not implement Writable. Non-Writable stage:" +
            s" ${other.uid} of type ${other.getClass}")
      }
    }

    /**
     * Save metadata and stages for a [[Pipeline]] or [[PipelineModel]]
     *  - save metadata to path/metadata
     *  - save stages to stages/IDX_UID
     */
    def saveImpl(
        instance: Params,
        stages: Array[PipelineStage],
        sc: SparkContext,
        path: String): Unit = {
      val stageUids = stages.map(_.uid)
      val jsonParams = List("stageUids" -> parse(compact(render(stageUids.toSeq))))
      DefaultParamsWriter.saveMetadata(instance, path, sc, paramMap = Some(jsonParams))

      // Save stages
      val stagesDir = new Path(path, "stages").toString
      stages.zipWithIndex.foreach { case (stage: MLWritable, idx: Int) =>
        stage.write.save(getStagePath(stage.uid, idx, stages.length, stagesDir))
      }
    }

    /**
     * Load metadata and stages for a [[Pipeline]] or [[PipelineModel]]
     * @return  (UID, list of stages)
     */
    def load(
        expectedClassName: String,
        sc: SparkContext,
        path: String): (String, Array[PipelineStage]) = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName)

      implicit val format = DefaultFormats
      val stagesDir = new Path(path, "stages").toString
      val stageUids: Array[String] = (metadata.params \ "stageUids").extract[Seq[String]].toArray
      val stages: Array[PipelineStage] = stageUids.zipWithIndex.map { case (stageUid, idx) =>
        val stagePath = SharedReadWrite.getStagePath(stageUid, idx, stageUids.length, stagesDir)
        DefaultParamsReader.loadParamsInstance[PipelineStage](stagePath, sc)
      }
      (metadata.uid, stages)
    }

    /** Get path for saving the given stage. */
    def getStagePath(stageUid: String, stageIdx: Int, numStages: Int, stagesDir: String): String = {
      val stageIdxDigits = numStages.toString.length
      val idxFormat = s"%0${stageIdxDigits}d"
      val stageDir = idxFormat.format(stageIdx) + "_" + stageUid
      new Path(stagesDir, stageDir).toString
    }
  }
}

/**
 * :: Experimental ::
 * Represents a fitted pipeline.
 */
@Experimental
class PipelineModel private[ml] (
    override val uid: String,
    val stages: Array[Transformer])
  extends Model[PipelineModel] with MLWritable with Logging {

  /** A Java/Python-friendly auxiliary constructor. */
  private[ml] def this(uid: String, stages: ju.List[Transformer]) = {
    this(uid, stages.asScala.toArray)
  }

  override def validateParams(): Unit = {
    super.validateParams()
    stages.foreach(_.validateParams())
  }

  override protected def transformImpl(dataset: DataFrame): DataFrame = {
    stages.foldLeft(dataset)((cur, transformer) => transformer.transformWithoutCheck(cur))
  }

  override protected def transformSchemaImpl(schema: StructType): StructType = {
    stages.foldLeft(schema) { (cur, transformer) =>
      transformer.transformSchema(cur, fitting = false, logging = false)
    }
  }

  override def copy(extra: ParamMap): PipelineModel = {
    new PipelineModel(uid, stages.map(_.copy(extra))).setParent(parent)
  }

  @Since("1.6.0")
  override def write: MLWriter = new PipelineModel.PipelineModelWriter(this)
}

@Since("1.6.0")
object PipelineModel extends MLReadable[PipelineModel] {

  import Pipeline.SharedReadWrite

  @Since("1.6.0")
  override def read: MLReader[PipelineModel] = new PipelineModelReader

  @Since("1.6.0")
  override def load(path: String): PipelineModel = super.load(path)

  private[PipelineModel] class PipelineModelWriter(instance: PipelineModel) extends MLWriter {

    SharedReadWrite.validateStages(instance.stages.asInstanceOf[Array[PipelineStage]])

    override protected def saveImpl(path: String): Unit = SharedReadWrite.saveImpl(instance,
      instance.stages.asInstanceOf[Array[PipelineStage]], sc, path)
  }

  private class PipelineModelReader extends MLReader[PipelineModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[PipelineModel].getName

    override def load(path: String): PipelineModel = {
      val (uid: String, stages: Array[PipelineStage]) = SharedReadWrite.load(className, sc, path)
      val transformers = stages map {
        case stage: Transformer => stage
        case other => throw new RuntimeException(s"PipelineModel.read loaded a stage but found it" +
          s" was not a Transformer.  Bad stage ${other.uid} of type ${other.getClass}")
      }
      new PipelineModel(uid, transformers)
    }
  }
}
