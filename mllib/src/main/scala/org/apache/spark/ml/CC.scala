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

import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

// scalastyle:off

object CC {

  val DST = "dst"
  val ID = "id"
  val SRC = "src"
  val LONG_ID = "long_id"
  val LONG_SRC = "long_src"
  val LONG_DST = "long_dst"
  val ORIG_ID = "ORIG_ID"
  val COMPONENT_ID = "component"

  def cacheDataFrame(df: DataFrame): DataFrame = {
    val tmpRdd = df.rdd.cache()
    df.sqlContext.createDataFrame(tmpRdd, df.schema)
  }

  def checkpointDataFrame(df: DataFrame): DataFrame = {
    val tmpRdd = df.rdd
    // Cache first so local checkpoint uses memory + disk.
    tmpRdd.cache()
    tmpRdd.localCheckpoint()
    df.sqlContext.createDataFrame(tmpRdd, df.schema)
  }

  def zipWithUniqueIdFrom0(df: DataFrame, dataType: DataType): DataFrame = {
    val dataTypeStr = dataType.simpleString
    val sqlContext = df.sqlContext
    val schema = df.schema
    val outputSchema = StructType(Seq(
      StructField("row", schema, false), StructField("uniq_id", dataType, false)))
    val rdd = if (dataTypeStr == "long") {
      df.rdd.zipWithIndex().map { case (row: Row, id: Long) => Row(row, id) }
    } else if (dataTypeStr == "int") {
      df.rdd.zipWithIndex().map { case (row: Row, id: Long) => Row(row, id.toInt) }
    } else {
      throw new IllegalArgumentException(s"Bad vertex index type: $dataTypeStr")
    }
    sqlContext.createDataFrame(rdd, outputSchema)
  }

  def getIndexedGraph(
                       vertices: DataFrame,
                       edges: DataFrame,
                       numVertices: Long): (DataFrame, DataFrame, DataType) = {
    // Special indexing in [0,...,numVertices).  This also drops the attribute columns.
    val hasIntegralIdType: Boolean = {
      vertices.schema(ID).dataType match {
        case _@(ByteType | IntegerType | ShortType) => true
        case _: LongType =>
          if (numVertices < Int.MaxValue) {
            // TODO: Do not assume there exists at least 1 vertex.
            val largestId =
              vertices.select(ID).groupBy().max(ID).rdd.map(_.getLong(0)).first()
            largestId < Int.MaxValue
          } else {
            false
          }
        case _ => false
      }
    }
    val dataType = if (numVertices < Int.MaxValue) DataTypes.IntegerType else DataTypes.LongType
    val dataTypeStr = dataType.simpleString

    val indexedV0: DataFrame = {
      if (hasIntegralIdType) {
        vertices.select(col(ID).cast(dataTypeStr).as(LONG_ID), col(ID).as(ID))
      } else {
        val indexedVertices = zipWithUniqueIdFrom0(vertices, dataType)
        indexedVertices.select(col("uniq_id").as(LONG_ID), col("row." + ID).as(ID))
      }
    }
    val indexedE0: DataFrame = {
      val packedEdges = edges.select(SRC, DST)
      val indexedSourceEdges = packedEdges
        .join(indexedV0.select(col(LONG_ID).as(LONG_SRC), col(ID).as(SRC)), SRC)
      val indexedEdges = indexedSourceEdges.select(SRC, LONG_SRC, DST)
        .join(indexedV0.select(col(LONG_ID).as(LONG_DST), col(ID).as(DST)), DST)
      indexedEdges.select(SRC, LONG_SRC, DST, LONG_DST)
    }

    val indexedVertices: DataFrame = indexedV0
      .select(col(ID).as(ORIG_ID), col(LONG_ID).as(ID), col(LONG_ID).as(COMPONENT_ID))
    val indexedEdges: DataFrame = indexedE0
      .select(col(LONG_SRC).as(SRC), col(LONG_DST).as(DST))
      .where(col(SRC) !== col(DST)) // remove self-edges
    (indexedVertices, indexedEdges, dataType)
  }

  def connectedComponentsNew(graphVertices: DataFrame, graphEdges: DataFrame): DataFrame = {
    // Add initial component column
    val NEW_COMPONENT_ID = "NEW_COMPONENT_ID"
    val maxIterations = 100

    def computeNewComponents(v: DataFrame, e: DataFrame): DataFrame = {
      // Send messages: smaller component ID -> replace larger component ID
      val msgsToSrc = e.join(v, e(DST) === v(ID), "inner")
        .select(e(SRC).as(ID), v(COMPONENT_ID))
      val msgsToDst = e.join(v, e(SRC) === v(ID), "inner")
        .select(e(DST).as(ID), v(COMPONENT_ID))
      val msgs = msgsToSrc.unionAll(msgsToDst)
      val newComponents = msgs.groupBy(ID)
        .agg(min(COMPONENT_ID).as(NEW_COMPONENT_ID))
      newComponents
    }

    val numOrigVertices = graphVertices.count()
    val (origVertices0: DataFrame, edges0: DataFrame, _: DataType) =
      getIndexedGraph(graphVertices, graphEdges, numOrigVertices)
    val origVertices: DataFrame = checkpointDataFrame(origVertices0)

    // Remove duplicate edges
    val edges1 = edges0.select(when(col(SRC) < col(DST), col(SRC)).otherwise(col(DST)).as(SRC),
      when(col(SRC) < col(DST), col(DST)).otherwise(col(SRC)).as(DST))
      .distinct()
    val origEdges: DataFrame = checkpointDataFrame(edges1)

    // Construct vertices2.
    // This also handles vertices without edges, which will be added back in at the end.
    val vInEdges2 = origEdges.select(explode(array(SRC, DST)).as(ID)).distinct()
    val vertices2: DataFrame = origVertices.join(vInEdges2, ID).select(ID, COMPONENT_ID)

    var v: DataFrame = vertices2
    var e: DataFrame = origEdges
    var iter = 0

    var lastCachedVertices: DataFrame = null
    var lastCachedEdges: DataFrame = null

    v = checkpointDataFrame(v)
    lastCachedVertices = v
    println(s"v schema:")
    v.printSchema()
    println(s"e schema:")
    e.printSchema()

    // Send messages: smaller component ID -> replace larger component ID
    // We copy this update of components before the loop to simplify the caching logic.
    var newComponents: DataFrame = computeNewComponents(v, e)
    var activeMessageCount: Long = Long.MaxValue

    while (iter < maxIterations && activeMessageCount > 0) {
      println(s"ITERATION $iter")

      // Update vertices with new components
      v = v.join(newComponents, v(ID) === newComponents(ID), "left_outer")
        .select(v(ID),
          when(col(NEW_COMPONENT_ID) < col(COMPONENT_ID), col(NEW_COMPONENT_ID))
            .otherwise(col(COMPONENT_ID)).as(COMPONENT_ID),
          (col(NEW_COMPONENT_ID) < col(COMPONENT_ID)).as("updated"))

      if (iter != 0 && iter % 4 == 0) {
        v = checkpointDataFrame(v)
      } else {
        v = cacheDataFrame(v)
      }

      if (iter != 0 && iter % 2 == 0) {
        // % x should have x >= 2
        // % x should have x >= 2
        // Update edges so each vertex connects to its component's master vertex.
        val newEdges = v.where(col(ID) !== col(COMPONENT_ID))
          .select(col(ID).as(SRC), col(COMPONENT_ID).as(DST))
        e = origEdges.unionAll(newEdges)
        e.cache()
        if (lastCachedEdges != null) lastCachedEdges.unpersist(blocking = false)
        lastCachedEdges = e
      }

      // Send messages: smaller component ID -> replace larger component ID
      newComponents = computeNewComponents(v, e)
      activeMessageCount = v.where(col("updated")).count()
      println(s"activeMessageCount: $activeMessageCount")

      if (lastCachedVertices != null) lastCachedVertices.unpersist(blocking = false)
      lastCachedVertices = v

      iter += 1
    }
    if (lastCachedEdges != null) lastCachedEdges.unpersist(blocking = false)
    // Unpersist here instead of before loop since we could have edges1 = edges2,
    // and the link between edges2 and origEdges is unclear (need to check local checkpoint impl).
    edges1.unpersist(blocking = false)
    origEdges.unpersist(blocking = false)

    // Handle vertices without edges.
    v = origVertices.join(v, origVertices(ID) === v(ID), "left_outer")
      .select(origVertices(ID).cast("long"), origVertices(ORIG_ID),
        coalesce(v(COMPONENT_ID), origVertices(COMPONENT_ID)).cast("long").as(COMPONENT_ID))
    // TODO: unpersist origVertices, bigComponents?
    // Join COMPONENT_ID column with original vertices.
    graphVertices.join(v.select(col(ORIG_ID).as(ID), col(COMPONENT_ID)), ID)
  }

  def shiftVertexIds(edges: DataFrame, i: Long): DataFrame = {
    require(Seq(DataTypes.LongType, DataTypes.IntegerType).contains(edges.schema("src").dataType))
    val newEdges = edges.select((col("src") + i).as("src"),
      (col("dst") + i).as("dst"))
    newEdges
  }

  def star(sqlContext: SQLContext, n: Int, numPartitions: Int): DataFrame = {
    val edges = sqlContext.range(start = 1, end = n + 1, step = 1, numPartitions = numPartitions)
      .toDF("src")
      .select(col("src"), lit(0L).as("dst"))
    edges
  }

  def chainOfStars(
      sqlContext: SQLContext, numStars: Int, starSize: Int, numPartitions: Int): DataFrame = {
    val edges0 = Range(0, numStars).map { i =>
      val edges = star(sqlContext, starSize, numPartitions)
      shiftVertexIds(edges, starSize * i)
    }.reduce((e0, e1) => e0.unionAll(e1))
    edges0.repartition(numPartitions)
  }

  def verticesFromEdges(e: DataFrame, numPartitions: Int): DataFrame = {
    val srcs = e.select(e("src").as("id"))
    val dsts = e.select(e("dst").as("id"))
    val v = srcs.unionAll(dsts).distinct()
    v.repartition(numPartitions)
  }

  def runTest(sqlContext: SQLContext): Unit = {
    // Create a star
    val numPartitions = 16
    val numStars = 10
    val starSize = 10000

    val edges = chainOfStars(sqlContext, numStars, starSize, numPartitions)
    val vertices = verticesFromEdges(edges, numPartitions)
    edges.cache().count()
    vertices.cache().count()

    val cc = connectedComponentsNew(vertices, edges)
    cc.count()
  }

}

// scalastyle:on
