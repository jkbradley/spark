package org.apache.spark.sql

import java.nio.file.Files
import java.util.concurrent.atomic.AtomicBoolean

import scala.util.Random

import org.scalatest.concurrent.Eventually

import org.apache.spark.sql.execution.streaming.MemoryStream
import org.apache.spark.sql.functions._
import org.apache.spark.sql.test.SharedSQLContext
import org.apache.spark.sql.util.ContinuousQueryListener
import org.apache.spark.sql.util.ContinuousQueryListener.{QueryTerminated, QueryProgress, QueryStarted}

class StreamingKMeans extends StreamTest with SharedSQLContext with Eventually {
  import testImplicits._

  test("streaming agg") {
    val source = MemoryStream[Int]
    val inputDf = source.toDS().toDF("point")
    val model = new StreamingKMeansModel(2)
    val rng = new Random()
    val numIter = 5
    val trigger = MLAlgorithmTrigger(numIter)

    val chkpoint = Files.createTempDirectory(null).toFile.toString
    sqlContext.streams.addListener(new StreamingMLModelListener(model, trigger))
    val cq = inputDf.write
      .option("checkpointLocation", chkpoint)
      .format("memory")
      .queryName("streaming_kmeans")
      .startStream()
    val numBatches = 10
    val totalIters = numIter * numBatches
    trigger.resetLatch()

    (1 to 10).foreach { _ =>
      val numRecords = rng.nextInt(5) + 1
      val records = Seq.tabulate(numRecords) { _ =>
        if (rng.nextDouble() >= 0.7) {
          7 + (rng.nextInt(7) - 3)
        } else {
          1 + (rng.nextInt(7) - 3)
        }
      }
      println("records for batch: ", records)
      source.addData(records)
      trigger.await()
    }
    
    println("final centers")
    model.printCenters()
    eventually(trigger.totalIters == totalIters)
    cq.stop()
  }
}

class StreamingKMeansModel(k: Int) extends Serializable {

  private val centers: Array[Double] = Array.tabulate(k)(_ => scala.util.Random.nextInt(11))
  println("initial centers")
  printCenters()

  def assignCluster(value: Int): Int = this.synchronized {
    val distances = Seq.tabulate(k)(center => (center, math.abs(value - centers(center))))
    val cluster = distances.sortBy(_._2).head._1
    println("assigned", cluster, value)
    cluster
  }

  def assignCenters(newCenters: Seq[(Int, Double)]): Unit = this.synchronized {
    newCenters.foreach { case (center, coordinate) =>
      centers(center) = coordinate
    }
    println("assigned new centers:")
    printCenters()
  }

  def printCenters(): Unit = {
    centers.zipWithIndex.foreach { case (point, index) =>
      println("cluster: " + index, point)
    }
  }
}

class StreamingMLModelListener(model: StreamingKMeansModel, trigger: MLAlgorithmTrigger)
  extends ContinuousQueryListener {
  private val isDone = new AtomicBoolean(false)

  override def onQueryStarted(queryStarted: QueryStarted): Unit = {}
  override def onQueryProgress(queryProgress: QueryProgress): Unit = {
    if (!isDone.get()) {
      try {
        val sqlContext = queryProgress.query.sqlContext
        import sqlContext.implicits._
        val localModel = model
        val assignClusters = udf((point: Int) => localModel.assignCluster(point))
        val latestData = sqlContext.table("streaming_kmeans")
        for (i <- 1 to trigger.numIter) {
          val assigned = latestData.select(assignClusters($"point").as("cluster"), $"point")
          val df = assigned.groupBy("cluster").agg(avg("point").as("newCenter"))
          model.assignCenters(df.collect().map(r => (r.getInt(0), r.getDouble(1))))
          trigger.acknowledge()
        }
        trigger.resetLatch()
      } catch {
        case _: IllegalStateException =>
          // Spark Context stopped. Do nothing. Not much we can do, because listener is async
      }
    }
  }
  override def onQueryTerminated(queryTerminated: QueryTerminated): Unit = {
    isDone.set(true)
  }
}

case class CenterAndPoint(assignedCenter: Int, point: Int)
