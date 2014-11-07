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

package org.apache.spark.mllib.tree.impl

import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{DenseVector, Vectors, Vector}
import org.apache.spark.rdd.RDD


private[tree] object Util {

  /**
   * Convert a dataset of [[DenseVector]] from row storage to column storage.
   *
   * WARNING: This shuffles the ENTIRE dataset across the network, so it is a VERY EXPENSIVE
   *          operation.  This can also fail if 1 column is too large to fit on 1 partition.
   *
   * This maintains sparsity in the data.
   *
   * This maintains matrix structure.  I.e., each partition of the output RDD holds adjacent
   * columns.  The number of partitions will be min(input RDD's number of partitions, numColumns).
   *
   * @param rowStore  The input vectors are data rows/instances.
   * @return RDD of (columnIndex, columnValues) pairs,
   *         where each pair corresponds to one entire column.
   *         If either dimension of the given data is 0, this returns an empty RDD.
   *         If vector lengths do not match, this throws an exception.
   *
   * TODO: Add implementation for sparse data.
   *       For sparse data, distribute more evenly based on number of non-zeros.
   *       (First collect stats to decide how to partition.)
   * TODO: Move elsewhere in MLlib.
   */
  def rowToColumnStoreDense(rowStore: RDD[Vector]): RDD[(Int, Vector)] = {
    import scala.collection.mutable.ArrayBuffer

    val numRows = {
      val longNumRows: Long = rowStore.count()
      require(longNumRows < Int.MaxValue, s"rowToColumnStore given RDD with $longNumRows rows," +
        s" but can handle at most ${Int.MaxValue} rows")
      longNumRows.toInt
    }
    if (numRows == 0) {
      return rowStore.sparkContext.parallelize(Seq.empty[(Int, DenseVector)])
    }
    val numCols = rowStore.take(1)(0).size
    val numSourcePartitions = rowStore.partitions.size
    val numTargetPartitions = Math.min(numCols, rowStore.partitions.size)
    if (numTargetPartitions == 0) {
      return rowStore.sparkContext.parallelize(Seq.empty[(Int, DenseVector)])
    }
    val maxColumnsPerPartition = Math.floor(numCols / (numTargetPartitions + 0.0)).toInt

    def getNumColsInGroup(groupIndex: Int) = {
      if (groupIndex + 1 < numTargetPartitions) {
        maxColumnsPerPartition
      } else {
        numCols - (numTargetPartitions - 1) * maxColumnsPerPartition // last partition
      }
    }

    /* On each partition, re-organize into groups of columns:
         (groupIndex, (sourcePartitionIndex, partCols)),
         where partCols(colIdx) = partial column.
       The groupIndex will be used to groupByKey.
       The sourcePartitionIndex is used to ensure instance indices match up after the shuffle.
       The partial columns will be stacked into full columns after the shuffle.
       Note: By design, partCols will always have at least 1 column.
     */
    val partialColumns: RDD[(Int, (Int, Array[Array[Double]]))] =
      rowStore.mapPartitionsWithIndex { case (sourcePartitionIndex, iterator) =>
        // columnSets(groupIndex)(colIdx)
        //   = column values for each instance in sourcePartitionIndex,
        // where colIdx is a 0-based index for columns for groupIndex
        val columnSets = new Array[Array[ArrayBuffer[Double]]](numTargetPartitions)
        Range(0, numTargetPartitions).foreach { groupIndex =>
          columnSets(groupIndex) =
            Array.fill[ArrayBuffer[Double]](getNumColsInGroup(groupIndex))(ArrayBuffer[Double]())
        }
        iterator.foreach { row =>
          Range(0, numTargetPartitions).foreach { groupIndex =>
            val fromCol = groupIndex * maxColumnsPerPartition
            val numColsInTargetPartition = getNumColsInGroup(groupIndex)
            var colIdx = 0
            while (colIdx < numColsInTargetPartition) {
              columnSets(groupIndex)(colIdx) += row(fromCol + colIdx)
              colIdx += 1
            }
          }
        }
        Range(0, numTargetPartitions).map { groupIndex =>
          (groupIndex,
            (sourcePartitionIndex, columnSets(groupIndex).map(_.toArray)))
        }.toIterator
      }

    // Shuffle data
    val groupedPartialColumns: RDD[(Int, Iterable[(Int, Array[Array[Double]])])] =
      partialColumns.groupByKey()

    // Each target partition now holds its set of columns.
    // Group the partial columns into full columns.
    val fullColumns = groupedPartialColumns.flatMap { case (groupIndex, iterator) =>
      // We do not know the number of rows per group, so we need to collect the groups
      // before filling the full columns.
      val collectedPartCols = new Array[Array[Array[Double]]](numSourcePartitions)
      iterator.foreach { case (sourcePartitionIndex, partCols) =>
        collectedPartCols(sourcePartitionIndex) = partCols
      }
      val rowOffsets: Array[Int] = collectedPartCols.map(_(0).size).scanLeft(0)(_ + _)
      val numRows = rowOffsets.last
      // Initialize full columns
      val fromCol = groupIndex * maxColumnsPerPartition
      val numColumnsInPartition = getNumColsInGroup(groupIndex)
      val partitionColumns: Array[Array[Double]] =
        Array.fill[Array[Double]](numColumnsInPartition)(new Array[Double](numRows))
      var colIdx = 0 // index within group
      while (colIdx < numColumnsInPartition) {
        var sourcePartitionIndex = 0
        while (sourcePartitionIndex < numSourcePartitions) {
          val partColLength =
            rowOffsets(sourcePartitionIndex + 1) - rowOffsets(sourcePartitionIndex)
          Array.copy(collectedPartCols(sourcePartitionIndex)(colIdx), 0,
            partitionColumns(colIdx), rowOffsets(sourcePartitionIndex), partColLength)
          sourcePartitionIndex += 1
        }
        colIdx += 1
      }
      val columnIndices = Range(0, numColumnsInPartition).map(_ + fromCol)
      val columns = partitionColumns.map(Vectors.dense)
      columnIndices.zip(columns)
    }

    fullColumns
  }
}
