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

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.tree.impl.Util._
import org.apache.spark.mllib.util.LocalSparkContext
import org.scalatest.FunSuite

/**
 * Test suite for [[org.apache.spark.mllib.tree.impl.Util]].
 */
class TreeUtilSuite extends FunSuite with LocalSparkContext  {

  private def check(rows: Seq[Vector]): Unit = {
    val numRowPartitions = 2
    val rowStore = sc.parallelize(rows, numRowPartitions)
    val colStore = rowToColumnStoreDense(rowStore)
    val numColPartitions = colStore.partitions.size
    val cols: Map[Int, Vector] = colStore.collect().toMap
    val numRows = rows.size
    if (numRows == 0) {
      assert(cols.size == 0)
      return
    }
    val numCols = rows(0).size
    if (numCols == 0) {
      assert(cols.size == 0)
      return
    }
    rows.zipWithIndex.foreach { case (row, i) =>
      var j = 0
      while (j < numCols) {
        assert(row(j) == cols(j)(i))
        j += 1
      }
    }
    val expectedNumColPartitions = Math.min(rowStore.partitions.size, numCols)
    assert(numColPartitions === expectedNumColPartitions)
  }

  test("rowToColumnStoreDense: small") {
    val rows = Seq(
      Vectors.dense(1.0, 2.0, 3.0, 4.0),
      Vectors.dense(1.1, 2.1, 3.1, 4.1),
      Vectors.dense(1.2, 2.2, 3.2, 4.2)
    )
    check(rows)
  }

  test("rowToColumnStoreDense: large") {
    var numRows = 100
    var numCols = 90
    val rows = Range(0, numRows).map { i =>
      Vectors.dense(Range(0, numCols).map(_ + numCols * i + 0.0).toArray)
    }
    check(rows)
  }

  test("rowToColumnStoreDense: 0 rows") {
    val rows = Seq.empty[Vector]
    check(rows)
  }

  test("rowToColumnStoreDense: 0 cols") {
    val rows = Seq(
      Vectors.dense(Array.empty[Double]),
      Vectors.dense(Array.empty[Double]),
      Vectors.dense(Array.empty[Double])
    )
    check(rows)
  }
}
