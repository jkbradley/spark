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

package org.apache.spark.ml.tree.impl

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.tree.ContinuousSplit
import org.apache.spark.ml.tree.impl.AltDT.{FeatureVector, PartitionInfo}
import org.apache.spark.ml.tree.impl.TreeUtil._
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.mllib.tree.configuration.FeatureType
import org.apache.spark.mllib.tree.model.Predict
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.util.collection.BitSet

/**
 * Test suite for [[AltDT]].
 */
class AltDTSuite extends SparkFunSuite with MLlibTestSparkContext  {

  test("FeatureVector") {
    val v = new FeatureVector(1, FeatureType.Continuous, Array(0.1, 0.3, 0.7), Array(1, 2, 0))

    val vCopy = v.deepCopy()
    vCopy.values(0) = 1000
    assert(v.values(0) !== vCopy.values(0))

    val original = Vectors.dense(0.7, 0.1, 0.3)
    val v2 = FeatureVector.fromOriginal(1, FeatureType.Continuous, original)
    assert(v === v2)
  }

  test("PartitionInfo") {
    val numRows = 4
    val col1 =
      FeatureVector.fromOriginal(0, FeatureType.Continuous, Vectors.dense(0.8, 0.2, 0.1, 0.6))
    val col2 =
      FeatureVector.fromOriginal(1, FeatureType.Categorical, Vectors.dense(0, 1, 0, 2))
    assert(col1.values.length === numRows)
    assert(col2.values.length === numRows)
    val nodeOffsets = Array(0, numRows)
    val activeNodes = new BitSet(1)
    activeNodes.set(0)

    val info = PartitionInfo(Array(col1, col2), nodeOffsets, activeNodes)

    // Create bitVector for splitting the 4 rows: L, R, L, R
    // New groups are {0, 2}, {1, 3}
    val bitVector = new BitSubvector(0, numRows)
    bitVector.set(1)
    bitVector.set(3)

    val newInfo = info.update(Array(bitVector), newNumNodeOffsets = 3)

    assert(newInfo.columns.length === 2)
    val expectedCol1a =
      new FeatureVector(0, FeatureType.Continuous, Array(0.1, 0.8, 0.2, 0.6), Array(2, 0, 1, 3))
    val expectedCol1b =
      new FeatureVector(1, FeatureType.Categorical, Array(0, 0, 1, 2), Array(0, 2, 1, 3))
    assert(newInfo.columns(0) === expectedCol1a)
    assert(newInfo.columns(1) === expectedCol1b)
    assert(newInfo.nodeOffsets === Array(0, 2, 4))
    assert(newInfo.activeNodes.iterator.toSet === Set(0, 1))

    // Create 2 bitVectors for splitting into: 0, 2, 1, 3
    val bv2a = new BitSubvector(0, 2)
    bv2a.set(1)
    val bv2b = new BitSubvector(2, 4)
    bv2b.set(3)

    val newInfo2 = newInfo.update(Array(bv2a, bv2b), newNumNodeOffsets = 5)

    assert(newInfo2.columns.length === 2)
    val expectedCol2a =
      new FeatureVector(0, FeatureType.Continuous, Array(0.8, 0.1, 0.2, 0.6), Array(0, 2, 1, 3))
    val expectedCol2b =
      new FeatureVector(1, FeatureType.Categorical, Array(0, 0, 1, 2), Array(0, 2, 1, 3))
    assert(newInfo2.columns(0) === expectedCol2a)
    assert(newInfo2.columns(1) === expectedCol2b)
    assert(newInfo2.nodeOffsets === Array(0, 1, 2, 3, 4))
    assert(newInfo2.activeNodes.iterator.toSet === Set(0, 1, 2, 3))
  }

  test("computeBestSplits") {

  }

  test("computeActiveNodePeriphery") {
  }

  test("collectBitVectors") {
    val col =
      FeatureVector.fromOriginal(0, FeatureType.Continuous, Vectors.dense(0.1, 0.2, 0.4, 0.6, 0.7))
    val numRows = col.values.length
    val activeNodes = new BitSet(1)
    activeNodes.set(0)
    val info = PartitionInfo(Array(col), Array(0, numRows), activeNodes)
    val partitionInfos = sc.parallelize(Seq(info))
    val bestSplitAndGain = (new ContinuousSplit(0, threshold = 0.5),
      new InfoGainStats(0, 0, 0, 0, 0, new Predict(0, 0), new Predict(0, 0)))
    val bitVectors = AltDT.collectBitVectors(partitionInfos, Array(bestSplitAndGain))
    assert(bitVectors.length === 1)
    val bitv = bitVectors.head
    assert(bitv.numBits === numRows)
    assert(bitv.iterator.toArray === Array(3, 4))
  }

}
