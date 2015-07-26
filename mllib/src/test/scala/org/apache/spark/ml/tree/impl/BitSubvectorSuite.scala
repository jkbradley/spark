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
import org.apache.spark.mllib.util.MLlibTestSparkContext

/**
 * Test suite for [[BitSubvector]].
 */
class BitSubvectorSuite extends SparkFunSuite with MLlibTestSparkContext {

  test("BitSubvector basic ops") {
    val from = 1
    val to = 4
    val bs = new BitSubvector(from, to)
    assert(bs.numBits === to - from)
    Range(from, to).foreach(x => assert(!bs.get(x)))
    val setVals = Array(from, to - 1)
    setVals.foreach { x =>
      bs.set(x)
      assert(bs.get(x))
    }
    assert(bs.iterator.toSet === setVals.toSet)
  }

  test("BitSubvector merge") {
    val b1 = new BitSubvector(0, 5)
    b1.set(1)
    val b2 = new BitSubvector(5, 7)
    b2.set(5)
    val b3 = new BitSubvector(9, 12)
    b3.set(11)
    val parts1 = Array(b1)
    val parts2 = Array(b2, b3)
    val newParts = BitSubvector.merge(parts1, parts2)

    val r1 = new BitSubvector(0, 7)
    r1.set(1)
    r1.set(5)
    val r2 = new BitSubvector(9, 12)
    r2.set(11)
    val expectedParts = Array(r1, r2)
    newParts.zip(expectedParts).foreach { case (x, y) =>
      assert(x.from === y.from)
      assert(x.to === x.to)
      assert(x.iterator.toSet === y.iterator.toSet)
    }
  }

  test("BitSubvector merge with empty BitSubvectors") {
    val parts = BitSubvector.merge(Array.empty[BitSubvector], Array.empty[BitSubvector])
  }
}
