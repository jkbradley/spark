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

package org.apache.spark.examples.mllib

import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

/*
This test is to figure out the best way to implement DecisionTree nodes,
which are partly internal nodes and partly leaf nodes.  During prediction,
we need to call a recursive prediction method on nodes which behaves
differently for the 2 types of nodes.

Med nothing: 2.311785
Avg nothing: 2.31083335

Med virtual: 2.168312
Avg virtual: 2.1700435000000002

Med match-case: 2.148092
Avg match-case: 2.1531757
*/

// Nothing
class MyNode3(val prediction: Double, val isLeafy: Boolean) {
  def isLeafA(x: Double): Boolean = x == 0.5
  def isLeafB(x: Double): Boolean = x == -0.5
}

/*
// Using virtual functions
abstract class MyNode2(val prediction: Double) {
  def isLeaf(x: Double): Boolean
}

class MyInternalNode2(prediction: Double) extends MyNode2(prediction) {
  override def isLeaf(x: Double): Boolean = x == 0.5
}

class MyLeafNode2(prediction: Double) extends MyNode2(prediction) {
  override def isLeaf(x: Double): Boolean = x == -0.5
}
*/

/*
// Using case-match
abstract class MyNode1(val prediction: Double)

class MyInternalNode1(prediction: Double) extends MyNode1(prediction) {
  def isLeaf(x: Double): Boolean = x == 0.5
}

class MyLeafNode1(prediction: Double) extends MyNode1(prediction) {
  def isLeaf(x: Double): Boolean = x == -0.5
}
*/


object VirtualMatchTest {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName(s"My test")
    val sc = new SparkContext(conf)

    val numIterations = args(0).toLong

    val numOuterIterations = 20
    val nothingVals = Array.fill[Double](numOuterIterations)(0.0)
    val virtualVals = Array.fill[Double](numOuterIterations)(0.0)
    val matchVals = Array.fill[Double](numOuterIterations)(0.0)

    var iter = 0
    while (iter < numOuterIterations) {

      // nothing
      {
        val myNode3 = new MyNode3(iter, true)
        val myNode3a = new MyNode3(iter, false)
        val start = System.nanoTime()
        var i = 0L
        while (i < numIterations) {
          if (myNode3.isLeafy) {
            if (myNode3.isLeafA(i)) println("blah")
          } else {
            if (myNode3.isLeafB(i)) println("blah")
          }
          i += 1
        }
        i = 0L
        while (i < numIterations) {
          if (myNode3a.isLeafy) {
            if (myNode3a.isLeafA(i)) println("blah")
          } else {
            if (myNode3a.isLeafB(i)) println("blah")
          }
          i += 1
        }
        val elapsed = (System.nanoTime() - start) / 1e9
        nothingVals(iter) += elapsed
      }

      /*
      // virtual
      {
        val myNode2: MyNode2 = new MyLeafNode2(iter)
        val myNode2a: MyNode2 = new MyInternalNode2(iter)
        val start = System.nanoTime()
        var i = 0L
        while (i < numIterations) {
          if (myNode2.isLeaf(i)) println("blah")
          i += 1
        }
        i = 0L
        while (i < numIterations) {
          if (myNode2a.isLeaf(i)) println("blah x")
          i += 1
        }
        val elapsed = (System.nanoTime() - start) / 1e9
        virtualVals(iter) += elapsed
      }
      */

      /*
      // match-case
      {
        val myNode1: MyNode1 = new MyLeafNode1(iter)
        val myNode1a: MyNode1 = new MyInternalNode1(iter)
        val start = System.nanoTime()
        var i = 0L
        while (i < numIterations) {
          myNode1 match {
            case n: MyInternalNode1 =>
              if (n.isLeaf(i)) println("blah")
            case n: MyLeafNode1 =>
              if (n.isLeaf(i)) println("blah x")
          }
          i += 1
        }
        i = 0L
        while (i < numIterations) {
          myNode1a match {
            case n: MyInternalNode1 =>
              if (n.isLeaf(i)) println("blah")
            case n: MyLeafNode1 =>
              if (n.isLeaf(i)) println("blah x")
          }
          i += 1
        }
        val elapsed = (System.nanoTime() - start) / 1e9
        matchVals(iter) += elapsed
      }
      */

      iter += 1
    }

    println(s"Med nothing: ${nothingVals.sorted.apply(numOuterIterations / 2)}")
    println(s"Med virtual: ${virtualVals.sorted.apply(numOuterIterations / 2)}")
    println(s"Med match-case: ${matchVals.sorted.apply(numOuterIterations / 2)}")

    println(s"Avg nothing: ${nothingVals.sum / numOuterIterations}")
    println(s"Avg virtual: ${virtualVals.sum / numOuterIterations}")
    println(s"Avg match-case: ${matchVals.sum / numOuterIterations}")

    sc.stop()
  }
}
