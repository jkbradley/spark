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

package org.apache.spark.ml.util

import scala.reflect.ClassTag

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.param.{Param, ParamMap, Params, ParamsSuite}


/**
 * Trait for testing [[Param]] settings used in tests.
 *
 * The companion objects of test suites for [[Params]] types should implement this trait.
 */
abstract class DefaultParamsTestable[T <: Params with MLWritable : ClassTag]
  extends SparkFunSuite with TempDirectory {

  val clazz = classOf[T]
  val className = clazz.getCanonicalName

  /**
   * Get an instance of this [[Params]] type using the constructor taking a UID.
   *  - UID is set to "my_[CLASSNAME]"
   *  - [[Param]] values are defaults
   */
  def getDefaultInstance: T = {
    val uid = "my_" + className
    clazz.getConstructor(classOf[String]).newInstance(uid)
  }

  /**
   * Non-default values for ALL Params.
   *  - These should be different from the default values (for more robust testing).
   *  - Fitting an [[org.apache.spark.ml.Estimator]] with these Params should be very fast.
   *
   * @see [[validateAllParamSettings()]]
   */
  def allParamSettings: Map[String, Any]

  /**
   * Set all [[Params]] in the given instance according to [[allParamSettings]].
   * @param p  Instance which is modified to match [[allParamSettings]]
   */
  private def setAllParams(p: T): Unit = {
    allParamSettings.foreach { case (name: String, value: Any) =>
      val param = p.getParam(name)
      p.set(param, value)
    }
  }

  /**
   * Check [[allParamSettings]] to ensure:
   *  - The map contains exactly 1 value for each [[Param]] member of the given [[Params]] instance.
   *  - Those values differ from the default values, if defaults exist.
   */
  def validateAllParamSettings(params: T): Unit = {
    val paramsCopy = params.copy(ParamMap.empty)
    allParamSettings.foreach { case (name: String, value: Any) =>
      assert(paramsCopy.hasParam(name))
      val param = paramsCopy.getParam(name)
      paramsCopy.set(param, value)
      if (paramsCopy.hasDefault(param)) {
        assert(paramsCopy.getDefault(param) !== value)
      }
    }
    // Try to check for Params missing from allParamSettings.
    // We filter out input columns since those should not be renamed for these default tests.
    val inputColNames = Set("featuresCol", "labelCol", "weightCol")
    val paramsWithoutInputCols = paramsCopy.params.filter(p => !inputColNames.contains(p.name))
    assert(paramsWithoutInputCols.length === allParamSettings.size)
  }

  /**
   * Checks common requirements for [[Params]] instances.
   * Also calls [[validateAllParamSettings()]].
   *
   * [[Params.params]] should obey these rules:
   *   - params are ordered by names
   *   - param parent has the same UID as the object's UID
   *   - param name is the same as the param method name
   *   - obj.copy should return the same type as the obj
   *
   * This may be overridden, e.g., for [[org.apache.spark.ml.Estimator]].
   */
  def runDefaultParamsTest(): Unit = {
    test("Params: default check") {
      val t = getDefaultInstance
      ParamsSuite.checkParams(t)
      validateAllParamSettings(t)
    }
  }

  /**
   * Checks ML persistence by:
   *  - Creating an instance of [[T]] and setting its params using [[allParamSettings]]
   *  - Saving the instance
   *  - Loading back the instance
   *  - Comparing the original and loaded instances in terms of:
   *     - UID
   *     - Params
   *  - Verifying that the "overwrite" option works
   *
   * This saves to and loads from [[tempDir]], but creates a subdirectory with a random name
   * in order to avoid conflicts from multiple calls to this method.
   */
  def runDefaultReadWriteTest(): Unit = {
    test(s"ML persistence: read/write") {
      val p = getDefaultInstance
      setAllParams(p)
      DefaultReadWriteTest.testDefaultReadWrite(p, testParams = true, tempDir = tempDir)
    }
  }
}
