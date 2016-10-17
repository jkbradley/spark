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

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.ml.param.{Param, ParamMap, Params, ParamsSuite}


/**
 * Trait for testing [[Param]] settings used in tests.
 *
 * The companion objects of test suites for [[Params]] types should implement this trait.
 */
abstract class DefaultParamsTest[T <: Params] extends SparkFunSuite {

  /** Get an instance of this [[Params]] type with default [[Param]] values */
  def getDefaultInstance: T

  /** Non-default values for ALL Params; these must be different from the default values. */
  def allParamSettings: Map[String, Any]

  /**
   * Check [[allParamSettings]] to ensure:
   *  - It contains exactly 1 value for each [[Param]] member of the given [[Params]] instance.
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

  test("Params: default check") {
    val t = getDefaultInstance
    ParamsSuite.checkParams(t)
    validateAllParamSettings(t)
  }
}
