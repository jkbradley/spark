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

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{Param, Params, ParamsSuite}
import org.apache.spark.sql.Dataset

/**
 * Trait for testing [[Param]] settings used in tests.
 *
 * The companion objects of test suites for [[Estimator]] types should implement this trait.
 */
abstract class DefaultEstimatorTest
  [E <: Estimator[M] with MLWritable, M <: Model[M] with MLWritable]
  extends DefaultParamsTest[E] {

  /**
   * Get default Estimator instance.
   * This MUST set the UID to a fixed value, rather than using a random UID.
   */
  def getDefaultEstimator: E

  /**
   * Get default Model instance.
   * This MUST set the UID to the same fixed value as [[getDefaultEstimator]].
   */
  def getDefaultModel: M

  override def getDefaultInstance: E = getDefaultEstimator

  test("Estimator: default check") {
    val estimator = getDefaultEstimator
    ParamsSuite.checkParams(estimator)
    val model = getDefaultModel
    ParamsSuite.checkParams(model)
  }

}
