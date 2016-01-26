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

package org.apache.spark.ml.feature

import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.sql.types.{DataTypes, DataType}

/**
 * :: Experimental ::
 * A tokenizer that converts the input string to lowercase and then splits it by white spaces.
 */
@Experimental
class Tokenizer(override val uid: String)
  extends UnaryTransformer[String, Seq[String], Tokenizer] with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("tok"))

  override protected def outputDataType: DataType =
    DataTypes.createArrayType(DataTypes.StringType, false)

  override protected def createTransformFunc: String => Seq[String] = {
    _.toLowerCase.split("\\s")
  }

  override def copy(extra: ParamMap): Tokenizer = defaultCopy(extra)
}

@Since("1.6.0")
object Tokenizer extends DefaultParamsReadable[Tokenizer] {

  @Since("1.6.0")
  override def load(path: String): Tokenizer = super.load(path)
}
