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

package org.apache.spark.mllib.linalg

import java.lang.{Double => JavaDouble, Integer => JavaInteger, Iterable => JavaIterable}
import java.util

import scala.annotation.varargs
import scala.collection.JavaConverters._

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}

import org.apache.spark.SparkException
import org.apache.spark.mllib.util.NumericParser
import org.apache.spark.sql.catalyst.UDTRegistry
import org.apache.spark.sql.catalyst.annotation.SQLUserDefinedType
import org.apache.spark.sql.catalyst.expressions.GenericMutableRow
import org.apache.spark.sql.catalyst.types._
import org.apache.spark.sql.Row

/**
 * Represents a numeric vector, whose index type is Int and value type is Double.
 *
 * Note: Users should not implement this interface.
 */
sealed trait Vector extends Serializable {

  /**
   * Size of the vector.
   */
  def size: Int

  /**
   * Converts the instance to a double array.
   */
  def toArray: Array[Double]

  override def equals(other: Any): Boolean = {
    other match {
      case v: Vector =>
        util.Arrays.equals(this.toArray, v.toArray)
      case _ => false
    }
  }

  override def hashCode(): Int = util.Arrays.hashCode(this.toArray)

  /**
   * Converts the instance to a breeze vector.
   */
  private[mllib] def toBreeze: BV[Double]

  /**
   * Gets the value of the ith element.
   * @param i index
   */
  def apply(i: Int): Double = toBreeze(i)

  /**
   * Makes a deep copy of this vector.
   */
  def copy: Vector = {
    throw new NotImplementedError(s"copy is not implemented for ${this.getClass}.")
  }
}

/**
 * Factory methods for [[org.apache.spark.mllib.linalg.Vector]].
 * We don't use the name `Vector` because Scala imports
 * [[scala.collection.immutable.Vector]] by default.
 */
object Vectors {

  UDTRegistry.registerType(scala.reflect.runtime.universe.typeOf[Vector], new VectorUDT())

  /**
   * Creates a dense vector from its values.
   */
  @varargs
  def dense(firstValue: Double, otherValues: Double*): Vector =
    new DenseVector((firstValue +: otherValues).toArray)

  // A dummy implicit is used to avoid signature collision with the one generated by @varargs.
  /**
   * Creates a dense vector from a double array.
   */
  def dense(values: Array[Double]): Vector = new DenseVector(values)

  /**
   * Creates a sparse vector providing its index array and value array.
   *
   * @param size vector size.
   * @param indices index array, must be strictly increasing.
   * @param values value array, must have the same length as indices.
   */
  def sparse(size: Int, indices: Array[Int], values: Array[Double]): Vector =
    new SparseVector(size, indices, values)

  /**
   * Creates a sparse vector using unordered (index, value) pairs.
   *
   * @param size vector size.
   * @param elements vector elements in (index, value) pairs.
   */
  def sparse(size: Int, elements: Seq[(Int, Double)]): Vector = {
    require(size > 0)

    val (indices, values) = elements.sortBy(_._1).unzip
    var prev = -1
    indices.foreach { i =>
      require(prev < i, s"Found duplicate indices: $i.")
      prev = i
    }
    require(prev < size)

    new SparseVector(size, indices.toArray, values.toArray)
  }

  /**
   * Creates a sparse vector using unordered (index, value) pairs in a Java friendly way.
   *
   * @param size vector size.
   * @param elements vector elements in (index, value) pairs.
   */
  def sparse(size: Int, elements: JavaIterable[(JavaInteger, JavaDouble)]): Vector = {
    sparse(size, elements.asScala.map { case (i, x) =>
      (i.intValue(), x.doubleValue())
    }.toSeq)
  }

  /**
   * Creates a dense vector of all zeros.
   *
   * @param size vector size
   * @return a zero vector
   */
  def zeros(size: Int): Vector = {
    new DenseVector(new Array[Double](size))
  }

  /**
   * Parses a string resulted from `Vector#toString` into
   * an [[org.apache.spark.mllib.linalg.Vector]].
   */
  def parse(s: String): Vector = {
    parseNumeric(NumericParser.parse(s))
  }

  private[mllib] def parseNumeric(any: Any): Vector = {
    any match {
      case values: Array[Double] =>
        Vectors.dense(values)
      case Seq(size: Double, indices: Array[Double], values: Array[Double]) =>
        Vectors.sparse(size.toInt, indices.map(_.toInt), values)
      case other =>
        throw new SparkException(s"Cannot parse $other.")
    }
  }

  /**
   * Creates a vector instance from a breeze vector.
   */
  private[mllib] def fromBreeze(breezeVector: BV[Double]): Vector = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1) {
          new DenseVector(v.data)
        } else {
          new DenseVector(v.toArray)  // Can't use underlying array directly, so make a new one
        }
      case v: BSV[Double] =>
        if (v.index.length == v.used) {
          new SparseVector(v.length, v.index, v.data)
        } else {
          new SparseVector(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: BV[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }
}

/**
 * A dense vector represented by a value array.
 */
@SQLUserDefinedType(udt = classOf[DenseVectorUDT])
class DenseVector(val values: Array[Double]) extends Vector {

  override def size: Int = values.length

  override def toString: String = values.mkString("[", ",", "]")

  override def toArray: Array[Double] = values

  private[mllib] override def toBreeze: BV[Double] = new BDV[Double](values)

  override def apply(i: Int) = values(i)

  override def copy: DenseVector = {
    new DenseVector(values.clone())
  }
}

/**
 * A sparse vector represented by an index array and an value array.
 *
 * @param size size of the vector.
 * @param indices index array, assume to be strictly increasing.
 * @param values value array, must have the same length as the index array.
 */
class SparseVector(
    override val size: Int,
    val indices: Array[Int],
    val values: Array[Double]) extends Vector {

  require(indices.length == values.length)

  override def toString: String =
    "(%s,%s,%s)".format(size, indices.mkString("[", ",", "]"), values.mkString("[", ",", "]"))

  override def toArray: Array[Double] = {
    val data = new Array[Double](size)
    var i = 0
    val nnz = indices.length
    while (i < nnz) {
      data(indices(i)) = values(i)
      i += 1
    }
    data
  }

  override def copy: SparseVector = {
    new SparseVector(size, indices.clone(), values.clone())
  }

  private[mllib] override def toBreeze: BV[Double] = new BSV[Double](indices, values, size)
}

/**
 * User-defined type for [[Vector]] which allows easy interaction with SQL
 * via [[org.apache.spark.sql.SchemaRDD]].
 */
private[spark] class VectorUDT extends UserDefinedType[Vector] {

  /**
   * vectorType: 0 = dense, 1 = sparse.
   * dense, sparse: One element holds the vector, and the other is null.
   */
  override def sqlType: StructType = StructType(Seq(
    StructField("vectorType", ByteType, nullable = false),
    StructField("dense", new DenseVectorUDT(), nullable = true),
    StructField("sparse", new SparseVectorUDT(), nullable = true)))

  override def serialize(obj: Any): Row = {
    val row = new GenericMutableRow(3)
    obj match {
      case v: DenseVector =>
        row.setByte(0, 0)
        row.update(1, new DenseVectorUDT().serialize(obj))
        row.setNullAt(2)
      case v: SparseVector =>
        row.setByte(0, 1)
        row.setNullAt(1)
        row.update(2, new SparseVectorUDT().serialize(obj))
    }
    row
  }

  override def deserialize(row: Row): Vector = {
    require(row.length == 3,
      s"VectorUDT.deserialize given row with length ${row.length} but requires length == 3")
    val vectorType = row.getByte(0)
    vectorType match {
      case 0 =>
        new DenseVectorUDT().deserialize(row.getAs[Row](1))
      case 1 =>
        new SparseVectorUDT().deserialize(row.getAs[Row](2))
    }
  }
}

/**
 * User-defined type for [[DenseVector]] which allows easy interaction with SQL
 * via [[org.apache.spark.sql.SchemaRDD]].
 */
private[spark] class DenseVectorUDT extends UserDefinedType[DenseVector] {

  override def sqlType: ArrayType = ArrayType(DoubleType, containsNull = false)

  override def serialize(obj: Any): Row = obj match {
    case v: DenseVector =>
      val row: GenericMutableRow = new GenericMutableRow(v.size)
      var i = 0
      while (i < v.size) {
        row.setDouble(i, v(i))
        i += 1
      }
      row
  }

  override def deserialize(row: Row): DenseVector = {
    val values = new Array[Double](row.length)
    var i = 0
    while (i < row.length) {
      values(i) = row.getDouble(i)
      i += 1
    }
    new DenseVector(values)
  }
}

/**
 * User-defined type for [[SparseVector]] which allows easy interaction with SQL
 * via [[org.apache.spark.sql.SchemaRDD]].
 */
private[spark] class SparseVectorUDT extends UserDefinedType[SparseVector] {

  override def sqlType: StructType = StructType(Seq(
    StructField("size", IntegerType, nullable = false),
    StructField("indices", ArrayType(DoubleType, containsNull = false), nullable = false),
    StructField("values", ArrayType(DoubleType, containsNull = false), nullable = false)))

  override def serialize(obj: Any): Row = obj match {
    case v: SparseVector =>
      val nnz = v.indices.size
      val row: GenericMutableRow = new GenericMutableRow(1 + 2 * nnz)
      row.setInt(0, v.size)
      var i = 0
      while (i < nnz) {
        row.setInt(1 + i, v.indices(i))
        i += 1
      }
      i = 0
      while (i < nnz) {
        row.setDouble(1 + nnz + i, v.values(i))
        i += 1
      }
      row
  }

  override def deserialize(row: Row): SparseVector = {
    require(row.length >= 1,
      s"SparseVectorUDT.deserialize given row with length ${row.length} but requires length >= 1")
    val vSize = row.getInt(0)
    val nnz: Int = (row.length - 1) / 2
    require(nnz * 2 + 1 == row.length,
      s"SparseVectorUDT.deserialize given row with non-matching indices, values lengths")
    val indices = new Array[Int](nnz)
    val values = new Array[Double](nnz)
    var i = 0
    while (i < nnz) {
      indices(i) = row.getInt(1 + i)
      values(i) = row.getDouble(1 + nnz + i)
      i += 1
    }
    new SparseVector(vSize, indices, values)
  }
}
