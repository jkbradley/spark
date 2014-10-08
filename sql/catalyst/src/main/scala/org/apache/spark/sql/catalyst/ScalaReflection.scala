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

package org.apache.spark.sql.catalyst

import java.sql.Timestamp

import org.apache.spark.sql.catalyst.expressions.{GenericRow, Attribute, AttributeReference, Row}
import org.apache.spark.sql.catalyst.plans.logical.LocalRelation
import org.apache.spark.sql.catalyst.types._

/**
 * Provides experimental support for generating catalyst schemas for scala objects.
 */
object ScalaReflection {
  import scala.reflect.runtime.universe._

  case class Schema(dataType: DataType, nullable: Boolean)

  /** Converts Scala objects to catalyst rows / types */
  /*
  def convertToCatalyst(a: Any, dataType: DataType): Any = a match {
    // TODO: Why does this not need to flatMap stuff?  Does it not support nesting?
    case o: Option[_] =>
      println(s"convertToCatalyst: option")
      o.map(convertToCatalyst(_, dataType)).orNull
    case s: Seq[_] =>
      println(s"convertToCatalyst: array")
      s.map(convertToCatalyst(_, null))
    case m: Map[_, _] =>
      println(s"convertToCatalyst: map")
      m.map { case (k, v) =>
      convertToCatalyst(k, null) -> convertToCatalyst(v, null)
    }
    case p: Product =>
      println(s"convertToCatalyst: struct")
      new GenericRow(p.productIterator.map(convertToCatalyst(_, null)).toArray)
    case other =>
      println(s"convertToCatalyst: other")
      other
  }
  */

  def convertToCatalyst(a: Any, dataType: DataType): Any = {
    println(s"convertToCatalyst: a = $a, dataType = $dataType")
    (a, dataType) match {
      // TODO: Why does this not need to flatMap stuff?  Does it not support nesting?
      case (o: Option[_], _) =>
        println(s"convertToCatalyst: option")
        o.map(convertToCatalyst(_, dataType)).orNull
      case (s: Seq[_], arrayType: ArrayType) =>
        println(s"convertToCatalyst: array")
        s.map(convertToCatalyst(_, arrayType.elementType))
      case (m: Map[_, _], mapType: MapType) =>
        println(s"convertToCatalyst: map")
        m.map { case (k, v) =>
          convertToCatalyst(k, mapType.keyType) -> convertToCatalyst(v, mapType.valueType)
        }
      case (p: Product, structType: StructType) =>
        println(s"convertToCatalyst: struct with")
        println(s"\t p: $p")
        println(s"\t structType: $structType")
        new GenericRow(
          p.productIterator.toSeq.zip(structType.fields).map { case (elem, field) =>
            convertToCatalyst(elem, field.dataType)
          }.toArray)
      case (udt: Any, udtType: UserDefinedType[_]) =>
        println(s"convertToCatalyst: udt")
        udtType.serialize(udt)
      case (other, _) =>
        println(s"convertToCatalyst: other")
        other
    }
  }

  /*
  /** Converts Catalyst types used internally in rows to standard Scala types */
  def convertToScala(a: Any, dataType: DataType): Any = (a, dataType) match {
    // TODO: Why does this not need to flatMap stuff?  Does it not support nesting?
    // TODO: What about Option and Product?
    case (s: Seq[_], arrayType: ArrayType) => s.map(convertToScala(_, arrayType.elementType))
    case (m: Map[_, _], mapType: MapType) => m.map { case (k, v) =>
      convertToScala(k, mapType.keyType) -> convertToScala(v, mapType.valueType)
    }
    case (udt: Row, udtType: UserDefinedType[_]) => udtType.deserialize(udt)
    case (other, _) => other
  }

  def convertRowToScala(r: Row, schema: StructType): Row = {
    new GenericRow(r.toArray.map(convertToScala(_, schema)))
  }
  */

  /** Returns a Sequence of attributes for the given case class type. */
  def attributesFor[T: TypeTag](
      udtRegistry: scala.collection.Map[Any, UserDefinedType[_]]): Seq[Attribute] = {
    schemaFor[T](udtRegistry) match {
      case Schema(s: StructType, _) =>
        s.fields.map(f => AttributeReference(f.name, f.dataType, f.nullable)())
    }
  }

  /** Returns a catalyst DataType and its nullability for the given Scala Type using reflection. */
  def schemaFor[T: TypeTag](udtRegistry: scala.collection.Map[Any, UserDefinedType[_]]): Schema = {
    println(s"schemaFor: ${typeTag[T]}")
    if (udtRegistry.contains(typeTag[T])) {
      println(s"  schemaFor T matched udtRegistry")
      val udtStructType: StructType = udtRegistry(typeTag[T]).dataType
      Schema(udtStructType, nullable = true)
    } else {
      schemaFor(typeOf[T])
    }
  }

  /**
   * Returns a catalyst DataType and its nullability for the given Scala Type using reflection.
   * TODO: ADD DOC
   */
  def schemaFor(tpe: `Type`): Schema = tpe match {
    case t if t <:< typeOf[Option[_]] =>
      val TypeRef(_, _, Seq(optType)) = t
      Schema(schemaFor(optType).dataType, nullable = true)
    case t if t <:< typeOf[Product] =>
      val formalTypeArgs = t.typeSymbol.asClass.typeParams
      val TypeRef(_, _, actualTypeArgs) = t
      val params = t.member(nme.CONSTRUCTOR).asMethod.paramss
      Schema(StructType(
        params.head.map { p =>
          val Schema(dataType, nullable) =
            schemaFor(p.typeSignature.substituteTypes(formalTypeArgs, actualTypeArgs))
          StructField(p.name.toString, dataType, nullable)
        }), nullable = true)
    // Need to decide if we actually need a special type here.
    case t if t <:< typeOf[Array[Byte]] => Schema(BinaryType, nullable = true)
    case t if t <:< typeOf[Array[_]] =>
      sys.error(s"Only Array[Byte] supported now, use Seq instead of $t")
    case t if t <:< typeOf[Seq[_]] =>
      val TypeRef(_, _, Seq(elementType)) = t
      val Schema(dataType, nullable) = schemaFor(elementType)
      Schema(ArrayType(dataType, containsNull = nullable), nullable = true)
    case t if t <:< typeOf[Map[_,_]] =>
      val TypeRef(_, _, Seq(keyType, valueType)) = t
      val Schema(valueDataType, valueNullable) = schemaFor(valueType)
      Schema(MapType(schemaFor(keyType).dataType,
        valueDataType, valueContainsNull = valueNullable), nullable = true)
    case t if t <:< typeOf[String]            => Schema(StringType, nullable = true)
    case t if t <:< typeOf[Timestamp] => Schema(TimestampType, nullable = true)
    case t if t <:< typeOf[BigDecimal] => Schema(DecimalType, nullable = true)
    case t if t <:< typeOf[java.lang.Integer] => Schema(IntegerType, nullable = true)
    case t if t <:< typeOf[java.lang.Long] => Schema(LongType, nullable = true)
    case t if t <:< typeOf[java.lang.Double] => Schema(DoubleType, nullable = true)
    case t if t <:< typeOf[java.lang.Float] => Schema(FloatType, nullable = true)
    case t if t <:< typeOf[java.lang.Short] => Schema(ShortType, nullable = true)
    case t if t <:< typeOf[java.lang.Byte] => Schema(ByteType, nullable = true)
    case t if t <:< typeOf[java.lang.Boolean] => Schema(BooleanType, nullable = true)
    case t if t <:< definitions.IntTpe => Schema(IntegerType, nullable = false)
    case t if t <:< definitions.LongTpe => Schema(LongType, nullable = false)
    case t if t <:< definitions.DoubleTpe => Schema(DoubleType, nullable = false)
    case t if t <:< definitions.FloatTpe => Schema(FloatType, nullable = false)
    case t if t <:< definitions.ShortTpe => Schema(ShortType, nullable = false)
    case t if t <:< definitions.ByteTpe => Schema(ByteType, nullable = false)
    case t if t <:< definitions.BooleanTpe => Schema(BooleanType, nullable = false)
  }

  def typeOfObject: PartialFunction[Any, DataType] = {
    // The data type can be determined without ambiguity.
    case obj: BooleanType.JvmType => BooleanType
    case obj: BinaryType.JvmType => BinaryType
    case obj: StringType.JvmType => StringType
    case obj: ByteType.JvmType => ByteType
    case obj: ShortType.JvmType => ShortType
    case obj: IntegerType.JvmType => IntegerType
    case obj: LongType.JvmType => LongType
    case obj: FloatType.JvmType => FloatType
    case obj: DoubleType.JvmType => DoubleType
    case obj: DecimalType.JvmType => DecimalType
    case obj: TimestampType.JvmType => TimestampType
    case null => NullType
    // For other cases, there is no obvious mapping from the type of the given object to a
    // Catalyst data type. A user should provide his/her specific rules
    // (in a user-defined PartialFunction) to infer the Catalyst data type for other types of
    // objects and then compose the user-defined PartialFunction with this one.
  }

  implicit class CaseClassRelation[A <: Product : TypeTag](data: Seq[A]) {

    /**
     * Implicitly added to Sequences of case class objects.  Returns a catalyst logical relation
     * for the the data in the sequence.
     */
    def asRelation: LocalRelation = {
      // Pass empty map to attributesFor since this method is only used for debugging Catalyst,
      // not used with SparkSQL.
      val output = attributesFor[A](Map.empty)
      LocalRelation(output, data)
    }
  }
}
