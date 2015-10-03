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

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.util.collection.BitSet


/**
 * A range of bits within a larger distributed bit vector.
 * @param from starting index (inclusive) of larger distributed bit vector represented by this instance
 * @param to ending index (exclusive) of larger distributed bit vector represented by this instance
 */
private[impl] class BitSubvector(val from: Int, val to: Int) extends Serializable {

  val numBits: Int = to - from

  private val bits: BitSet = new BitSet(numBits)

  /** Set a bit in this instance using an external index */
  def set(idx: Int): Unit = bits.set(toInternalIdx(idx))

  def get(idx: Int): Boolean = bits.get(toInternalIdx(idx))

  /** Get an iterator over the external indices of the set bits. */
  def iterator: Iterator[Int] = new Iterator[Int] {
    val iter = bits.iterator
    override def hasNext: Boolean = iter.hasNext
    override def next(): Int = toExternalIdx(iter.next())
  }

  /**
   * Bit-wise OR with another BitSubvector. Bits are matched according to external index (ORed against 0 if absent in
   * the other BitSubvector). This method mutates the current instance in-place.
   */
  def |=(other: BitSubvector): Unit = {
    require(from <= other.from && to >= other.to)
    val delta = other.from - from
    bits.orWithOffset(other.bits, delta, math.min(numBits, other.numBits - delta))
  }

  private def toInternalIdx(idx: Int): Int = {
    require(idx >= from && idx < to)
    idx - from
  }
  private def toExternalIdx(idx: Int): Int = {
    idx + from
  }
}

private[impl] object BitSubvector {

  def merge(parts1: Array[BitSubvector], parts2: Array[BitSubvector]): Array[BitSubvector] = {
    // Merge sorted parts1, parts2
    val sortedSubvectors = (parts1 ++ parts2).sortBy(_.from)
    if (sortedSubvectors.nonEmpty) {
      // Merge adjacent PartialBitVectors (for adjacent node ranges)
      val newSubvectorRanges: Array[(Int, Int)] = {
        val newSubvRanges = ArrayBuffer.empty[(Int, Int)]
        var i = 1
        var currentFrom = sortedSubvectors.head.from
        while (i < sortedSubvectors.length) {
          if (sortedSubvectors(i - 1).to != sortedSubvectors(i).from) {
            newSubvRanges.append((currentFrom, sortedSubvectors(i - 1).to))
            currentFrom = sortedSubvectors(i).from
          }
          i += 1
        }
        newSubvRanges.append((currentFrom, sortedSubvectors.last.to))
        newSubvRanges.toArray
      }
      val newSubvectors = newSubvectorRanges.map { case (from, to) => new BitSubvector(from, to) }
      var curNewSubvIdx = 0
      sortedSubvectors.foreach { subv =>
        if (subv.to > newSubvectors(curNewSubvIdx).to) curNewSubvIdx += 1
        val newSubv = newSubvectors(curNewSubvIdx)
        // TODO: More efficient (word-level) copy.>
//        newSubv.copyBitsFrom(subv)
//        subv.iterator.foreach(idx => newSubv.set(idx))
        newSubv |= subv
      }
      assert(curNewSubvIdx + 1 == newSubvectors.length) // sanity check
      newSubvectors
    } else {
      Array.empty[BitSubvector]
    }
  }
}
