// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import chisel3._
import pt.kcry.blake3.{Blake3, Hasher}

private[chisel3] object TypeIdHash {

  private def intToBytes(v: Int): Array[Byte] =
    Array((v >> 24).toByte, (v >> 16).toByte, (v >> 8).toByte, v.toByte)

  private def longToBytes(v: Long): Array[Byte] =
    Array(
      (v >> 56).toByte,
      (v >> 48).toByte,
      (v >> 40).toByte,
      (v >> 32).toByte,
      (v >> 24).toByte,
      (v >> 16).toByte,
      (v >> 8).toByte,
      v.toByte
    )

  /** Update a Blake3 hasher with the structural type identity of a Data node.
    *
    * For Records, uses their pre-computed hash (hierarchical, no recursion).
    * For Vecs, hashes length + sample element type.
    * For Elements, hashes class name + width.
    */
  def updateHash(hasher: Hasher, data: Data): Unit = data match {
    case r: Record =>
      hasher.update(longToBytes(r.typeIdHashHi))
      hasher.update(longToBytes(r.typeIdHashLo))
    case v: Vec[_] =>
      hasher.update("Vec")
      hasher.update(intToBytes(v.length))
      updateHash(hasher, v.sample_element)
    case e: EnumType =>
      hasher.update("Enum")
      hasher.update(intToBytes(e.factory.hashCode))
    case e: Element =>
      hasher.update(e.getClass.getName)
      e.width match {
        case KnownWidth(w) => hasher.update(intToBytes(w))
        case UnknownWidth  => hasher.update(intToBytes(-1))
      }
  }

  /** Compute the 128-bit type ID hash for a Record from its _elements. */
  def computeRecordHash(record: Record): (Long, Long) = {
    val hasher = Blake3.newHasher()
    for ((name, data) <- record._elements) {
      hasher.update(name)
      updateHash(hasher, data)
    }
    val result = new Array[Byte](16)
    hasher.done(result)
    val hi = java.nio.ByteBuffer.wrap(result, 0, 8).getLong
    val lo = java.nio.ByteBuffer.wrap(result, 8, 8).getLong
    (hi, lo)
  }
}
