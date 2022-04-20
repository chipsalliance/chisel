// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

/** Represents a clone of an underlying object. This is used to support CloneModuleAsRecord and Instance/Definition.
  *
  * @note We don't actually "clone" anything in the traditional sense but is a placeholder so we lazily clone internal state
  */
trait IsClone[+T] {
  // Underlying object of which this is a clone of
  private[chisel3] def getProto: T

  /** Determines whether another object is a clone of the same underlying proto
    *
    * @param a
    */
  def hasSameProto(a: Any): Boolean = {
    val aProto = a match {
      case x: IsClone[_] => x.getProto
      case o => o
    }
    this == aProto || getProto == aProto
  }
}
