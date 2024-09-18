// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal._
import chisel3.experimental.{OpaqueType, SourceInfo}

import scala.collection.immutable.ListMap

// We could just an OpaqueType, but since OpaqueTypes have some API holes, this non-Data type
private[chisel3] trait DisableImpl {
  private[chisel3] def value: Bool

  def _impl_unary_!(implicit sourceInfo: SourceInfo): Disable = new Disable(!this.value)
}

private[chisel3] trait ObectDisableImpl { self: Disable.type =>

  sealed trait Type

  /** Never disable
    *
    * The simulation construct will always be executing.
    */
  case object Never extends Type

  /** Disable before reset has been seen
    *
    * Disable until reset has been asserted and then deasserted.
    * Also requires
    */
  case object BeforeReset extends Type

  private[chisel3] def withDisable[T](option: Type)(block: => T): T = {
    // Save parentScope
    val parentDisable = Builder.currentDisable

    Builder.currentDisable = option

    val res = block // execute block

    // Return to old scope
    Builder.currentDisable = parentDisable
    res
  }
}

/** Creates a new [[Disable]] scope */
object withDisable {

  /** Creates a new Disable scope
    *
    * @param disable an Optional new implicit Disable, None means no disable
    * @param block the block of code to run with new implicit Disable
    * @return the result of the block
    */
  def apply[T](disable: Disable.Type)(block: => T): T = Disable.withDisable(disable)(block)
}
