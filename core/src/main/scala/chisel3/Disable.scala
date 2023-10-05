// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal._
import chisel3.experimental.{IntrinsicModule, OpaqueType, SourceInfo}
import chisel3.internal.sourceinfo.SourceInfoTransform

import scala.language.experimental.macros
import scala.collection.immutable.ListMap

/** API for handling disabling of simulation constructs
  *
  * Disables may be non-synthesizable so they can only be used for disabling simulation constructs
  *
  * The default disable is the "hasBeenReset" of the currently in scope reset.
  * It can be set by the user via the [[withDisable]] API
  *
  * Users can access the current `Disable` with [[Module.disable]]
  */
// We could just an OpaqueType, but since OpaqueTypes have some API holes, this non-Data type
class Disable private[chisel3] (private[chisel3] val value: Bool) {

  /** Logical not
    *
    * @return invert the logical value of this `Disable`
    * @group Bitwise
    */
  final def unary_! : Disable = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_!(implicit sourceInfo: SourceInfo): Disable = new Disable(!this.value)
}

object Disable {

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

/**
  */
// Note because this uses abstract reset, it cannot be instantiable until we have the ability to
// create both sync and sync reset instances from the same Definition
private[chisel3] class HasBeenResetIntrinsic(implicit sourceInfo: SourceInfo)
    extends IntrinsicModule(f"circt_has_been_reset") {
  // Compiler plugin does not run on core so we have to suggest names
  val clock = IO(Input(Clock())).suggestName("clock")
  val reset = IO(Input(Reset())).suggestName("reset")
  val out = IO(Output(Bool())).suggestName("out")
}
