// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo
import chisel3.internal.Builder
import chisel3.internal.firrtl.ir

/** A [[Placeholder]] is an _advanced_ API for Chisel generators to generate
  * additional hardware at a different point in the design.
  *
  * For example, this can be used to add a wire _before_ another wire:
  * {{{
  * val placeholder = new Placeholder()
  * val a = Wire(Bool())
  * val b = placeholder.append {
  *   Wire(Bool())
  * }
  * }}}
  *
  * This will generate the following FIRRTL where `b` is declared _before_ `a`:
  * {{{
  * wire b : UInt<1>
  * wire a : UInt<1>
  * }}}
  */
private[chisel3] class Placeholder()(implicit sourceInfo: SourceInfo) {

  private val state = Builder.State.save

  private val placeholder = Builder.pushCommand(new ir.Placeholder(sourceInfo = sourceInfo))

  /** Generate the hardware of the `thunk` and append the result at the point
    * where this [[Placeholder]] was created.  The return value will be what the
    * `thunk` returns which enables getting a reference to the generated
    * hardware.
    *
    * @param thunk the hardware to generate and append to the [[Placeholder]]
    * @return the return value of the `thunk`
    */
  def append[A](thunk: => A): A = Builder.State.guard(state) {
    Builder.currentBlock.get.appendToPlaceholder(placeholder)(thunk)
  }

}
