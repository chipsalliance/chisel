// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3.internal.{Builder, HasId}
import chisel3.internal.firrtl.ir.FirrtlComment

/** Use to leave a comment in the generated FIRRTL.
  * 
  * These can be useful for debugging complex generators.
  * Note that these are comments and thus are ignored by the FIRRTL compiler.
  * 
  * @example {{{
  * 
  * firrtlComment("This is a comment")
  * val w = Wire(Bool())
  * firrtlComment("This is another comment")
  * 
  * }}}
  */
object firrtlComment {

  /** Leave a comment in the generated FIRRTL.
    * 
    * @param text The text of the comment
    */
  def apply(text: String): Unit = {
    Builder.pushCommand(FirrtlComment(text))
  }
}
