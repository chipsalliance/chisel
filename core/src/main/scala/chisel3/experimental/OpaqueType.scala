// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._
import chisel3.internal.{Builder, ChildBinding}
import chisel3.internal.firrtl.Arg

/** Indicates if this Record represents an "Opaque Type"
  *
  * Opaque types provide a mechanism for user-defined types
  * that do not impose any "boxing" overhead in the emitted FIRRTL and Verilog.
  * You can think about an opaque type Record as a box around
  * a single element that only exists at Chisel elaboration time.
  * Put another way, if this trait is mixed into a Record,
  * the Record may only contain a single element with an empty name
  * and there will be no `_` in the name for that element in the emitted Verilog.
  *
  * @see OpaqueTypeSpec in Chisel's tests for example usage and expected output
  */
// Having both extends Data and the self type of Record may seem redundant, but it isn't
// The self-type has to do with how they are implemented (via a single unnamed element),
//   we eventually want to lift it in a backwards compatible way, by adding a new API
trait OpaqueType extends Data { self: Record =>

  abstract override private[chisel3] def _asUIntImpl(first: Boolean)(implicit sourceInfo: SourceInfo): UInt = {
    if (errorOnAsUInt) {
      Builder.error(s"${this._localErrorContext} does not support .asUInt.")
    }
    super._asUIntImpl(first)
  }

  /** If set to true, calling .asUInt on instances of this type will throw an Exception
    *
    * Users can override this to increase the "opacity" of their type.
    */
  protected def errorOnAsUInt: Boolean = false

  /** If set to true, indicates that this Record is an OpaqueType
    *
    * Users can override this if they need more dynamic control over the behavior for when
    * instances of this type are considered opaque
    */
  def opaqueType: Boolean = true
}
