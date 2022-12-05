// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._

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
  * @see RecordSpec in Chisel's tests for example usage and expected output
  */
trait OpaqueType { self: Record =>

  /** If set to true, indicates that this Record is an OpaqueType
    *
    * Users can override this if they need more dynamic control over the behavior for when
    * instances of this type are considered opaque
    */
  def opaqueType: Boolean = true
}
