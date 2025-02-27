// SPDX-License-Identifier: Apache-2.0

package chisel3

object FormalContract extends ObjectFormalContractImpl {

  /** Create a `contract` block with one or more arguments and results. */
  transparent inline def apply[T <: Data](inline args: T*): Any =
    ${ FormalContractMacro('args) }
}

object TestFormalContract {
  val a = IO(Input(Clock()))
  val b = IO(Input(UInt(42.W)))
  FormalContract {}
  val x: Clock = FormalContract(a) { case a => }
  val y: (Clock, UInt) = FormalContract(a, b) { case (a, b) => }
}
