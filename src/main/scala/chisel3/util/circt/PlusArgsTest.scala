// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.SourceInfo
import chisel3.internal.Builder

object PlusArgsTest {

  /** Creates an intrinsic which calls \$test\$plusargs to test for the existence of the string str in the simulator command line.
    *
    * @example {{{
    * b := PlusArgsTest("FOO")
    * }}}
    */
  def apply(str: String)(implicit sourceInfo: SourceInfo): Bool = {
    IntrinsicExpr("circt_plusargs_test", Bool(), "FORMAT" -> str)()
  }

  @deprecated("use PlusArgsTest(str) instead")
  def apply[T <: Data](gen: T, str: String)(implicit sourceInfo: SourceInfo): Bool = apply(str)
}
