// See LICENSE for license details.

package chisel3

import chisel3.internal.ExplicitCompileOptions


object Strict {
  implicit object CompileOptions extends ExplicitCompileOptions {
    val connectFieldsMustMatch = true
    val declaredTypeMustBeUnbound = true
    val requireIOWrap = true
    val dontTryConnectionsSwapped = true
    val dontAssumeDirectionality = true
  }
}
