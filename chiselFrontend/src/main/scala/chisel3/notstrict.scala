// See LICENSE for license details.

package chisel3

import chisel3.internal.ExplicitCompileOptions


object NotStrict {
  implicit object NotStrictCompileOptions extends ExplicitCompileOptions {
    val connectFieldsMustMatch = false
    val declaredTypeMustBeUnbound = false
    val requireIOWrap = false
    val dontTryConnectionsSwapped = false
    val dontAssumeDirectionality = false
  }
}
