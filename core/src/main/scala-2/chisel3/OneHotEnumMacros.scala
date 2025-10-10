// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.reflect.macros.blackbox.Context
import scala.language.experimental.macros

private[chisel3] trait OneHotEnumIntf extends ChiselEnumIntf { self: OneHotEnum =>
  override def Value:           Type = macro OneHotEnumMacros.ValImpl
  override def Value(id: UInt): Type = macro OneHotEnumMacros.ValCustomImpl
}

private[chisel3] object OneHotEnumMacros {
  def ValImpl(c: Context): c.Tree = {
    import c.universe._

    val term = c.internal.enclosingOwner
    val name = term.name.decodedName.toString.trim

    if (name.contains(" ")) {
      c.abort(c.enclosingPosition, "Value cannot be called without assigning to an enum")
    }

    q"""this.do_OHValue($name)"""
  }

  def ValCustomImpl(c: Context)(id: c.Expr[UInt]): c.universe.Tree = {
    c.abort(c.enclosingPosition, "OneHotEnum does not support custom values")
  }
}
