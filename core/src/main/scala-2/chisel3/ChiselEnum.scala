// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context
import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.SourceInfoTransform

abstract class EnumType(factory: ChiselEnum, selfAnnotating: Boolean = true)
    extends EnumTypeImpl(factory, selfAnnotating) {

  final def ===(that: EnumType): Bool = macro SourceInfoTransform.thatArg
  final def =/=(that: EnumType): Bool = macro SourceInfoTransform.thatArg
  final def <(that:   EnumType): Bool = macro SourceInfoTransform.thatArg
  final def <=(that:  EnumType): Bool = macro SourceInfoTransform.thatArg
  final def >(that:   EnumType): Bool = macro SourceInfoTransform.thatArg
  final def >=(that:  EnumType): Bool = macro SourceInfoTransform.thatArg

  def do_===(that: EnumType)(implicit sourceInfo: SourceInfo): Bool = _impl_===(that)
  def do_=/=(that: EnumType)(implicit sourceInfo: SourceInfo): Bool = _impl_=/=(that)
  def do_<(that:   EnumType)(implicit sourceInfo: SourceInfo): Bool = _impl_<(that)
  def do_>(that:   EnumType)(implicit sourceInfo: SourceInfo): Bool = _impl_>(that)
  def do_<=(that:  EnumType)(implicit sourceInfo: SourceInfo): Bool = _impl_<=(that)
  def do_>=(that:  EnumType)(implicit sourceInfo: SourceInfo): Bool = _impl_>=(that)
}

abstract class ChiselEnum extends ChiselEnumImpl {
  protected def Value: Type = macro EnumMacros.ValImpl
  protected def Value(id: UInt): Type = macro EnumMacros.ValCustomImpl
}

private[chisel3] object EnumMacros {
  def ValImpl(c: Context): c.Tree = {
    import c.universe._

    // Much thanks to michael_s for this solution:
    // stackoverflow.com/questions/18450203/retrieve-the-name-of-the-value-a-scala-macro-invocation-will-be-assigned-to
    val term = c.internal.enclosingOwner
    val name = term.name.decodedName.toString.trim

    if (name.contains(" ")) {
      c.abort(c.enclosingPosition, "Value cannot be called without assigning to an enum")
    }

    q"""this.do_Value($name)"""
  }

  def ValCustomImpl(c: Context)(id: c.Expr[UInt]): c.universe.Tree = {
    import c.universe._

    val term = c.internal.enclosingOwner
    val name = term.name.decodedName.toString.trim

    if (name.contains(" ")) {
      c.abort(c.enclosingPosition, "Value cannot be called without assigning to an enum")
    }

    q"""this.do_Value($name, $id)"""
  }
}
