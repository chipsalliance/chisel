// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo
import scala.quoted.*

private[chisel3] trait EnumTypeIntf { self: EnumType =>

  final def ===(that: EnumType)(using SourceInfo): Bool = _impl_===(that)
  final def =/=(that: EnumType)(using SourceInfo): Bool = _impl_=/=(that)
  final def <(that:   EnumType)(using SourceInfo): Bool = _impl_<(that)
  final def <=(that:  EnumType)(using SourceInfo): Bool = _impl_<=(that)
  final def >(that:   EnumType)(using SourceInfo): Bool = _impl_>(that)
  final def >=(that:  EnumType)(using SourceInfo): Bool = _impl_>=(that)
}

private[chisel3] trait ChiselEnumIntf { self: ChiselEnum =>
  protected inline def Value: Type =
    ${ ChiselEnumMacros.valImpl[Type]('{ this }) }

  protected inline def Value(id: UInt): Type =
    ${ ChiselEnumMacros.valCustomImpl[Type]('{ this }, '{ id }) }

  private[chisel3] def _value_impl(name: String):           Type = do_Value(name)
  private[chisel3] def _value_impl(name: String, id: UInt): Type = do_Value(name, id)
}

private[chisel3] object ChiselEnumMacros {

  // Derive the name of the Enum value from the symbol owner of the
  // macro expansion
  //
  // Expansion site should be a val inside a concrete object that
  // extends ChiselEnum. In that case, walk up the owner chain
  // from `Symbol.spliceOwner` to find the enclosing val
  private def enumValueName(using Quotes): String = {
    import quotes.reflect.*

    def enclosingTerm(sym: Symbol): Symbol = sym match {
      case sym if sym.flags.is(Flags.Macro) => enclosingTerm(sym.owner)
      case sym if !sym.isTerm               => enclosingTerm(sym.owner)
      case sym if sym.isValDef              => sym
      case _                                => enclosingTerm(sym.owner)
    }

    val owner = enclosingTerm(Symbol.spliceOwner)
    val name = owner.name.trim

    if name.contains(" ") then report.errorAndAbort("Value cannot be called without assigning to an enum")
    name
  }

  def valImpl[T: Type](self: Expr[ChiselEnum])(using Quotes): Expr[T] = {
    import quotes.reflect.*
    val name = enumValueName
    '{ $self._value_impl(${ Expr(name) }).asInstanceOf[T] }
  }

  def valCustomImpl[T: Type](self: Expr[ChiselEnum], id: Expr[UInt])(using Quotes): Expr[T] = {
    import quotes.reflect.*
    val name = enumValueName
    '{ $self._value_impl(${ Expr(name) }, $id).asInstanceOf[T] }
  }
}
