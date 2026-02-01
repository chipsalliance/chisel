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

  def isValid(using SourceInfo): Bool = _isValidImpl

  final def isOneOf(s: Seq[EnumType])(using SourceInfo): Bool = _isOneOfSeqImpl(s)

  final def isOneOf(
    u1: EnumType,
    u2: EnumType*
  )(
    using SourceInfo
  ): Bool = _isOneOfImpl(u1, u2.toSeq)

  def nameContains(s: String)(using SourceInfo): Bool = _nameContainsImpl(s)

  def next(using SourceInfo): this.type = _nextImpl
}

private[chisel3] trait ChiselEnum$Intf { self: ChiselEnum =>

  def apply(n: UInt)(using SourceInfo): Type = _applyImpl(n)

  def safe(n: UInt)(using SourceInfo): (Type, Bool) = _safeImpl(n)
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
  private def enumValueName(using Quotes): String = {
    import quotes.reflect.*

    // Owner is always the macro call whose owner is the actual ValDef
    val owner = Symbol.spliceOwner.owner
    if !owner.isValDef then report.errorAndAbort("Invalid Value definition")

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
