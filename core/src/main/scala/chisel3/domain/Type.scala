// SPDX-License-Identifier: Apache-2.0

package chisel3.domain

import chisel3.{fromIntToLiteral, Data, Element, Printable, UInt, UnknownWidth, Width}
import chisel3.experimental.SourceInfo
import chisel3.internal.{throwException, Builder}

/** A [[Data]] that is used to communicate information of a specific domain
  * kind.
  */
final class Type private[domain] (val domain: Domain) extends Element { self =>

  private[chisel3] def _asUIntImpl(first: Boolean)(implicit sourceInfo: SourceInfo): chisel3.UInt = {
    Builder.error(s"${this._localErrorContext} does not support .asUInt.")
    0.U
  }

  override protected def _fromUInt(that: UInt)(implicit sourceInfo: SourceInfo): Data = {
    Builder.exception(s"${this._localErrorContext} cannot be driven by UInt")
  }

  override def cloneType: this.type = new Type(domain).asInstanceOf[this.type]

  override def toPrintable: Printable =
    throwException(s"'domain.Type' does not support hardware printing" + this._errorContext)

  private[chisel3] def width: Width = UnknownWidth

  addDomain(domain)

}
