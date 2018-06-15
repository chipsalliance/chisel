// See LICENSE.SiFive for license details.

package chisel3.internal

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context

case class ValNameImpl(name: String)

object ValNameImpl
{
  implicit def materialize: ValNameImpl = macro detail
  def detail(c: Context): c.Expr[ValNameImpl] = {
    import c.universe._
    def allOwners(s: c.Symbol): Seq[c.Symbol] =
      if (s == `NoSymbol`) Nil else s +: allOwners(s.owner)
    val terms = allOwners(c.internal.enclosingOwner).filter(_.isTerm).map(_.asTerm)
    terms.filter(t => t.isVal || t.isLazy).map(_.name.toString).find(_(0) != '$').map { s =>
      val trim = s.replaceAll("\\s", "")
      c.Expr[ValNameImpl] { q"_root_.chisel3.internal.ValNameImpl(${trim})" }
    }.getOrElse(c.abort(c.enclosingPosition, "Not a valid application."))
  }
}
