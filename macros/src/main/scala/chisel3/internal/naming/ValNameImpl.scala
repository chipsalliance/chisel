// See LICENSE for license details.

package chisel3.internal.prefixing

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context

/** Used to contain the val name found during the macro transformation
  * Is later implicitly cast to a ValName
  *
  * @param name Name of the val you assigned to
  */
case class ValNameImpl(name: String)

object ValNameImpl {

  // Enables running a macro on the call site
  implicit def materialize: ValNameImpl = macro detail

  // Update context tree by finding the val name, and providing the implicit value
  def detail(c: Context): c.Expr[ValNameImpl] = {
    import c.universe._
    def allOwners(s: c.Symbol): Seq[c.Symbol] =
      if (s == `NoSymbol`) Nil else s +: allOwners(s.owner)

    val terms = allOwners(c.internal.enclosingOwner).filter(_.isTerm).map(_.asTerm)
    terms.filter(t => t.isVal || t.isLazy).map(_.name.toString).find(_ (0) != '$').map { s =>
      val trim = s.replaceAll("\\s", "")
      c.Expr[ValNameImpl] {
        q"_root_.chisel3.internal.prefixing.ValNameImpl(${trim})"
      }
    }.getOrElse(c.abort(c.enclosingPosition, "Not a valid application."))
  }
}
