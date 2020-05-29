package fix

import scalafix.v1._
import scala.meta._

case class AmbiguousLiteralExtract(lit: Lit.Int) extends Diagnostic {
  override def position: Position = lit.pos
  override def message: String =
    s"Passing an Int to .U is usually a mistake: it does not set the width but does a bit extract."
}

class ChiselLinter extends SemanticRule("ChiselLinter") {

  override def fix(implicit doc: SemanticDocument): Patch = {
    doc.tree.collect {
      case Term.Apply(Term.Select(Lit.Int(_), Term.Name("U")), Seq(il: Lit.Int)) =>
        Patch.lint(AmbiguousLiteralExtract(il))
    }.asPatch
  }

}
