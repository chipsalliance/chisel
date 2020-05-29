package fix

import scalafix.v1._
import scala.meta._

case class AmbiguousLiteralExtract(lit: Lit.Int) extends Diagnostic {
  override def position: Position = lit.pos
  override def message: String =
    s"Passing an Int to .U is usually a mistake: it does not set the width but does a bit extract."
}

case class ReferentialEqualityOnData(eq: Term.ApplyInfix) extends Diagnostic {
  override def position: Position = eq.pos
  override def message: String =
    s"Using == on a hardware type (e.g. UInt) is usually a mistake: did you mean to use the === operator?"
}

class ChiselLinter extends SemanticRule("ChiselLinter") {

  private def isChiselType(symbol: Symbol)(implicit doc: SemanticDocument): Boolean = symbol.info match {
    case Some(i) =>
      i.signature match {
        case TypeSignature(_, _, upper: TypeRef) => upper.symbol.toString.contains("chisel3")
        case MethodSignature(_, _, tr: TypeRef) => tr.symbol.toString.contains("chisel3")
        case _ => false
      }
    case _ => false
  }

  override def fix(implicit doc: SemanticDocument): Patch = {
    doc.tree.collect {
      case Term.Apply(Term.Select(Lit.Int(_), Term.Name("U")), Seq(il: Lit.Int)) =>
        Patch.lint(AmbiguousLiteralExtract(il))
      case eq @ Term.ApplyInfix(lhs, Term.Name("=="), _, _) if (isChiselType(lhs.symbol)) =>
        Patch.lint(ReferentialEqualityOnData(eq))
    }.asPatch
  }

}
