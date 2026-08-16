package fix

import scalafix.v1._
import scala.meta._

case class AmbiguousLiteralExtract(conv: String, valLit: Lit.Int, bitLit: Lit.Int) extends Diagnostic {
  override def position: Position = valLit.pos
  override def message: String =
    s"Passing an integer to .${conv} does not set width but does a bit extract. If you really want this bit extract, use ${valLit}.${conv}.apply(${bitLit}) instead."
}

case class ReferentialEqualityOnData(eq: Term.ApplyInfix) extends Diagnostic {
  override def position: Position = eq.pos
  override def message: String =
    s"Did you mean === (binary equality) instead of == (referential equality? If you really want referential equality, use the named equals method."
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

  private val litConvs = Set("U", "S", "F")

  override def fix(implicit doc: SemanticDocument): Patch = {
    doc.tree.collect {
      case Term.Apply(Term.Select(vl: Lit.Int, Term.Name(conv)), Seq(bl: Lit.Int)) if litConvs.contains(conv) =>
        Patch.lint(AmbiguousLiteralExtract(conv, vl, bl))
      case eq @ Term.ApplyInfix(lhs, Term.Name("=="), _, _) if (isChiselType(lhs.symbol)) =>
        Patch.lint(ReferentialEqualityOnData(eq))
    }.asPatch
  }

}
