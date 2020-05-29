package fix

import scalafix.v1._
import scala.meta._

case class AmbiguousLiteralExtract(pos: Position) extends Diagnostic {
  override def position: Position = pos
  override def message: String =
    s"Passing an Int to .U is usually a mistake: it does not set the width but does a bit extract."
}

class ChiselLinter extends SemanticRule("ChiselLinter") {

  override def fix(implicit doc: SemanticDocument): Patch = {
    ???
  }

}
