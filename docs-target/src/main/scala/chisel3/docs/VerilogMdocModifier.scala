package chisel3.docs

import java.nio.file.Files
import java.nio.file.Paths
import mdoc._
import scala.meta.inputs.Position

/** Custom modifier for rendering Chisel-generated Verilog
  *
  * See chisel3/docs/README.md for use
  */
class VerilogMdocModifier extends PostModifier {
  val name = "verilog"
  def process(ctx: PostModifierContext): String = {
    val result =
      ctx.variables.foldLeft(Option("")) {
        case (Some(acc), variable) if variable.staticType == "String" =>
          Some(acc + variable.runtimeValue)
        case (Some(_), badVar) =>
          ctx.reporter.error(
            badVar.pos,
            s"""type mismatch:
               |expected: String
               |received: ${badVar.runtimeValue}""".stripMargin
          )
          None
        case (None, _) => None
      }
    result match {
      case Some(content) => s"```verilog\n$content```"
      case None          => ""
    }
  }
}
