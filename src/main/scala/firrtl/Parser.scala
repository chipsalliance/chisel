// See LICENSE for license details.

package firrtl

import org.antlr.v4.runtime._
import org.antlr.v4.runtime.atn._
import com.typesafe.scalalogging.LazyLogging
import firrtl.ir._
import firrtl.Utils.time
import firrtl.antlr.{FIRRTLParser, _}

class ParserException(message: String) extends FIRRTLException(message)

case class ParameterNotSpecifiedException(message: String) extends ParserException(message)
case class ParameterRedefinedException(message: String) extends ParserException(message)
case class InvalidStringLitException(message: String) extends ParserException(message)
case class InvalidEscapeCharException(message: String) extends ParserException(message)
case class SyntaxErrorsException(message: String) extends ParserException(message)


object Parser extends LazyLogging {

  /** Parses a file in a given filename and returns a parsed [[Circuit]] */
  def parseFile(filename: String, infoMode: InfoMode): Circuit =
    parseCharStream(CharStreams.fromFileName(filename), infoMode)

  /** Parses a String and returns a parsed [[Circuit]] */
  def parseString(text: String, infoMode: InfoMode): Circuit =
    parseCharStream(CharStreams.fromString(text), infoMode)

  /** Parses a org.antlr.v4.runtime.CharStream and returns a parsed [[Circuit]] */
  def parseCharStream(charStream: CharStream, infoMode: InfoMode): Circuit = {
    val (parseTimeMillis, cst) = time {
      val parser = {
        val lexer = new FIRRTLLexer(charStream)
        new FIRRTLParser(new CommonTokenStream(lexer))
      }

      parser.getInterpreter.setPredictionMode(PredictionMode.SLL)

      // Concrete Syntax Tree
      val cst = parser.circuit

      val numSyntaxErrors = parser.getNumberOfSyntaxErrors
      if (numSyntaxErrors > 0) throw new SyntaxErrorsException(s"$numSyntaxErrors syntax error(s) detected")
      cst
    }

    val visitor = new Visitor(infoMode)
    val (visitTimeMillis, visit) = time {
      visitor.visit(cst)
    }
    val ast = visit match {
      case c: Circuit => c
      case x => throw new ClassCastException("Error! AST not rooted with Circuit node!")
    }

    ast
  }
  /** Takes Iterator over lines of FIRRTL, returns FirrtlNode (root node is Circuit) */
  def parse(lines: Iterator[String], infoMode: InfoMode = UseInfo): Circuit =
    parseString(lines.mkString("\n"), infoMode)

  def parse(lines: Seq[String]): Circuit = parseString(lines.mkString("\n"), UseInfo)

  def parse(text: String): Circuit = parseString(text, UseInfo)

  sealed abstract class InfoMode

  case object IgnoreInfo extends InfoMode

  case object UseInfo extends InfoMode

  case class GenInfo(filename: String) extends InfoMode

  case class AppendInfo(filename: String) extends InfoMode

}
