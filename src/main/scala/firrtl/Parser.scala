// SPDX-License-Identifier: Apache-2.0

package firrtl

import org.antlr.v4.runtime._
import org.antlr.v4.runtime.atn._
import logger.LazyLogging
import firrtl.ir._
import firrtl.parser.Listener
import firrtl.Utils.time
import firrtl.antlr.{FIRRTLParser, _}

import scala.util.control.NonFatal

class ParserException(message: String) extends FirrtlUserException(message)

case class ParameterNotSpecifiedException(message: String) extends ParserException(message)
case class ParameterRedefinedException(message: String) extends ParserException(message)
case class InvalidStringLitException(message: String) extends ParserException(message)
case class InvalidEscapeCharException(message: String) extends ParserException(message)
case class SyntaxErrorsException(message: String) extends ParserException(message)
case class UnsupportedVersionException(message: String) extends ParserException(message)

object Parser extends LazyLogging {

  /** Parses a file in a given filename and returns a parsed [[firrtl.ir.Circuit Circuit]] */
  def parseFile(filename: String, infoMode: InfoMode): Circuit =
    parseCharStream(CharStreams.fromFileName(filename), infoMode)

  /** Parses a String and returns a parsed [[firrtl.ir.Circuit Circuit]] */
  def parseString(text: String, infoMode: InfoMode): Circuit =
    parseCharStream(CharStreams.fromString(text), infoMode)

  /** Parses a org.antlr.v4.runtime.CharStream and returns a parsed [[firrtl.ir.Circuit Circuit]] */
  def parseCharStream(charStream: CharStream, infoMode: InfoMode): Circuit = {
    val (parseTimeMillis, ast) = time {
      val parser = {
        val lexer = new FIRRTLLexer(charStream)
        new FIRRTLParser(new CommonTokenStream(lexer))
      }

      val listener = new Listener(infoMode)

      parser.getInterpreter.setPredictionMode(PredictionMode.SLL)
      parser.addParseListener(listener)

      // Syntax errors may violate assumptions in the Listener and Visitor.
      // We need to handle these errors gracefully.
      val throwable =
        try {
          parser.circuit
          None
        } catch {
          case e: ParserException => throw e
          case NonFatal(e) => Some(e)
        }

      val numSyntaxErrors = parser.getNumberOfSyntaxErrors
      if (numSyntaxErrors > 0) throw new SyntaxErrorsException(s"$numSyntaxErrors syntax error(s) detected")

      // Note that this should never happen because any throwables caught should be due to syntax
      // errors that are reported above. This is just to ensure that we don't accidentally mask any
      // bugs in the Parser, Listener, or Visitor.
      if (throwable.nonEmpty) Utils.throwInternalError(exception = throwable)

      listener.getCircuit
    }

    ast
  }

  /** Takes Iterator over lines of FIRRTL, returns FirrtlNode (root node is Circuit) */
  def parse(lines: Iterator[String], infoMode: InfoMode = UseInfo): Circuit =
    parseString(lines.mkString("\n"), infoMode)

  def parse(lines: Seq[String]): Circuit = parseString(lines.mkString("\n"), UseInfo)

  /** Parse the concrete syntax of a FIRRTL [[firrtl.ir.Circuit]], e.g.
    *   {{{
    *     """circuit Top:
    *       |  module Top:
    *       |    input x: UInt
    *       |    node y = x
    *       |""".stripMargin
    *   }}}
    *   becomes:
    *   {{{
    *     Circuit(
    *       NoInfo,
    *       Seq(Module(
    *         NoInfo,
    *         "Top",
    *         Seq(Port(NoInfo, "x", Input, UIntType(UnknownWidth))),
    *         Block(DefNode(NoInfo, "y", Ref("x", UnknownType)))
    *       )),
    *       "Top"
    *     )
    *   }}}
    * @param text concrete Circuit syntax
    * @return
    */
  def parse(text: String): Circuit = parseString(text, UseInfo)

  /** Parse the concrete syntax of a FIRRTL [[firrtl.ir.Type]], e.g.
    *   "UInt<3>" becomes:
    *   {{{
    *     UIntType(IntWidth(BigInt(3)))
    *   }}}
    * @param tpe concrete Type syntax
    * @return
    */
  def parseType(tpe: String): Type = {
    val input = Seq("circuit Top:\n", "  module Top:\n", s"    input x:$tpe\n")
    val circuit = parse(input)
    circuit.modules.head.ports.head.tpe
  }

  def parse(lines: Seq[String], infoMode: InfoMode): Circuit = parse(lines.iterator, infoMode)

  def parse(text: String, infoMode: InfoMode): Circuit = parse(text.split("\n"), infoMode)

  /** Parse the concrete syntax of a FIRRTL [[firrtl.ir.Expression]], e.g.
    *   "add(x, y)" becomes:
    *   {{{
    *     DoPrim(Add, Seq(Ref("x", UnknownType), Ref("y", UnknownType), Nil, UnknownType)
    *   }}}
    * @param expr concrete Expression syntax
    * @return
    */
  def parseExpression(expr: String): Expression = {
    val input = Seq("circuit Top:\n", "  module Top:\n", s"    node x = $expr\n")
    val circuit = parse(input)
    circuit.modules match {
      case Seq(Module(_, _, _, Block(Seq(DefNode(_, _, value))))) => value
    }
  }

  /** Parse the concrete syntax of a FIRRTL [[firrtl.ir.Statement]], e.g.
    *   "wire x: UInt" becomes:
    *   {{{
    *     DefWire(NoInfo, "x", UIntType(UnknownWidth))
    *   }}}
    * @param statement concrete Statement syntax
    * @return
    */
  def parseStatement(statement: String): Statement = {
    val input = Seq("circuit Top:\n", "  module Top:\n") ++ statement.split("\n").map("    " + _)
    val circuit = parse(input)
    circuit.modules.head.asInstanceOf[Module].body
  }

  /** Parse the concrete syntax of a FIRRTL [[firrtl.ir.Port]], e.g.
    *   "input x: UInt" becomes:
    *   {{{
    *     Port(NoInfo, "x", Input, UIntType(UnknownWidth))
    *   }}}
    * @param port concrete Port syntax
    * @return
    */
  def parsePort(port: String): Port = {
    val input = Seq("circuit Top:\n", "  module Top:\n", s"    $port\n")
    val circuit = parse(input)
    circuit.modules.head.ports.head
  }

  /** Parse the concrete syntax of a FIRRTL [[firrtl.ir.DefModule]], e.g.
    *   {{{
    *     """module Top:
    *       |  input x: UInt
    *       |  node y = x
    *       |""".stripMargin
    *   }}}
    *   becomes:
    *   {{{
    *     Module(
    *       NoInfo,
    *       "Top",
    *       Seq(Port(NoInfo, "x", Input, UIntType(UnknownWidth))),
    *       Block(DefNode(NoInfo, "y", Ref("x", UnknownType)))
    *     )
    *   }}}
    * @param module concrete DefModule syntax
    * @return
    */
  def parseDefModule(module: String): DefModule = {
    val input = Seq("circuit Top:\n") ++ module.split("\n").map("  " + _)
    val circuit = parse(input)
    circuit.modules.head
  }

  /** Parse the concrete syntax of a FIRRTL [[firrtl.ir.Info]], e.g.
    *   "@[FPU.scala 509:25]" becomes:
    *   {{{
    *     FileInfo("FPU.scala 509:25")
    *   }}}
    * @param info concrete Info syntax
    * @return
    */
  def parseInfo(info: String): Info = {
    val input = Seq(s"circuit Top: $info\n", "  module Top:\n", "    input x: UInt\n")
    val circuit = parse(input)
    circuit.info
  }

  sealed abstract class InfoMode

  case object IgnoreInfo extends InfoMode

  case object UseInfo extends InfoMode

  case class GenInfo(filename: String) extends InfoMode

  case class AppendInfo(filename: String) extends InfoMode

}
