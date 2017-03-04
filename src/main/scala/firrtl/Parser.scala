// See LICENSE for license details.

package firrtl

import java.io.{ByteArrayInputStream, SequenceInputStream}

import org.antlr.v4.runtime._
import org.antlr.v4.runtime.atn._
import com.typesafe.scalalogging.LazyLogging
import firrtl.ir._
import firrtl.Utils.time
import firrtl.antlr.{FIRRTLParser, _}

class ParserException(message: String) extends Exception(message)

case class ParameterNotSpecifiedException(message: String) extends ParserException(message)

case class ParameterRedefinedException(message: String) extends ParserException(message)

case class InvalidStringLitException(message: String) extends ParserException(message)

case class InvalidEscapeCharException(message: String) extends ParserException(message)


object Parser extends LazyLogging {
  /** Takes Iterator over lines of FIRRTL, returns FirrtlNode (root node is Circuit) */
  def parse(lines: Iterator[String], infoMode: InfoMode = UseInfo): Circuit = {

    val (parseTimeMillis, cst) = time {
      val parser = {
        import scala.collection.JavaConverters._
        val inStream = new SequenceInputStream(
          lines.map{s => new ByteArrayInputStream((s + "\n").getBytes("UTF-8")) }.asJavaEnumeration
        )
        val lexer = new FIRRTLLexer(new ANTLRInputStream(inStream))
        new FIRRTLParser(new CommonTokenStream(lexer))
      }

      parser.getInterpreter.setPredictionMode(PredictionMode.SLL)

      // Concrete Syntax Tree
      val cst = parser.circuit

      val numSyntaxErrors = parser.getNumberOfSyntaxErrors
      if (numSyntaxErrors > 0) throw new ParserException(s"$numSyntaxErrors syntax error(s) detected")
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

  def parse(lines: Seq[String]): Circuit = parse(lines.iterator)

  def parse(text: String): Circuit = parse(text split "\n")

  sealed abstract class InfoMode

  case object IgnoreInfo extends InfoMode

  case object UseInfo extends InfoMode

  case class GenInfo(filename: String) extends InfoMode

  case class AppendInfo(filename: String) extends InfoMode

}
