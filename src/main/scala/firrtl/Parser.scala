/*
Copyright (c) 2014 - 2016 The Regents of the University of
California (Regents). All Rights Reserved.  Redistribution and use in
source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
   * Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer in the documentation and/or other materials
     provided with the distribution.
   * Neither the name of the Regents nor the names of its contributors
     may be used to endorse or promote products derived from this
     software without specific prior written permission.
IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS.
*/
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

    val cst = time("ANTLR Parser") {
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
    val ast = time("Visitor") {
      visitor.visit(cst)
    } match {
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
