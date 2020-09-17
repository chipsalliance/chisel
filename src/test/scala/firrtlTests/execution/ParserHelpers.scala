// SPDX-License-Identifier: Apache-2.0

package firrtlTests.execution

import firrtl._
import firrtl.ir._

class ParserHelperException(val pe: ParserException, input: String)
    extends FirrtlUserException(s"Got error ${pe.toString} while parsing input:\n${input}")

/**
  * A utility class that parses a FIRRTL string representing a statement to a sub-AST
  */
object ParseStatement {
  private def wrapStmtStr(stmtStr: String): String = {
    val indent = "    "
    val indented = stmtStr.split("\n").mkString(indent, s"\n${indent}", "")
    s"""circuit ${DUTRules.dutName} :
       |  module ${DUTRules.dutName} :
       |    input clock : Clock
       |    input reset : UInt<1>
       |${indented}""".stripMargin
  }

  private def parse(stmtStr: String): Circuit = {
    try {
      Parser.parseString(wrapStmtStr(stmtStr), Parser.IgnoreInfo)
    } catch {
      case e: ParserException => throw new ParserHelperException(e, stmtStr)
    }
  }

  def apply(stmtStr: String): Statement = {
    val c = parse(stmtStr)
    val stmt = c.modules.collectFirst { case Module(_, _, _, b: Block) => b.stmts.head }
    stmt.get
  }

  private[execution] def makeDUT(body: String): Circuit = parse(body)
}

/**
  * A utility class that parses a FIRRTL string representing an expression to a sub-AST
  */
object ParseExpression {
  def apply(expStr: String): Expression = {
    try {
      val s = ParseStatement(s"${expStr} is invalid")
      s.asInstanceOf[IsInvalid].expr
    } catch {
      case e: ParserHelperException => throw new ParserHelperException(e.pe, expStr)
    }
  }
}
