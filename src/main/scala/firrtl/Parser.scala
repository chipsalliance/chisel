package firrtl

import org.antlr.v4.runtime._;
import org.antlr.v4.runtime.atn._;
import org.antlr.v4.runtime.tree._;
import java.io.FileInputStream
import scala.collection.JavaConverters._
import scala.io.Source
import Utils._
import antlr._

object Parser
{
  /** Takes Iterator over lines of FIRRTL, returns AST (root node is Circuit)
    *
    * Parser performs conversion to machine firrtl
    */
  def parse(filename: String, lines: Iterator[String]): Circuit = {
    val fixedInput = Translator.addBrackets(lines)
    val antlrStream = new ANTLRInputStream(fixedInput.result)
    val lexer = new FIRRTLLexer(antlrStream)
    val tokens = new CommonTokenStream(lexer)
    val parser = new FIRRTLParser(tokens)

    // FIXME Dangerous (TODO)
    parser.getInterpreter.setPredictionMode(PredictionMode.SLL)

    // Concrete Syntax Tree
    val cst = parser.circuit

    val visitor = new Visitor(filename) 
    //val ast = visitor.visitCircuit(cst) match {
    val ast = visitor.visit(cst) match {
      case c: Circuit => c
      case x => throw new ClassCastException("Error! AST not rooted with Circuit node!")
    }

    ast
  }

  def parse(lines: Seq[String]): Circuit = parse("<None>", lines.iterator)

}
