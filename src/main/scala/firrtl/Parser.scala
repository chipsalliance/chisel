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
  def parseModule(string: String): Module = {
    val fixedInput = Translator.addBrackets(Iterator(string))
    val antlrStream = new ANTLRInputStream(fixedInput.result)
    val lexer = new FIRRTLLexer(antlrStream)
    val tokens = new CommonTokenStream(lexer)
    val parser = new FIRRTLParser(tokens)

    // FIXME Dangerous
    parser.getInterpreter.setPredictionMode(PredictionMode.SLL)

    // Concrete Syntax Tree
    val cst = parser.module

    val visitor = new Visitor("none") 
    //val ast = visitor.visitCircuit(cst) match {
    val ast = visitor.visit(cst) match {
      case m: Module => m
      case x => throw new ClassCastException("Error! AST not rooted with Module node!")
    }

    ast

  }

  /** Takes a firrtl filename, returns AST (root node is Circuit)
    *
    * Currently must be standard FIRRTL file
    * Parser performs conversion to machine firrtl
    */
  def parse(filename: String): Circuit = {
    //val antlrStream = new ANTLRInputStream(input.reader) 
    val fixedInput = Translator.addBrackets(Source.fromFile(filename).getLines)
    val antlrStream = new ANTLRInputStream(fixedInput.result)
    val lexer = new FIRRTLLexer(antlrStream)
    val tokens = new CommonTokenStream(lexer)
    val parser = new FIRRTLParser(tokens)

    // FIXME Dangerous
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

}
