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
/*
 * TODO
 *  - Implement UBits and SBits
 *  - Get correct line number for memory field errors
*/

package firrtl

import org.antlr.v4.runtime.tree.AbstractParseTreeVisitor;
import org.antlr.v4.runtime.ParserRuleContext
import org.antlr.v4.runtime.tree.ParseTree
import org.antlr.v4.runtime.tree.ErrorNode
import org.antlr.v4.runtime.tree.TerminalNode
import scala.collection.JavaConversions._
import antlr._
import PrimOps._
import FIRRTLParser._
import Parser.{InfoMode, IgnoreInfo, UseInfo, GenInfo, AppendInfo}
import scala.annotation.tailrec

class Visitor(infoMode: InfoMode) extends FIRRTLBaseVisitor[AST]
{
  // Strip file path
  private def stripPath(filename: String) = filename.drop(filename.lastIndexOf("/")+1)

  def visit[AST](ctx: FIRRTLParser.CircuitContext): Circuit = visitCircuit(ctx)

  //  These regex have to change if grammar changes
  private def string2BigInt(s: String): BigInt = {
    // private define legal patterns
    val HexPattern = """\"*h([a-zA-Z0-9]+)\"*""".r
    val DecPattern = """(\+|-)?([1-9]\d*)""".r
    val ZeroPattern = "0".r
    val NegPattern = "(89AaBbCcDdEeFf)".r
    s match {
      case ZeroPattern(_*) => BigInt(0)
      case HexPattern(hexdigits) => 
         hexdigits(0) match {
            case NegPattern(_) =>{
               BigInt("-" + hexdigits,16)
            }
            case _ => BigInt(hexdigits, 16)
         }
      case DecPattern(sign, num) => {
         if (sign != null) BigInt(sign + num,10)
         else BigInt(num,10)
      }
      case  _  => throw new Exception("Invalid String for conversion to BigInt " + s)
    }
  }
  private def string2Int(s: String): Int = string2BigInt(s).toInt

  private def visitInfo(ctx: Option[FIRRTLParser.InfoContext], parentCtx: ParserRuleContext): Info = {
    def genInfo(filename: String): String =
      stripPath(filename) + "@" + parentCtx.getStart.getLine + "." +
      parentCtx.getStart.getCharPositionInLine
    lazy val useInfo: String = ctx match {
      case Some(info) => info.getText.drop(2).init // remove surrounding @[ ... ]
      case None => ""
    }
    infoMode match {
      case UseInfo =>
        if (useInfo.length == 0) NoInfo else FileInfo(FIRRTLStringLitHandler.unescape(useInfo))
      case AppendInfo(filename) =>
        val newInfo = useInfo + ":" + genInfo(filename)
        FileInfo(FIRRTLStringLitHandler.unescape(newInfo))
      case GenInfo(filename) => FileInfo(FIRRTLStringLitHandler.unescape(genInfo(filename)))
      case IgnoreInfo => NoInfo
    }
  }

	private def visitCircuit[AST](ctx: FIRRTLParser.CircuitContext): Circuit =
    Circuit(visitInfo(Option(ctx.info), ctx), ctx.module.map(visitModule), (ctx.id.getText))
    
  private def visitModule[AST](ctx: FIRRTLParser.ModuleContext): DefModule = {
    val info = visitInfo(Option(ctx.info), ctx)
    ctx.getChild(0).getText match {
      case "module" => Module(info, ctx.id.getText, ctx.port.map(visitPort), visitBlock(ctx.block))
      case "extmodule" => ExtModule(info, ctx.id.getText, ctx.port.map(visitPort))
    }
  }

  private def visitPort[AST](ctx: FIRRTLParser.PortContext): Port = {
    Port(visitInfo(Option(ctx.info), ctx), (ctx.id.getText), visitDir(ctx.dir), visitType(ctx.`type`))
  }
  private def visitDir[AST](ctx: FIRRTLParser.DirContext): Direction =
    ctx.getText match {
      case "input" => Input
      case "output" => Output
    }
  private def visitMdir[AST](ctx: FIRRTLParser.MdirContext): MPortDir =
    ctx.getText match {
      case "infer" => MInfer
      case "read" => MRead
      case "write" => MWrite
      case "rdwr" => MReadWrite
    }

  // Match on a type instead of on strings?
  private def visitType[AST](ctx: FIRRTLParser.TypeContext): Type = {
    ctx.getChild(0) match {
      case term: TerminalNode => 
        term.getText match {
          case "UInt" => if (ctx.getChildCount > 1) UIntType(IntWidth(string2BigInt(ctx.IntLit.getText))) 
                         else UIntType( UnknownWidth() )
          case "SInt" => if (ctx.getChildCount > 1) SIntType(IntWidth(string2BigInt(ctx.IntLit.getText))) 
                         else SIntType( UnknownWidth() )
          case "Clock" => ClockType
          case "{" => BundleType(ctx.field.map(visitField))
        }
      case tpe: TypeContext => new VectorType(visitType(ctx.`type`), string2Int(ctx.IntLit.getText))
    }
  }
      
	private def visitField[AST](ctx: FIRRTLParser.FieldContext): Field = {
    val flip = if(ctx.getChild(0).getText == "flip") REVERSE else DEFAULT
    Field((ctx.id.getText), flip, visitType(ctx.`type`))
  }
     

  // visitBlock
	private def visitBlock[AST](ctx: FIRRTLParser.BlockContext): Stmt = 
    Begin(ctx.stmt.map(visitStmt)) 

  // Memories are fairly complicated to translate thus have a dedicated method
  private def visitMem[AST](ctx: FIRRTLParser.StmtContext): Stmt = {
    def parseChildren(children: Seq[ParseTree], map: Map[String, Seq[ParseTree]]): Map[String, Seq[ParseTree]] = {
      val field = children(0).getText
      if (field == "}") map
      else {
        val newMap = 
          if (field == "reader" || field == "writer" || field == "readwriter") {
            val seq = map getOrElse (field, Seq())
            map + (field -> (seq :+ children(2)))
          } else { // data-type, depth, read-latency, write-latency, read-under-write
            if (map.contains(field)) throw new ParameterRedefinedException(s"Redefinition of ${field}")
            else map + (field -> Seq(children(2)))
          }
        parseChildren(children.drop(3), newMap) // We consume tokens in groups of three (eg. 'depth' '=>' 5)
      }
    }

    val info = visitInfo(Option(ctx.info), ctx)
    // Build map of different Memory fields to their values
    val map = try {
      parseChildren(ctx.children.drop(4), Map[String, Seq[ParseTree]]()) // First 4 tokens are 'mem' id ':' '{', skip to fields
    } catch { // attach line number
      case e: ParameterRedefinedException => throw new ParameterRedefinedException(s"[${info}] ${e.message}") 
    }
    // Check for required fields
    Seq("data-type", "depth", "read-latency", "write-latency") foreach { field =>
      map.getOrElse(field, throw new ParameterNotSpecifiedException(s"[${info}] Required mem field ${field} not found"))
    }
    // Each memory field value has been left as ParseTree type, need to convert
    // TODO Improve? Remove dynamic typecast of data-type
    DefMemory(info, (ctx.id(0).getText), visitType(map("data-type").head.asInstanceOf[FIRRTLParser.TypeContext]), 
              string2Int(map("depth").head.getText), string2Int(map("write-latency").head.getText), 
              string2Int(map("read-latency").head.getText), map.getOrElse("reader", Seq()).map(x => (x.getText)),
              map.getOrElse("writer", Seq()).map(x => (x.getText)), map.getOrElse("readwriter", Seq()).map(x => (x.getText)))
  }

  // visitStringLit
  private def visitStringLit[AST](node: TerminalNode): StringLit = {
    val raw = node.getText.tail.init // Remove surrounding double quotes
    FIRRTLStringLitHandler.unescape(raw)
  }

  // visitStmt
	private def visitStmt[AST](ctx: FIRRTLParser.StmtContext): Stmt = {
    val info = visitInfo(Option(ctx.info), ctx)
    ctx.getChild(0) match {
      case term: TerminalNode => term.getText match {
        case "wire" => DefWire(info, (ctx.id(0).getText), visitType(ctx.`type`(0)))
        case "reg"  => {
          val name = (ctx.id(0).getText)
          val tpe = visitType(ctx.`type`(0))
          val reset = if (ctx.exp(1) != null) visitExp(ctx.exp(1)) else UIntValue(0, IntWidth(1))
          val init  = if (ctx.exp(2) != null) visitExp(ctx.exp(2)) else Ref(name, tpe)
          DefRegister(info, name, tpe, visitExp(ctx.exp(0)), reset, init)
        }
        case "mem" => visitMem(ctx)
        case "cmem" => {
           val t = visitType(ctx.`type`(0))
           t match {
              case (t:VectorType) => CDefMemory(info,ctx.id(0).getText,t.tpe,t.size,false)
              case _ => throw new ParserException(s"${info}: Must provide cmem with vector type")
           }
        }
        case "smem" => {
           val t = visitType(ctx.`type`(0))
           t match {
              case (t:VectorType) => CDefMemory(info,ctx.id(0).getText,t.tpe,t.size,true)
              case _ => throw new ParserException(s"${info}: Must provide cmem with vector type")
           }
        }
        case "inst"  => DefInstance(info, (ctx.id(0).getText), (ctx.id(1).getText))
        case "node" =>  DefNode(info, (ctx.id(0).getText), visitExp(ctx.exp(0)))
        case "when" => {
          val alt = if (ctx.block.length > 1) visitBlock(ctx.block(1)) else Empty()
          Conditionally(info, visitExp(ctx.exp(0)), visitBlock(ctx.block(0)), alt)
        }
        case "stop(" => Stop(info, string2Int(ctx.IntLit(0).getText), visitExp(ctx.exp(0)), visitExp(ctx.exp(1)))
        case "printf(" => Print(info, visitStringLit(ctx.StringLit), ctx.exp.drop(2).map(visitExp),
                                visitExp(ctx.exp(0)), visitExp(ctx.exp(1)))
        case "skip" => Empty()
      }
      // If we don't match on the first child, try the next one
      case _ => {
        ctx.getChild(1).getText match {
          case "<=" => Connect(info, visitExp(ctx.exp(0)), visitExp(ctx.exp(1)) )
          case "<-" => BulkConnect(info, visitExp(ctx.exp(0)), visitExp(ctx.exp(1)) )
          case "is" => IsInvalid(info, visitExp(ctx.exp(0)))
          case "mport" => CDefMPort(info, ctx.id(0).getText, UnknownType,ctx.id(1).getText,Seq(visitExp(ctx.exp(0)),visitExp(ctx.exp(1))),visitMdir(ctx.mdir))
        }
      }
    }
  }
     
  // add visitRuw ?
	//T visitRuw(FIRRTLParser.RuwContext ctx);
  //private def visitRuw[AST](ctx: FIRRTLParser.RuwContext): 

  // TODO 
  // - Add mux
  // - Add validif
	private def visitExp[AST](ctx: FIRRTLParser.ExpContext): Expression = 
    if( ctx.getChildCount == 1 ) 
      Ref((ctx.getText), UnknownType)
    else
      ctx.getChild(0).getText match {
        case "UInt" => { // This could be better
          val (width, value) = 
            if (ctx.getChildCount > 4) 
              (IntWidth(string2BigInt(ctx.IntLit(0).getText)), string2BigInt(ctx.IntLit(1).getText))
            else {
               val bigint = string2BigInt(ctx.IntLit(0).getText)
               (IntWidth(BigInt(scala.math.max(bigint.bitLength,1))),bigint)
            }
          UIntValue(value, width)
        }
        case "SInt" => {
          val (width, value) = 
            if (ctx.getChildCount > 4) 
              (IntWidth(string2BigInt(ctx.IntLit(0).getText)), string2BigInt(ctx.IntLit(1).getText))
            else {
               val bigint = string2BigInt(ctx.IntLit(0).getText)
               (IntWidth(BigInt(bigint.bitLength + 1)),bigint)
            }
          SIntValue(value, width)
        }
        case "validif(" => ValidIf(visitExp(ctx.exp(0)), visitExp(ctx.exp(1)), UnknownType)
        case "mux(" => Mux(visitExp(ctx.exp(0)), visitExp(ctx.exp(1)), visitExp(ctx.exp(2)), UnknownType)
        case _ => 
          ctx.getChild(1).getText match {
            case "." => new SubField(visitExp(ctx.exp(0)), (ctx.id.getText), UnknownType)
            case "[" => if (ctx.exp(1) == null)  
                          new SubIndex(visitExp(ctx.exp(0)), string2Int(ctx.IntLit(0).getText), UnknownType)
                        else new SubAccess(visitExp(ctx.exp(0)), visitExp(ctx.exp(1)), UnknownType)
            // Assume primop
            case _ => DoPrim(visitPrimop(ctx.primop), ctx.exp.map(visitExp),
                             ctx.IntLit.map(x => string2BigInt(x.getText)), UnknownType)
          }
      }
  
  // stripSuffix("(") is included because in ANTLR concrete syntax we have to include open parentheses, 
  //  see grammar file for more details
	private def visitPrimop[AST](ctx: FIRRTLParser.PrimopContext): PrimOp = fromString(ctx.getText.stripSuffix("("))

  // visit Id and Keyword?
}
