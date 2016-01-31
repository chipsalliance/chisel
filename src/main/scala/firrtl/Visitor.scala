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

class Visitor(val fullFilename: String) extends FIRRTLBaseVisitor[AST] 
{
  // Strip file path
  private val filename = fullFilename.drop(fullFilename.lastIndexOf("/")+1)

  // For some reason visitCircuit does not visit the right function 
  // FIXME for some reason this cannot be private, probably because it extends
  //   FIRRTLBaseVisitor which is in a subpackage?
  def visit[AST](ctx: FIRRTLParser.CircuitContext): Circuit = visitCircuit(ctx)

  //  These regex have to change if grammar changes
  private def string2BigInt(s: String): BigInt = {
    // private define legal patterns
    val HexPattern = """\"*h([a-zA-Z0-9]+)\"*""".r
    val DecPattern = """(\+|-)?([1-9]\d*)""".r
    val ZeroPattern = "0".r
    s match {
      case ZeroPattern(_*) => BigInt(0)
      case HexPattern(hexdigits) => BigInt(hexdigits, 16)
      case DecPattern(sign, num) => BigInt(num)
      case  _  => throw new Exception("Invalid String for conversion to BigInt " + s)
    }
  }
  private def string2Int(s: String): Int = string2BigInt(s).toInt
  private def getInfo(ctx: ParserRuleContext): Info = 
    FileInfo(filename, ctx.getStart().getLine(), ctx.getStart().getCharPositionInLine())

	private def visitCircuit[AST](ctx: FIRRTLParser.CircuitContext): Circuit = 
    Circuit(getInfo(ctx), ctx.module.map(visitModule), (ctx.id.getText)) 
    
  private def visitModule[AST](ctx: FIRRTLParser.ModuleContext): Module = 
    InModule(getInfo(ctx), (ctx.id.getText), ctx.port.map(visitPort), visitBlock(ctx.block))

	private def visitPort[AST](ctx: FIRRTLParser.PortContext): Port = 
    Port(getInfo(ctx), (ctx.id.getText), visitDir(ctx.dir), visitType(ctx.`type`))

	private def visitDir[AST](ctx: FIRRTLParser.DirContext): Direction =
    ctx.getText match {
      case "input" => INPUT
      case "output" => OUTPUT
    }

  // Match on a type instead of on strings?
	private def visitType[AST](ctx: FIRRTLParser.TypeContext): Type = {
    ctx.getChild(0).getText match {
      case "UInt" => if (ctx.getChildCount > 1) UIntType(IntWidth(string2BigInt(ctx.IntLit.getText))) 
                     else UIntType( UnknownWidth() )
      case "SInt" => if (ctx.getChildCount > 1) SIntType(IntWidth(string2BigInt(ctx.IntLit.getText))) 
                     else SIntType( UnknownWidth() )
      case "Clock" => ClockType()
      case "{" => BundleType(ctx.field.map(visitField))
      case _ => new VectorType( visitType(ctx.`type`), string2Int(ctx.IntLit.getText) )
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
            if (map.contains(field)) throw new Exception(s"Redefinition of mem field ${field}")
            else map + (field -> Seq(children(2)))
          }
        parseChildren(children.drop(3), newMap) // We consume tokens in groups of three (eg. 'depth' '=>' 5)
      }
    }

    val info = getInfo(ctx)
    // Build map of different Memory fields to their values
    val map = try {
      parseChildren(ctx.children.drop(4), Map[String, Seq[ParseTree]]()) // First 4 tokens are 'mem' id ':' '{', skip to fields
    } catch {
      case e: Exception => throw new Exception(s"[${info}] ${e.getMessage}") // attach line number
    }
    // Each memory field value has been left as ParseTree type, need to convert
    // TODO Improve? Remove dynamic typecast of data-type
    DefMemory(info, (ctx.id(0).getText), visitType(map("data-type").head.asInstanceOf[FIRRTLParser.TypeContext]), 
              string2Int(map("depth").head.getText), string2Int(map("write-latency").head.getText), 
              string2Int(map("read-latency").head.getText), map.getOrElse("reader", Seq()).map(x => (x.getText)),
              map.getOrElse("writer", Seq()).map(x => (x.getText)), map.getOrElse("readwriter", Seq()).map(x => (x.getText)))
  }

  // visitStmt
	private def visitStmt[AST](ctx: FIRRTLParser.StmtContext): Stmt = {
    val info = getInfo(ctx)

    ctx.getChild(0).getText match {
      case "wire" => DefWire(info, (ctx.id(0).getText), visitType(ctx.`type`(0)))
      case "reg"  => {
        val name = (ctx.id(0).getText)
        val tpe = visitType(ctx.`type`(0))
        val reset = if (ctx.exp(1) != null) visitExp(ctx.exp(1)) else UIntValue(0, IntWidth(1))
        val init  = if (ctx.exp(2) != null) visitExp(ctx.exp(2)) else Ref(name, tpe)
        DefRegister(info, name, tpe, visitExp(ctx.exp(0)), reset, init)
      }
      case "mem" => visitMem(ctx)
      case "inst"  => DefInstance(info, (ctx.id(0).getText), (ctx.id(1).getText))
      case "node" =>  DefNode(info, (ctx.id(0).getText), visitExp(ctx.exp(0)))
      case "when" => { 
        val alt = if (ctx.block.length > 1) visitBlock(ctx.block(1)) else Empty()
        Conditionally(info, visitExp(ctx.exp(0)), visitBlock(ctx.block(0)), alt)
      }
      case "stop(" => Stop(info, string2Int(ctx.IntLit(0).getText), visitExp(ctx.exp(0)), visitExp(ctx.exp(1)))
      case "printf(" => Print(info, ctx.StringLit.getText, ctx.exp.drop(2).map(visitExp), 
                              visitExp(ctx.exp(0)), visitExp(ctx.exp(1)))
      case "skip" => Empty()
      // If we don't match on the first child, try the next one
      case _ => {
        ctx.getChild(1).getText match {
          case "<=" => Connect(info, visitExp(ctx.exp(0)), visitExp(ctx.exp(1)) )
          case "<-" => BulkConnect(info, visitExp(ctx.exp(0)), visitExp(ctx.exp(1)) )
          case "is" => IsInvalid(info, visitExp(ctx.exp(0)))
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
      Ref((ctx.getText), UnknownType())
    else
      ctx.getChild(0).getText match {
        case "UInt" => { // This could be better
          val (width, value) = 
            if (ctx.getChildCount > 4) 
              (IntWidth(string2BigInt(ctx.IntLit(0).getText)), string2BigInt(ctx.IntLit(1).getText))
            else {
               val bigint = string2BigInt(ctx.IntLit(0).getText)
               (IntWidth(BigInt(bigint.bitLength)),bigint)
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
        case "validif(" => ValidIf(visitExp(ctx.exp(0)), visitExp(ctx.exp(1)), UnknownType())
        case "mux(" => Mux(visitExp(ctx.exp(0)), visitExp(ctx.exp(1)), visitExp(ctx.exp(2)), UnknownType())
        case _ => 
          ctx.getChild(1).getText match {
            case "." => new SubField(visitExp(ctx.exp(0)), (ctx.id.getText), UnknownType())
            case "[" => if (ctx.exp(1) == null)  
                          new SubIndex(visitExp(ctx.exp(0)), string2Int(ctx.IntLit(0).getText), UnknownType())
                        else new SubAccess(visitExp(ctx.exp(0)), visitExp(ctx.exp(1)), UnknownType())
            // Assume primop
            case _ => DoPrim(visitPrimop(ctx.primop), ctx.exp.map(visitExp),
                             ctx.IntLit.map(x => string2BigInt(x.getText)), UnknownType())
          }
      }
  
  // stripSuffix("(") is included because in ANTLR concrete syntax we have to include open parentheses, 
  //  see grammar file for more details
	private def visitPrimop[AST](ctx: FIRRTLParser.PrimopContext): PrimOp = fromString(ctx.getText.stripSuffix("("))

  // visit Id and Keyword?
}
