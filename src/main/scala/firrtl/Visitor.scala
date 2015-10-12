/*
 * TODO FIXME
 *  - Support all integer types (not just "h...")
 *  - In ANTLR examples they use just visit, why am I having to use visitModule or other specific functions?
 *  - Make visit private?
 *  - More elegant way to insert UnknownWidth?
*/

package firrtl

import org.antlr.v4.runtime.tree.AbstractParseTreeVisitor;
import org.antlr.v4.runtime.ParserRuleContext
import org.antlr.v4.runtime.tree.ErrorNode
import org.antlr.v4.runtime.tree.TerminalNode
import scala.collection.JavaConversions._
import antlr._

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
  private def getFileInfo(ctx: ParserRuleContext): FileInfo = 
    FileInfo(filename, ctx.getStart().getLine(), ctx.getStart().getCharPositionInLine())

	private def visitCircuit[AST](ctx: FIRRTLParser.CircuitContext): Circuit = 
    Circuit(getFileInfo(ctx), ctx.id.getText, ctx.module.map(visitModule)) 
    
  private def visitModule[AST](ctx: FIRRTLParser.ModuleContext): Module = 
    Module(getFileInfo(ctx), ctx.id.getText, ctx.port.map(visitPort), visitBlockStmt(ctx.blockStmt))

	private def visitPort[AST](ctx: FIRRTLParser.PortContext): Port = 
    Port(getFileInfo(ctx), ctx.id.getText, visitPortKind(ctx.portKind), visitType(ctx.`type`))
  
	private def visitPortKind[AST](ctx: FIRRTLParser.PortKindContext): PortDir =
    ctx.getText match {
      case "input" => Input
      case "output" => Output
    }

  // Match on a type instead of on strings?
	private def visitType[AST](ctx: FIRRTLParser.TypeContext): Type = {
    ctx.getChild(0).getText match {
      case "UInt" => if (ctx.getChildCount > 1) UIntType( visitWidth(ctx.width) ) 
                     else UIntType( UnknownWidth )
      case "SInt" => if (ctx.getChildCount > 1) SIntType( visitWidth(ctx.width) )
                     else SIntType( UnknownWidth )
      case "Clock" => ClockType
      case "{" => BundleType(ctx.field.map(visitField))
      case _ => new VectorType( visitType(ctx.`type`), string2BigInt(ctx.IntLit.getText) )
    }
  }
      
	private def visitField[AST](ctx: FIRRTLParser.FieldContext): Field = 
    Field(ctx.id.getText, visitOrientation(ctx.orientation), visitType(ctx.`type`))
      
	private def visitOrientation[AST](ctx: FIRRTLParser.OrientationContext): FieldDir = 
    ctx.getText match {
      case "flip" => Reverse
      case _ => Default
    }

	private def visitWidth[AST](ctx: FIRRTLParser.WidthContext): Width = {
    val text = ctx.getText
    text match {
      case "?" => UnknownWidth
      case _   => IntWidth(string2BigInt(text))
    }
  }
	
	private def visitBlockStmt[AST](ctx: FIRRTLParser.BlockStmtContext): Stmt = 
    Block(ctx.stmt.map(visitStmt)) 

	private def visitStmt[AST](ctx: FIRRTLParser.StmtContext): Stmt = {
    val info = getFileInfo(ctx)

    ctx.getChild(0).getText match {
      case "wire" => DefWire(info, ctx.id(0).getText, visitType(ctx.`type`))
      case "reg"  => DefReg(info, ctx.id(0).getText, visitType(ctx.`type`),
                              visitExp(ctx.exp(0)), visitExp(ctx.exp(1)))
      case "smem"  => DefMemory(info, ctx.id(0).getText, true,
                              visitType(ctx.`type`), visitExp(ctx.exp(0)))
      case "cmem"  => DefMemory(info, ctx.id(0).getText, false,
                              visitType(ctx.`type`), visitExp(ctx.exp(0)))
      case "inst"  => DefInst(info, ctx.id(0).getText, Ref(ctx.id(1).getText, UnknownType))
      case "node" =>  DefNode(info, ctx.id(0).getText, visitExp(ctx.exp(0)))
      case "poison" => DefPoison(info, ctx.id(0).getText, visitType(ctx.`type`))
      case "onreset" => OnReset(info, visitExp(ctx.exp(0)), visitExp(ctx.exp(1)))
      case "when" => { 
        val alt = if (ctx.blockStmt.length > 1) visitBlockStmt(ctx.blockStmt(1)) else EmptyStmt
        When(info, visitExp(ctx.exp(0)), visitBlockStmt(ctx.blockStmt(0)), alt)
      }
      case "assert" => Assert(info, visitExp(ctx.exp(0)))
      case "skip" => EmptyStmt
      // If we don't match on the first child, try the next one
      case _ => {
        ctx.getChild(1).getText match {
          case "accessor" => DefAccessor(info, ctx.id(0).getText,
                              visitDir(ctx.dir), visitExp(ctx.exp(0)), visitExp(ctx.exp(1)))
          case ":=" => Connect(info, visitExp(ctx.exp(0)), visitExp(ctx.exp(1)) )
          case "<>" => BulkConnect(info, visitExp(ctx.exp(0)), visitExp(ctx.exp(1)) )
        }
      }
    }
  }
      
	private def visitDir[AST](ctx: FIRRTLParser.DirContext): AccessorDir = 
    ctx.getText match {
      case "infer" => Infer
      case "read" => Read
      case "write" => Write
      case "rdwr" => RdWr // TODO remove exceptions for unmatched things
    }

	private def visitExp[AST](ctx: FIRRTLParser.ExpContext): Exp = 
    if( ctx.getChildCount == 1 ) 
      Ref(ctx.getText, UnknownType)
    else
      ctx.getChild(0).getText match {
        case "UInt" => {
          val width = if (ctx.getChildCount > 4) visitWidth(ctx.width) else UnknownWidth
          UIntValue(string2BigInt(ctx.IntLit(0).getText), width)
        }
        //case "SInt" => SIntValue(string2BigInt(ctx.IntLit(0).getText), string2BigInt(ctx.width.getText))
        case "SInt" => {
          val width = if (ctx.getChildCount > 4) visitWidth(ctx.width) else UnknownWidth
          SIntValue(string2BigInt(ctx.IntLit(0).getText), width)
        }
        case _ => 
          ctx.getChild(1).getText match {
            case "." => new Subfield(visitExp(ctx.exp(0)), ctx.id.getText, UnknownType)
            case "[" => new Subindex(visitExp(ctx.exp(0)), string2BigInt(ctx.IntLit(0).getText))
            case "(" => 
              DoPrimOp(visitPrimop(ctx.primop), ctx.exp.map(visitExp), ctx.IntLit.map(x => string2BigInt(x.getText)))
          }
      }
   
   // TODO can I create this and have the opposite? create map and invert it?
	private def visitPrimop[AST](ctx: FIRRTLParser.PrimopContext): PrimOp = 
    ctx.getText match {
      case "add" => Add
      case "sub" => Sub
      case "addw" => Addw
      case "subw" => Subw
      case "mul" => Mul
      case "div" => Div
      case "mod" => Mod
      case "quo" => Quo
      case "rem" => Rem
      case "lt" => Lt
      case "leq" => Leq
      case "gt" => Gt
      case "geq" => Geq
      case "eq" => Eq
      case "neq" => Neq
      case "mux" => Mux
      case "pad" => Pad
      case "asUInt" => AsUInt
      case "asSInt" => AsSInt
      case "shl" => Shl
      case "shr" => Shr
      case "dshl" => Dshl
      case "dshr" => Dshr
      case "cvt" => Cvt
      case "neg" => Neg
      case "not" => Not
      case "and" => And
      case "or" => Or
      case "xor" => Xor
      case "andr" => Andr
      case "orr" => Orr
      case "xorr" => Xorr
      case "cat" => Cat
      case "bit" => Bit
      case "bits" => Bits
    }
}
