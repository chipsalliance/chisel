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

import org.antlr.v4.runtime.ParserRuleContext
import org.antlr.v4.runtime.tree.TerminalNode
import scala.collection.JavaConversions._
import scala.collection.mutable
import firrtl.antlr._
import PrimOps._
import FIRRTLParser._
import Parser.{AppendInfo, GenInfo, IgnoreInfo, InfoMode, UseInfo}
import firrtl.ir._


class Visitor(infoMode: InfoMode) extends FIRRTLBaseVisitor[FirrtlNode] {
  // Strip file path
  private def stripPath(filename: String) = filename.drop(filename.lastIndexOf("/") + 1)

  def visit[FirrtlNode](ctx: FIRRTLParser.CircuitContext): Circuit = visitCircuit(ctx)

  //  These regex have to change if grammar changes
  private def string2BigInt(s: String): BigInt = {
    // private define legal patterns
    val HexPattern =
      """\"*h([a-zA-Z0-9]+)\"*""".r
    val DecPattern = """(\+|-)?([1-9]\d*)""".r
    val ZeroPattern = "0".r
    val NegPattern = "(89AaBbCcDdEeFf)".r
    s match {
      case ZeroPattern(_*) => BigInt(0)
      case HexPattern(hexdigits) =>
        hexdigits(0) match {
          case NegPattern(_) =>
            BigInt("-" + hexdigits, 16)
          case _ => BigInt(hexdigits, 16)
        }
      case DecPattern(sign, num) =>
        if (sign != null) BigInt(sign + num, 10)
        else BigInt(num, 10)
      case _ => throw new Exception("Invalid String for conversion to BigInt " + s)
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
        if (useInfo.length == 0) NoInfo
        else ir.FileInfo(FIRRTLStringLitHandler.unescape(useInfo))
      case AppendInfo(filename) =>
        val newInfo = useInfo + ":" + genInfo(filename)
        ir.FileInfo(FIRRTLStringLitHandler.unescape(newInfo))
      case GenInfo(filename) =>
        ir.FileInfo(FIRRTLStringLitHandler.unescape(genInfo(filename)))
      case IgnoreInfo => NoInfo
    }
  }

  private def visitCircuit[FirrtlNode](ctx: FIRRTLParser.CircuitContext): Circuit =
    Circuit(visitInfo(Option(ctx.info), ctx), ctx.module.map(visitModule), ctx.id.getText)

  private def visitModule[FirrtlNode](ctx: FIRRTLParser.ModuleContext): DefModule = {
    val info = visitInfo(Option(ctx.info), ctx)
    ctx.getChild(0).getText match {
      case "module" => Module(info, ctx.id.getText, ctx.port.map(visitPort),
        if (ctx.moduleBlock() != null)
          visitBlock(ctx.moduleBlock())
        else EmptyStmt)
      case "extmodule" => ExtModule(info, ctx.id.getText, ctx.port.map(visitPort))
    }
  }

  private def visitPort[FirrtlNode](ctx: FIRRTLParser.PortContext): Port = {
    Port(visitInfo(Option(ctx.info), ctx), ctx.id.getText, visitDir(ctx.dir), visitType(ctx.`type`))
  }

  private def visitDir[FirrtlNode](ctx: FIRRTLParser.DirContext): Direction =
    ctx.getText match {
      case "input" => Input
      case "output" => Output
    }

  private def visitMdir[FirrtlNode](ctx: FIRRTLParser.MdirContext): MPortDir =
    ctx.getText match {
      case "infer" => MInfer
      case "read" => MRead
      case "write" => MWrite
      case "rdwr" => MReadWrite
    }

  // Match on a type instead of on strings?
  private def visitType[FirrtlNode](ctx: FIRRTLParser.TypeContext): Type = {
    def getWidth(n: TerminalNode): Width = IntWidth(string2BigInt(n.getText))
    ctx.getChild(0) match {
      case term: TerminalNode =>
        term.getText match {
          case "UInt" => if (ctx.getChildCount > 1) UIntType(IntWidth(string2BigInt(ctx.IntLit(0).getText)))
          else UIntType(UnknownWidth)
          case "SInt" => if (ctx.getChildCount > 1) SIntType(IntWidth(string2BigInt(ctx.IntLit(0).getText)))
          else SIntType(UnknownWidth)
          case "Fixed" => ctx.IntLit.size match {
            case 0 => FixedType(UnknownWidth, UnknownWidth)
            case 1 => ctx.getChild(2).getText match {
              case "<" => FixedType(UnknownWidth, getWidth(ctx.IntLit(0)))
              case _ => FixedType(getWidth(ctx.IntLit(0)), UnknownWidth)
            }
            case 2 => FixedType(getWidth(ctx.IntLit(0)), getWidth(ctx.IntLit(1)))
          }
          case "Clock" => ClockType
          case "Analog" => if (ctx.getChildCount > 1) AnalogType(IntWidth(string2BigInt(ctx.IntLit(0).getText)))
          else AnalogType(UnknownWidth)
          case "{" => BundleType(ctx.field.map(visitField))
        }
      case typeContext: TypeContext => new VectorType(visitType(ctx.`type`), string2Int(ctx.IntLit(0).getText))
    }
  }

  private def visitField[FirrtlNode](ctx: FIRRTLParser.FieldContext): Field = {
    val flip = if (ctx.getChild(0).getText == "flip") Flip else Default
    Field(ctx.id.getText, flip, visitType(ctx.`type`))
  }

  private def visitBlock[FirrtlNode](ctx: FIRRTLParser.ModuleBlockContext): Statement =
    Block(ctx.simple_stmt().map(_.stmt).filter(_ != null).map(visitStmt))

  private def visitSuite[FirrtlNode](ctx: FIRRTLParser.SuiteContext): Statement =
    Block(ctx.simple_stmt().map(_.stmt).filter(_ != null).map(visitStmt))


  // Memories are fairly complicated to translate thus have a dedicated method
  private def visitMem[FirrtlNode](ctx: FIRRTLParser.StmtContext): Statement = {
    val readers = mutable.ArrayBuffer.empty[String]
    val writers = mutable.ArrayBuffer.empty[String]
    val readwriters = mutable.ArrayBuffer.empty[String]
    case class ParamValue(typ: Option[Type] = None, lit: Option[Int] = None, ruw: Option[String] = None, unique: Boolean = true)
    val fieldMap = mutable.HashMap[String, ParamValue]()

    def parseMemFields(memFields: Seq[MemFieldContext]): Unit =
      memFields.foreach { field =>
        val fieldName = field.children(0).getText

        fieldName match {
          case "reader" => readers ++= field.id().map(_.getText)
          case "writer" => writers ++= field.id().map(_.getText)
          case "readwriter" => readwriters ++= field.id().map(_.getText)
          case _ =>
            val paramDef = fieldName match {
              case "data-type" => ParamValue(typ = Some(visitType(field.`type`())))
              case "read-under-write" => ParamValue(ruw = Some(field.ruw().getText)) // TODO
              case _ => ParamValue(lit = Some(field.IntLit().getText.toInt))
            }
            if (fieldMap.contains(fieldName))
              throw new ParameterRedefinedException(s"Redefinition of $fieldName in FIRRTL line:${field.start.getLine}")
            else
              fieldMap += fieldName -> paramDef
        }
      }

    val info = visitInfo(Option(ctx.info), ctx)

    // Build map of different Memory fields to their values
    try {
      parseMemFields(ctx.memField())
    } catch {
      // attach line number
      case e: ParameterRedefinedException => throw new ParameterRedefinedException(s"[$info] ${e.message}")
    }

    // Check for required fields
    Seq("data-type", "depth", "read-latency", "write-latency") foreach { field =>
      fieldMap.getOrElse(field, throw new ParameterNotSpecifiedException(s"[$info] Required mem field $field not found"))
    }

    def lit(param: String) = fieldMap(param).lit.get
    val ruw = fieldMap.get("read-under-write").map(_.ruw).getOrElse(None)

    DefMemory(info,
      name = ctx.id(0).getText, dataType = fieldMap("data-type").typ.get,
      depth = lit("depth"),
      writeLatency = lit("write-latency"), readLatency = lit("read-latency"),
      readers = readers, writers = writers, readwriters = readwriters,
      readUnderWrite = ruw
    )
  }

  // visitStringLit
  private def visitStringLit[FirrtlNode](node: TerminalNode): StringLit = {
    val raw = node.getText.tail.init // Remove surrounding double quotes
    FIRRTLStringLitHandler.unescape(raw)
  }

  private def visitWhen[FirrtlNode](ctx: WhenContext): Conditionally = {
    val info = visitInfo(Option(ctx.info(0)), ctx)

    val alt: Statement =
      if (ctx.when() != null)
        visitWhen(ctx.when())
      else if (ctx.suite().length > 1)
        visitSuite(ctx.suite(1))
      else
        EmptyStmt

    Conditionally(info, visitExp(ctx.exp()), visitSuite(ctx.suite(0)), alt)
  }

  // visitStmt
  private def visitStmt[FirrtlNode](ctx: FIRRTLParser.StmtContext): Statement = {
    val info = visitInfo(Option(ctx.info), ctx)
    ctx.getChild(0) match {
      case when: WhenContext => visitWhen(when)
      case term: TerminalNode => term.getText match {
        case "wire" => DefWire(info, ctx.id(0).getText, visitType(ctx.`type`()))
        case "reg" =>
          val name = ctx.id(0).getText
          val tpe = visitType(ctx.`type`())
          val (reset, init) = {
            val rb = ctx.reset_block()
            if (rb != null) {
              val sr = rb.simple_reset(0).simple_reset0()
              (visitExp(sr.exp(0)), visitExp(sr.exp(1)))
            }
            else
              (UIntLiteral(0, IntWidth(1)), Reference(name, tpe))
          }
          DefRegister(info, name, tpe, visitExp(ctx.exp(0)), reset, init)
        case "mem" => visitMem(ctx)
        case "cmem" =>
          val t = visitType(ctx.`type`())
          t match {
            case (t: VectorType) => CDefMemory(info, ctx.id(0).getText, t.tpe, t.size, seq = false)
            case _ => throw new ParserException(s"${
              info
            }: Must provide cmem with vector type")
          }
        case "smem" =>
          val t = visitType(ctx.`type`())
          t match {
            case (t: VectorType) => CDefMemory(info, ctx.id(0).getText, t.tpe, t.size, seq = true)
            case _ => throw new ParserException(s"${
              info
            }: Must provide cmem with vector type")
          }
        case "inst" => DefInstance(info, ctx.id(0).getText, ctx.id(1).getText)
        case "node" => DefNode(info, ctx.id(0).getText, visitExp(ctx.exp(0)))

        case "stop(" => Stop(info, string2Int(ctx.IntLit().getText), visitExp(ctx.exp(0)), visitExp(ctx.exp(1)))
        case "attach" => Attach(info, visitExp(ctx.exp.head), ctx.exp.tail map visitExp)
        case "printf(" => Print(info, visitStringLit(ctx.StringLit), ctx.exp.drop(2).map(visitExp),
          visitExp(ctx.exp(0)), visitExp(ctx.exp(1)))
        case "skip" => EmptyStmt
      }
      // If we don't match on the first child, try the next one
      case _ =>
        ctx.getChild(1).getText match {
          case "<=" => Connect(info, visitExp(ctx.exp(0)), visitExp(ctx.exp(1)))
          case "<-" => PartialConnect(info, visitExp(ctx.exp(0)), visitExp(ctx.exp(1)))
          case "is" => IsInvalid(info, visitExp(ctx.exp(0)))
          case "mport" => CDefMPort(info, ctx.id(0).getText, UnknownType, ctx.id(1).getText, Seq(visitExp(ctx.exp(0)), visitExp(ctx.exp(1))), visitMdir(ctx.mdir))
        }
    }
  }

  // TODO
  // - Add mux
  // - Add validif
  private def visitExp[FirrtlNode](ctx: FIRRTLParser.ExpContext): Expression =
    if (ctx.getChildCount == 1)
      Reference(ctx.getText, UnknownType)
    else
      ctx.getChild(0).getText match {
        case "UInt" =>
          // This could be better
          val (width, value) =
            if (ctx.getChildCount > 4)
              (IntWidth(string2BigInt(ctx.IntLit(0).getText)), string2BigInt(ctx.IntLit(1).getText))
            else {
              val bigint = string2BigInt(ctx.IntLit(0).getText)
              (IntWidth(BigInt(scala.math.max(bigint.bitLength, 1))), bigint)
            }
          UIntLiteral(value, width)
        case "SInt" =>
          val (width, value) =
            if (ctx.getChildCount > 4)
              (IntWidth(string2BigInt(ctx.IntLit(0).getText)), string2BigInt(ctx.IntLit(1).getText))
            else {
              val bigint = string2BigInt(ctx.IntLit(0).getText)
              (IntWidth(BigInt(bigint.bitLength + 1)), bigint)
            }
          SIntLiteral(value, width)
        case "validif(" => ValidIf(visitExp(ctx.exp(0)), visitExp(ctx.exp(1)), UnknownType)
        case "mux(" => Mux(visitExp(ctx.exp(0)), visitExp(ctx.exp(1)), visitExp(ctx.exp(2)), UnknownType)
        case _ =>
          ctx.getChild(1).getText match {
            case "." => new SubField(visitExp(ctx.exp(0)), ctx.id.getText, UnknownType)
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
  private def visitPrimop[FirrtlNode](ctx: FIRRTLParser.PrimopContext): PrimOp = fromString(ctx.getText.stripSuffix("("))

  // visit Id and Keyword?
}
