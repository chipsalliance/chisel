// See LICENSE for license details.

package firrtl

import org.antlr.v4.runtime.ParserRuleContext
import org.antlr.v4.runtime.tree.TerminalNode
import scala.collection.JavaConverters._
import scala.collection.mutable
import firrtl.antlr._
import PrimOps._
import FIRRTLParser._
import Parser.{AppendInfo, GenInfo, IgnoreInfo, InfoMode, UseInfo}
import firrtl.ir._
import Utils.throwInternalError


class Visitor(infoMode: InfoMode) extends FIRRTLBaseVisitor[FirrtlNode] {
  // Strip file path
  private def stripPath(filename: String) = filename.drop(filename.lastIndexOf("/") + 1)

  // Check if identifier is made of legal characters
  private def legalId(id: String) = {
    val legalChars = ('A' to 'Z').toSet ++ ('a' to 'z').toSet ++ ('0' to '9').toSet ++ Set('_', '$')
    id forall legalChars
  }

  def visit[FirrtlNode](ctx: CircuitContext): Circuit = visitCircuit(ctx)

  //  These regex have to change if grammar changes
  private val HexPattern = """\"*h([+\-]?[a-zA-Z0-9]+)\"*""".r
  private val DecPattern = """([+\-]?[1-9]\d*)""".r
  private val ZeroPattern = "0".r

  private def string2BigInt(s: String): BigInt = {
    // private define legal patterns
    s match {
      case ZeroPattern(_*) => BigInt(0)
      case HexPattern(hexdigits) => BigInt(hexdigits, 16)
      case DecPattern(num) => BigInt(num, 10)
      case _ => throw new Exception("Invalid String for conversion to BigInt " + s)
    }
  }

  private def string2Int(s: String): Int = string2BigInt(s).toInt

  private def visitInfo(ctx: Option[InfoContext], parentCtx: ParserRuleContext): Info = {
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
        else ir.FileInfo(ir.StringLit.unescape(useInfo))
      case AppendInfo(filename) =>
        val newInfo = useInfo + ":" + genInfo(filename)
        ir.FileInfo(ir.StringLit.unescape(newInfo))
      case GenInfo(filename) =>
        ir.FileInfo(ir.StringLit.unescape(genInfo(filename)))
      case IgnoreInfo => NoInfo
    }
  }

  private def visitCircuit[FirrtlNode](ctx: CircuitContext): Circuit =
    Circuit(visitInfo(Option(ctx.info), ctx), ctx.module.asScala.map(visitModule), ctx.id.getText)

  private def visitModule[FirrtlNode](ctx: ModuleContext): DefModule = {
    val info = visitInfo(Option(ctx.info), ctx)
    ctx.getChild(0).getText match {
      case "module" => Module(info, ctx.id.getText, ctx.port.asScala.map(visitPort),
        if (ctx.moduleBlock() != null)
          visitBlock(ctx.moduleBlock())
        else EmptyStmt)
      case "extmodule" =>
        val defname = if (ctx.defname != null) ctx.defname.id.getText else ctx.id.getText
        val ports = ctx.port.asScala map visitPort
        val params = ctx.parameter.asScala map visitParameter
        ExtModule(info, ctx.id.getText, ports, defname, params)
    }
  }

  private def visitPort[FirrtlNode](ctx: PortContext): Port = {
    Port(visitInfo(Option(ctx.info), ctx), ctx.id.getText, visitDir(ctx.dir), visitType(ctx.`type`))
  }

  private def visitParameter[FirrtlNode](ctx: ParameterContext): Param = {
    val name = ctx.id.getText
    (ctx.intLit, ctx.StringLit, ctx.DoubleLit, ctx.RawString) match {
      case (int, null, null, null) => IntParam(name, string2BigInt(int.getText))
      case (null, str, null, null) => StringParam(name, visitStringLit(str))
      case (null, null, dbl, null) => DoubleParam(name, dbl.getText.toDouble)
      case (null, null, null, raw) => RawStringParam(name, raw.getText.tail.init.replace("\\'", "'")) // Remove "\'"s
      case _ => throwInternalError(s"visiting impossible parameter ${ctx.getText}")
    }
  }

  private def visitDir[FirrtlNode](ctx: DirContext): Direction =
    ctx.getText match {
      case "input" => Input
      case "output" => Output
    }

  private def visitMdir[FirrtlNode](ctx: MdirContext): MPortDir =
    ctx.getText match {
      case "infer" => MInfer
      case "read" => MRead
      case "write" => MWrite
      case "rdwr" => MReadWrite
    }

  // Match on a type instead of on strings?
  private def visitType[FirrtlNode](ctx: TypeContext): Type = {
    def getWidth(n: IntLitContext): Width = IntWidth(string2BigInt(n.getText))
    ctx.getChild(0) match {
      case term: TerminalNode =>
        term.getText match {
          case "UInt" => if (ctx.getChildCount > 1) UIntType(getWidth(ctx.intLit(0)))
          else UIntType(UnknownWidth)
          case "SInt" => if (ctx.getChildCount > 1) SIntType(getWidth(ctx.intLit(0)))
          else SIntType(UnknownWidth)
          case "Fixed" => ctx.intLit.size match {
            case 0 => FixedType(UnknownWidth, UnknownWidth)
            case 1 => ctx.getChild(2).getText match {
              case "<" => FixedType(UnknownWidth, getWidth(ctx.intLit(0)))
              case _ => FixedType(getWidth(ctx.intLit(0)), UnknownWidth)
            }
            case 2 => FixedType(getWidth(ctx.intLit(0)), getWidth(ctx.intLit(1)))
          }
          case "Clock" => ClockType
          case "AsyncReset" => AsyncResetType
          case "Analog" => if (ctx.getChildCount > 1) AnalogType(getWidth(ctx.intLit(0)))
          else AnalogType(UnknownWidth)
          case "{" => BundleType(ctx.field.asScala.map(visitField))
        }
      case typeContext: TypeContext => new VectorType(visitType(ctx.`type`), string2Int(ctx.intLit(0).getText))
    }
  }

  // Special case "type" of CHIRRTL mems because their size can be BigInt
  private def visitCMemType(ctx: TypeContext): (Type, BigInt) = {
    def loc: String = s"${ctx.getStart.getLine}:${ctx.getStart.getCharPositionInLine}"
    ctx.getChild(0) match {
      case typeContext: TypeContext =>
        val tpe = visitType(ctx.`type`)
        val size = string2BigInt(ctx.intLit(0).getText)
        (tpe, size)
      case _ =>
        throw new ParserException(s"[$loc] Must provide cmem or smem with vector type, got ${ctx.getText}")
    }
  }

  private def visitField[FirrtlNode](ctx: FieldContext): Field = {
    val flip = if (ctx.getChild(0).getText == "flip") Flip else Default
    Field(ctx.fieldId.getText, flip, visitType(ctx.`type`))
  }

  private def visitBlock[FirrtlNode](ctx: ModuleBlockContext): Statement =
    Block(ctx.simple_stmt().asScala.flatMap(x => Option(x.stmt).map(visitStmt)))

  private def visitSuite[FirrtlNode](ctx: SuiteContext): Statement =
    Block(ctx.simple_stmt().asScala.flatMap(x => Option(x.stmt).map(visitStmt)))


  // Memories are fairly complicated to translate thus have a dedicated method
  private def visitMem[FirrtlNode](ctx: StmtContext): Statement = {
    val readers = mutable.ArrayBuffer.empty[String]
    val writers = mutable.ArrayBuffer.empty[String]
    val readwriters = mutable.ArrayBuffer.empty[String]
    case class ParamValue(typ: Option[Type] = None, lit: Option[BigInt] = None, ruw: Option[String] = None, unique: Boolean = true)
    val fieldMap = mutable.HashMap[String, ParamValue]()

    def parseMemFields(memFields: Seq[MemFieldContext]): Unit =
      memFields.foreach { field =>
        val fieldName = field.children.asScala(0).getText

        fieldName match {
          case "reader" => readers ++= field.id().asScala.map(_.getText)
          case "writer" => writers ++= field.id().asScala.map(_.getText)
          case "readwriter" => readwriters ++= field.id().asScala.map(_.getText)
          case _ =>
            val paramDef = fieldName match {
              case "data-type" => ParamValue(typ = Some(visitType(field.`type`())))
              case "read-under-write" => ParamValue(ruw = Some(field.ruw().getText)) // TODO
              case _ => ParamValue(lit = Some(BigInt(field.intLit().getText)))
            }
            if (fieldMap.contains(fieldName))
              throw new ParameterRedefinedException(s"Redefinition of $fieldName in FIRRTL line:${field.start.getLine}")
            else
              fieldMap(fieldName) = paramDef
        }
      }

    val info = visitInfo(Option(ctx.info), ctx)

    // Build map of different Memory fields to their values
    try {
      parseMemFields(ctx.memField().asScala)
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
      writeLatency = lit("write-latency").toInt,
      readLatency = lit("read-latency").toInt,
      readers = readers, writers = writers, readwriters = readwriters,
      readUnderWrite = ruw
    )
  }

  // visitStringLit
  private def visitStringLit[FirrtlNode](node: TerminalNode): StringLit = {
    val raw = node.getText.tail.init // Remove surrounding double quotes
    ir.StringLit.unescape(raw)
  }

  private def visitWhen[FirrtlNode](ctx: WhenContext): Conditionally = {
    val info = visitInfo(Option(ctx.info(0)), ctx)

    val alt: Statement =
      if (ctx.when() != null)
        visitWhen(ctx.when())
      else if (ctx.suite().asScala.length > 1)
        visitSuite(ctx.suite(1))
      else
        EmptyStmt

    Conditionally(info, visitExp(ctx.exp()), visitSuite(ctx.suite(0)), alt)
  }

  // visitStmt
  private def visitStmt[FirrtlNode](ctx: StmtContext): Statement = {
    val ctx_exp = ctx.exp.asScala
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
          DefRegister(info, name, tpe, visitExp(ctx_exp(0)), reset, init)
        case "mem" => visitMem(ctx)
        case "cmem" =>
          val (tpe, size) = visitCMemType(ctx.`type`())
          CDefMemory(info, ctx.id(0).getText, tpe, size, seq = false)
        case "smem" =>
          val (tpe, size) = visitCMemType(ctx.`type`())
          CDefMemory(info, ctx.id(0).getText, tpe, size, seq = true)
        case "inst" => DefInstance(info, ctx.id(0).getText, ctx.id(1).getText)
        case "node" => DefNode(info, ctx.id(0).getText, visitExp(ctx_exp(0)))

        case "stop(" => Stop(info, string2Int(ctx.intLit().getText), visitExp(ctx_exp(0)), visitExp(ctx_exp(1)))
        case "attach" => Attach(info, ctx_exp map visitExp)
        case "printf(" => Print(info, visitStringLit(ctx.StringLit), ctx_exp.drop(2).map(visitExp),
          visitExp(ctx_exp(0)), visitExp(ctx_exp(1)))
        case "skip" => EmptyStmt
      }
      // If we don't match on the first child, try the next one
      case _ =>
        ctx.getChild(1).getText match {
          case "<=" => Connect(info, visitExp(ctx_exp(0)), visitExp(ctx_exp(1)))
          case "<-" => PartialConnect(info, visitExp(ctx_exp(0)), visitExp(ctx_exp(1)))
          case "is" => IsInvalid(info, visitExp(ctx_exp(0)))
          case "mport" => CDefMPort(info, ctx.id(0).getText, UnknownType, ctx.id(1).getText, Seq(visitExp(ctx_exp(0)), visitExp(ctx_exp(1))), visitMdir(ctx.mdir))
        }
    }
  }

  private def visitExp[FirrtlNode](ctx: ExpContext): Expression = {
    val ctx_exp = ctx.exp.asScala
    ctx.getChild(0) match {
      case _: IdContext => Reference(ctx.getText, UnknownType)
      case _: ExpContext =>
        ctx.getChild(1).getText match {
          case "." =>
            val expr1 = visitExp(ctx_exp(0))
            // TODO Workaround for #470
            if (ctx.fieldId == null) {
              ctx.DoubleLit.getText.split('.') match {
                case Array(a, b) if legalId(a) && legalId(b) =>
                  val inner = new SubField(expr1, a, UnknownType)
                  new SubField(inner, b, UnknownType)
                case Array() => throw new ParserException(s"Illegal Expression at ${ctx.getText}")
              }
            } else {
              new SubField(expr1, ctx.fieldId.getText, UnknownType)
            }
          case "[" =>
            if (ctx.exp(1) == null)
              new SubIndex(visitExp(ctx_exp(0)), string2Int(ctx.intLit(0).getText), UnknownType)
            else
              new SubAccess(visitExp(ctx_exp(0)), visitExp(ctx_exp(1)), UnknownType)
        }
      case _: PrimopContext =>
        DoPrim(visitPrimop(ctx.primop),
               ctx_exp.map(visitExp),
               ctx.intLit.asScala.map(x => string2BigInt(x.getText)),
               UnknownType)
      case _ =>
        ctx.getChild(0).getText match {
          case "UInt" =>
            if (ctx.getChildCount > 4) {
              val width = IntWidth(string2BigInt(ctx.intLit(0).getText))
              val value = string2BigInt(ctx.intLit(1).getText)
              UIntLiteral(value, width)
            } else {
              val value = string2BigInt(ctx.intLit(0).getText)
              UIntLiteral(value)
            }
          case "SInt" =>
            if (ctx.getChildCount > 4) {
              val width = string2BigInt(ctx.intLit(0).getText)
              val value = string2BigInt(ctx.intLit(1).getText)
              SIntLiteral(value, IntWidth(width))
            } else {
              val str = ctx.intLit(0).getText
              val value = string2BigInt(str)
              SIntLiteral(value)
            }
          case "validif(" => ValidIf(visitExp(ctx_exp(0)), visitExp(ctx_exp(1)), UnknownType)
          case "mux(" => Mux(visitExp(ctx_exp(0)), visitExp(ctx_exp(1)), visitExp(ctx_exp(2)), UnknownType)
        }
    }
  }

  // stripSuffix("(") is included because in ANTLR concrete syntax we have to include open parentheses,
  //  see grammar file for more details
  private def visitPrimop[FirrtlNode](ctx: PrimopContext): PrimOp = fromString(ctx.getText.stripSuffix("("))

  // visit Id and Keyword?
}
