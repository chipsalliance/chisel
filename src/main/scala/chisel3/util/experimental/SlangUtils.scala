package chisel3.util.experimental

import chisel3._
import chisel3.experimental.{Analog, DoubleParam, IntParam, Param, StringParam}
import logger.LazyLogging
import ujson.Value.Value
case object SLangNotFoundException extends Exception

object SlangUtils extends LazyLogging {

  /** Use Slang parse Verilog into Json AST. */
  def getVerilogAst(verilog: String): Value = {
    val astFile = os.temp()
    try {
      os.proc(
        "slang",
        os.temp(verilog),
        "--single-unit",
        "--ignore-unknown-modules",
        "--compat",
        "vcs",
        "--ast-json",
        astFile
      ).call()
    } catch {
      case e: java.io.IOException if e.getMessage.contains("error=2, No such file or directory") =>
        logger.error(s"slang is not found in your PATH:\n${sys.env("PATH").split(":").mkString("\n")}".stripMargin)
        throw SLangNotFoundException
    }
    ujson.read(os.read(astFile))
  }

  /** Extract IO from Verilog AST. */
  def verilogModuleIO(verilogAst: Value): Seq[(String, Data)] = {
    verilogAst.obj("members").arr.flatMap {
      case value: ujson.Obj if value.value("kind").strOpt.contains("Instance") =>
        value.value("body").obj.value("members").arr.flatMap {
          case value: ujson.Obj if value.value("kind").strOpt.contains("Port") =>
            Some(
              value.value("name").str -> {
                val width: Int =
                  """logic\[(\d+):(\d+)\]""".r("MSB", "LSB").findFirstMatchIn(value.value("type").str) match {
                    case Some(m) => m.group("MSB").toInt - m.group("LSB").toInt + 1
                    case None =>
                      value.value("type").str match {
                        case "logic" => 1
                        case _       => throw new ChiselException(s"Unhandled type ${value.value("type").str}")
                      }
                  }

                value.value("direction").str match {
                  case "In"    => Input(UInt(width.W))
                  case "Out"   => Output(UInt(width.W))
                  case "InOut" => Analog(width.W)
                }
              }
            )
          case _ => None
        }
      case _ => None
    }
  }.toSeq

  /** Extract Module Name from Verilog AST. */
  def verilogModuleName(verilogAst: Value): String = {
    val names = verilogAst.obj("members").arr.flatMap {
      case value: ujson.Obj if value.value("kind").strOpt.contains("Instance") => Some(value.value("name").str)
      case _ => None
    }
    require(names.length == 1, "only support one verilog module currently")
    names.head
  }

  /** Extract Module Parameter from Verilog AST. */
  def verilogParameter(verilogAst: Value): Seq[(String, Param)] = {
    verilogAst.obj("members").arr.flatMap {
      case value: ujson.Obj if value.value("kind").strOpt.contains("Instance") =>
        value.value("body").obj.value("members").arr.flatMap {
          case value: ujson.Obj if value.value("kind").strOpt.contains("Parameter") =>
            Some(value.value("name").str -> (value.value("type").str match {
              case "real"   => DoubleParam(value.value("value").str.toDouble)
              case "string" => StringParam(value.value("value").str.stripPrefix("\"").stripSuffix("\""))
              case "int"    => IntParam(BigInt(value.value("value").str.toInt))
              case lit      => throw new ChiselException(s"unsupported literal: $lit")
            }))
          case _ => None
        }
      case _ => None
    }
  }.toSeq
}
