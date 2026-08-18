// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental

import chisel3._
import chisel3.experimental.Analog
import ujson.Value.Value

import scala.collection.immutable.SeqMap

object SlangUtils {

  /** Use Slang to parse Verilog into Json AST. */
  def getVerilogAst(verilog: String): Value = {
    import scala.sys.process._
    import java.nio.charset.StandardCharsets
    import java.nio.file.Files
    val astFile = Files.createTempFile("slang", ".json")
    val verilogFile = Files.createTempFile("slang", ".sv")
    astFile.toFile.deleteOnExit()
    verilogFile.toFile.deleteOnExit()
    Files.write(verilogFile, verilog.getBytes(StandardCharsets.UTF_8))
    val command = Seq(
      "slang",
      verilogFile.toString,
      "--single-unit",
      "--ignore-unknown-modules",
      "--compat",
      "vcs",
      "--ast-json",
      astFile.toString
    )
    val stderr = new StringBuilder
    val exitCode = command.!(ProcessLogger(_ => (), line => stderr.append(line).append('\n')))
    if (exitCode != 0) {
      throw new ChiselException(s"slang exited with non-zero exit code $exitCode:\n${stderr.toString}")
    }
    ujson.read(new String(Files.readAllBytes(astFile), StandardCharsets.UTF_8))
  }

  /** Extract IO from Verilog AST. */
  def verilogModuleIO(verilogAst: Value): SeqMap[String, Data] = {
    verilogAst.obj("design").obj("members").arr.flatMap {
      case value: ujson.Obj if value.value("kind").strOpt.contains("Instance") =>
        value.value("body").obj.value("members").arr.flatMap {
          case value: ujson.Obj if value.value("kind").strOpt.contains("Port") =>
            Some(
              value.value("name").str -> {
                val width: Int =
                  """logic\[(?<MSB>\d+):(?<LSB>\d+)\]""".r
                    .findFirstMatchIn(value.value("type").str) match {
                    case Some(m) =>
                      m.group("MSB").toInt - m.group("LSB").toInt + 1
                    case None =>
                      value.value("type").str match {
                        case "logic" => 1
                        case _ =>
                          throw new ChiselException(
                            s"Unhandled type ${value.value("type").str}"
                          )
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
  }.to(collection.immutable.SeqMap)

  /** Extract module name from Verilog AST. */
  def verilogModuleName(verilogAst: Value): String = {
    val names = verilogAst.obj("design").obj("members").arr.flatMap {
      case value: ujson.Obj if value.value("kind").strOpt.contains("Instance") =>
        Some(value.value("name").str)
      case _ => None
    }
    require(names.length == 1, "only support one verilog module currently")
    names.head
  }

  /** Extract module parameter from Verilog AST. */
  def verilogParameter(verilogAst: Value): Seq[(String, Param)] = {
    verilogAst.obj("design").obj("members").arr.flatMap {
      case value: ujson.Obj if value.value("kind").strOpt.contains("Instance") =>
        value.value("body").obj.value("members").arr.flatMap {
          case value: ujson.Obj if value.value("kind").strOpt.contains("Parameter") =>
            Some(value.value("name").str -> (value.value("type").str match {
              case "real" => DoubleParam(value.value("value").str.toDouble)
              case "string" =>
                StringParam(
                  value.value("value").str.stripPrefix("\"").stripSuffix("\"")
                )
              case "int" => IntParam(BigInt(value.value("value").str.toInt))
              case lit =>
                throw new ChiselException(
                  s"unsupported literal: $lit in\n ${ujson.reformat(value, 2)}"
                )
            }))
          case _ => None
        }
      case _ => None
    }
  }.toSeq
}
