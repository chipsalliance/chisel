// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator.scalatest

import chisel3.testing.scalatest.HasConfigMap
import org.scalatest.TestSuite
import scala.collection.mutable.HashMap
import svsim.Backend.HarnessCompilationFlags.enableVcdTracingSupport
import svsim.CommonCompilationSettings.VerilogPreprocessorDefine
import svsim.{Backend, CommonCompilationSettings}

object HasCliArguments {

  case class CliOption(
    name:                  String,
    validate:              (String) => Boolean,
    updateCommonSettings:  (String, CommonCompilationSettings) => CommonCompilationSettings,
    updateBackendSettings: (String, Backend.Settings) => Backend.Settings
  )

}

trait HasCliArguments extends HasConfigMap { this: TestSuite =>

  import HasCliArguments._

  private val options = HashMap.empty[String, CliOption]

  final def addOption(option: CliOption): Unit = {
    if (options.contains(option.name))
      throw new Exception("unable to add option with name '$name' because this is already taken by another option")

    options += option.name -> option
  }

  implicit def commonSettingsModifications: svsim.CommonSettingsModifications = {
    options.values.foldLeft(_: CommonCompilationSettings) { case (acc, option) =>
      configMap.getOptional[String](option.name) match {
        case None => acc
        case Some(value) =>
          option.validate(value)
          option.updateCommonSettings.apply(value, acc)
      }
    }
  }

  implicit def backendSettingsModifications: svsim.BackendSettingsModifications =
    options.values.foldLeft(_: Backend.Settings) { case (acc, option) =>
      configMap.getOptional[String](option.name) match {
        case None => acc
        case Some(value) =>
          option.validate(value)
          option.updateBackendSettings.apply(value, acc)
      }
    }

}

object CLI {

  import HasCliArguments.CliOption

  trait WaveformVCD { this: HasCliArguments =>

    println("adding VCD option")
    addOption(
      CliOption(
        name = "emitVCD",
        validate = _ match {
          case "true" | "1" => true
          case invalid =>
            throw new IllegalArgumentException(
              "invalid argument '$invalid' for option '$name', must be one of [true, 1]"
            )
        },
        updateCommonSettings = (_, options) => {
          options.copy(verilogPreprocessorDefines =
            options.verilogPreprocessorDefines :+ VerilogPreprocessorDefine(enableVcdTracingSupport)
          )
        },
        updateBackendSettings = (_, options) =>
          options match {
            case options: svsim.vcs.Backend.CompilationSettings =>
              options.copy(
                traceSettings = options.traceSettings.copy(enableVcd = true)
              )
            case options: svsim.verilator.Backend.CompilationSettings =>
              options.copy(
                traceStyle = options.traceStyle match {
                  case None => Some(svsim.verilator.Backend.CompilationSettings.TraceStyle.Vcd(filename = "trace.vcd"))
                  case alreadySet => alreadySet
                }
              )
          }
      )
    )

  }

}
