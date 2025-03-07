// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator.scalatest

import chisel3.testing.scalatest.HasConfigMap
import firrtl.options.StageUtils.dramaticMessage
import org.scalatest.TestSuite
import scala.collection.mutable
import scala.util.control.NoStackTrace
import svsim.Backend.HarnessCompilationFlags.enableVcdTracingSupport
import svsim.CommonCompilationSettings.VerilogPreprocessorDefine
import svsim.{Backend, CommonCompilationSettings}

object HasCliArguments {

  /** A ScalaTest command line option of the form `-D<name>=<value>`.
    *
    * @param name the name of the option
    * @param convert conver the `<value>` to the internal type `A`
    * @param updateCommonSettings a function to update the common compilation
    * settings
    * @param updateBackendSettings a function to update the backend-specific
    * compilation settings
    * @tparam the internal type of the option.  This is what the `<value>` will
    * be converted to.
    */
  case class CliOption[A](
    name:                  String,
    help:                  String,
    convert:               (String) => A,
    updateCommonSettings:  (A, CommonCompilationSettings) => CommonCompilationSettings,
    updateBackendSettings: (A, Backend.Settings) => Backend.Settings
  )

}

trait HasCliArguments extends HasConfigMap { this: TestSuite =>

  import HasCliArguments._

  private val options = mutable.HashMap.empty[String, CliOption[_]]

  final def addOption(option: CliOption[_]): Unit = {
    if (options.contains(option.name))
      throw new Exception("unable to add option with name '$name' because this is already taken by another option")

    options += option.name -> option
  }

  private def helpBody = {
    val optionsHelp = options.map { case (_, option) =>
      s"""|  ${option.name}
          |      ${option.help}
          |""".stripMargin
    }.mkString
    s"""|Usage: <ScalaTest> [-D<name>=<value>...]
        |
        |This ChiselSim ScalaTest test supports passing command line arguments via
        |ScalaTest's "config map" feature.  To access this, append `-D<name>=<value>` for
        |a legal option listed below.
        |
        |Options:
        |
        |$optionsHelp""".stripMargin
  }

  private def illegalOptionCheck(): Unit = {
    configMap.keys.foreach { case name =>
      if (!options.contains(name)) {
        throw new IllegalArgumentException(
          dramaticMessage(
            header = Some(s"illegal ChiselSim ScalaTest option '$name'"),
            body = helpBody
          )
        ) with NoStackTrace
      }
    }
  }

  implicit def commonSettingsModifications: svsim.CommonSettingsModifications = (original: CommonCompilationSettings) =>
    {
      illegalOptionCheck()
      options.values.foldLeft(original) { case (acc, option) =>
        configMap.getOptional[String](option.name) match {
          case None => acc
          case Some(value) =>
            option.updateCommonSettings.apply(option.convert(value), acc)
        }
      }
    }

  implicit def backendSettingsModifications: svsim.BackendSettingsModifications = (original: Backend.Settings) => {
    illegalOptionCheck()
    options.values.foldLeft(original) { case (acc, option) =>
      configMap.getOptional[String](option.name) match {
        case None => acc
        case Some(value) =>
          option.updateBackendSettings.apply(option.convert(value), acc)
      }
    }
  }

  addOption(
    CliOption[Unit](
      name = "help",
      help = "display this help text",
      convert = _ => {
        throw new IllegalArgumentException(
          dramaticMessage(
            header = Some("help text requested"),
            body = helpBody
          )
        ) with NoStackTrace
      },
      updateCommonSettings = (_, a) => a,
      updateBackendSettings = (_, a) => a
    )
  )

}

object CLI {

  import HasCliArguments.CliOption

  trait VcdCapability { this: HasCliArguments =>

    addOption(
      CliOption[Unit](
        name = "withVcdCapability",
        help = "compiles the simulator with VCD support. (Use `enableWaves` to dump a VCD.)",
        convert = value => {
          val trueValue = Set("true", "1")
          trueValue.contains(value) match {
            case true => ()
            case false =>
              throw new IllegalArgumentException(
                s"""invalid argument '$value' for option 'enableVcdSupport', must be one of ${trueValue
                    .mkString("[", ", ", "]")}"""
              ) with NoStackTrace
          }
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
