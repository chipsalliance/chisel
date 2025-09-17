// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator.scalatest

import chisel3.simulator.{ChiselOptionsModifications, FirtoolOptionsModifications, HasSimulator}
import chisel3.testing.scalatest.HasConfigMap
import firrtl.options.StageUtils.dramaticMessage
import org.scalatest.TestSuite
import scala.collection.mutable
import scala.util.control.NoStackTrace
import svsim.Backend.HarnessCompilationFlags.{
  enableFsdbTracingSupport,
  enableVcdTracingSupport,
  enableVpdTracingSupport
}
import svsim.CommonCompilationSettings.VerilogPreprocessorDefine
import svsim.{Backend, CommonCompilationSettings}

object HasCliOptions {

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
    name:                       String,
    help:                       String,
    convert:                    (String) => A,
    updateChiselOptions:        (A, Array[String]) => Array[String],
    updateFirtoolOptions:       (A, Array[String]) => Array[String],
    updateCommonSettings:       (A, CommonCompilationSettings) => CommonCompilationSettings,
    updateBackendSettings:      (A, Backend.Settings) => Backend.Settings,
    updateUnsetChiselOptions:   (Array[String]) => Array[String],
    updateUnsetFirtoolOptions:  (Array[String]) => Array[String],
    updateUnsetCommonSettings:  (CommonCompilationSettings) => CommonCompilationSettings,
    updateUnsetBackendSettings: (Backend.Settings) => Backend.Settings
  )

  object CliOption {

    def apply[A](
      name:                  String,
      help:                  String,
      convert:               (String) => A,
      updateChiselOptions:   (A, Array[String]) => Array[String],
      updateFirtoolOptions:  (A, Array[String]) => Array[String],
      updateCommonSettings:  (A, CommonCompilationSettings) => CommonCompilationSettings,
      updateBackendSettings: (A, Backend.Settings) => Backend.Settings
    ): CliOption[A] = {
      apply(
        name = name,
        help = help,
        convert = convert,
        updateChiselOptions = updateChiselOptions,
        updateFirtoolOptions = updateFirtoolOptions,
        updateCommonSettings = updateCommonSettings,
        updateBackendSettings = updateBackendSettings,
        updateUnsetChiselOptions = identity,
        updateUnsetFirtoolOptions = identity,
        updateUnsetCommonSettings = identity,
        updateUnsetBackendSettings = identity
      )
    }

    /** A simple command line option which does not affect common or backend settings.
      *
      * This is intended to be used to create options which are passed directly
      * to tests as opposed to creating options which are used to affect
      * compilation or simulation settings.
      *
      * @param name the name of the option
      * @param help help text to show to tell the user how to use this option
      * @param convert convert the `<value>` to type `A`
      */
    def simple[A](name: String, help: String, convert: (String => A)): CliOption[A] = new CliOption[A](
      name = name,
      help = help,
      convert = convert,
      updateChiselOptions = (_, a) => a,
      updateFirtoolOptions = (_, a) => a,
      updateCommonSettings = (_, a) => a,
      updateBackendSettings = (_, a) => a,
      updateUnsetChiselOptions = identity,
      updateUnsetFirtoolOptions = identity,
      updateUnsetCommonSettings = identity,
      updateUnsetBackendSettings = identity
    )

    /** Add a double option to a test.
      *
      * @param name the name of the option
      * @param help help text to show to tell the user how to use this option
      * @throws IllegalArgumentException if the value is not convertible to a
      * double precision floating point number
      */
    def double(name: String, help: String): CliOption[Double] = simple[Double](
      name = name,
      help = help,
      convert = value =>
        try {
          value.toDouble
        } catch {
          case e: NumberFormatException =>
            throw new java.lang.IllegalArgumentException(
              s"illegal value '$value' for ChiselSim ScalaTest option '$name'.  The value must be convertible to a floating point number."
            ) with NoStackTrace
        }
    )

    /** Add an integer option to a test.
      *
      * @param name the name of the option
      * @param help help text to show to tell the user how to use this option
      * @throws IllegalArgumentException if the value is not convertible to an
      * integer
      */
    def int(name: String, help: String): CliOption[Int] = simple[Int](
      name = name,
      help = help,
      convert = value =>
        try {
          value.toInt
        } catch {
          case e: NumberFormatException =>
            throw new java.lang.IllegalArgumentException(
              s"illegal value '$value' for ChiselSim ScalaTest option '$name'.  The value must be convertible to an integer."
            ) with NoStackTrace
        }
    )

    /** Add a string option to a test.
      *
      * @param name the name of the option
      * @param help help text to show to tell the user how to use this option
      */
    def string(name: String, help: String): CliOption[String] = simple[String](
      name = name,
      help = help,
      convert = identity
    )

    /** Add a flag option to a test.
      *
      * This is an option which can only take one of two "truthy" values: `1` or
      * `true`.  Any "falsey" values are not allowed.  This option is a stand-in
      * for any option which is supposed to be a flag to a test which has some
      * effect if set.
      *
      * This option exists because Scalatest forces options to have a value.  It
      * is illegal to pass an option like `-Dfoo`.  This [[flag]] option exists
      * to problem a single flag-style option as opposed to having users roll
      * their own.
      *
      * @param name the name of the option
      * @param help help text to show to tell the user how to use this option
      */
    def flag(name: String, help: String): CliOption[Unit] = simple[Unit](
      name = name,
      help = help,
      convert = value => {
        val trueValue = Set("true", "1")
        trueValue.contains(value) match {
          case true => ()
          case false =>
            throw new IllegalArgumentException(
              s"""invalid argument '$value' for option '$name', must be one of ${trueValue.mkString("[", ", ", "]")}"""
            ) with NoStackTrace
        }
      }
    )
  }

}

trait HasCliOptions extends HasConfigMap { this: TestSuite =>

  import HasCliOptions._

  private val options = mutable.HashMap.empty[String, CliOption[_]]

  final def addOption(option: CliOption[_]): Unit = {
    if (options.contains(option.name))
      throw new Exception(
        s"unable to add option with name '${option.name}' because this is already taken by another option"
      )

    options += option.name -> option
  }

  final def getOption[A](name: String): Option[A] = {
    val value: Option[Any] = configMap.get(name)
    value.map(_.asInstanceOf[String]).map(options(name).convert(_)).map(_.asInstanceOf[A])
  }

  private def helpBody = {
    // Sort the options by name to give predictable output.
    val optionsHelp = options.keys.toSeq.sorted
      .map(options)
      .map { case option =>
        s"""|  ${option.name}
            |      ${option.help}
            |""".stripMargin
      }
      .mkString
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

  implicit def chiselOptionsModifications: ChiselOptionsModifications = (original: Array[String]) => {
    illegalOptionCheck()
    options.values.foldLeft(original) { case (acc, option) =>
      configMap.getOptional[String](option.name) match {
        case None => option.updateUnsetChiselOptions.apply(acc)
        case Some(value) =>
          option.updateChiselOptions.apply(option.convert(value), acc)
      }
    }
  }

  implicit def firtoolOptionsModifications: FirtoolOptionsModifications = (original: Array[String]) => {
    illegalOptionCheck()
    options.values.foldLeft(original) { case (acc, option) =>
      configMap.getOptional[String](option.name) match {
        case None => option.updateUnsetFirtoolOptions.apply(acc)
        case Some(value) =>
          option.updateFirtoolOptions.apply(option.convert(value), acc)
      }
    }
  }

  implicit def commonSettingsModifications: svsim.CommonSettingsModifications = (original: CommonCompilationSettings) =>
    {
      illegalOptionCheck()
      options.values.foldLeft(original) { case (acc, option) =>
        configMap.getOptional[String](option.name) match {
          case None => option.updateUnsetCommonSettings.apply(acc)
          case Some(value) =>
            option.updateCommonSettings.apply(option.convert(value), acc)
        }
      }
    }

  implicit def backendSettingsModifications: svsim.BackendSettingsModifications = (original: Backend.Settings) => {
    illegalOptionCheck()
    options.values.foldLeft(original) { case (acc, option) =>
      configMap.getOptional[String](option.name) match {
        case None => option.updateUnsetBackendSettings.apply(acc)
        case Some(value) =>
          option.updateBackendSettings.apply(option.convert(value), acc)
      }
    }
  }

  addOption(
    CliOption.simple[Unit](
      name = "help",
      help = "display this help text",
      convert = _ => {
        throw new IllegalArgumentException(
          dramaticMessage(
            header = Some("help text requested"),
            body = helpBody
          )
        ) with NoStackTrace
      }
    )
  )

}
