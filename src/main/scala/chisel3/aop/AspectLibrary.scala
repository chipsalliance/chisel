// SPDX-License-Identifier: Apache-2.0

package chisel3.aop

import chisel3.RawModule
import firrtl.options.{OptionsException, RegisteredLibrary, ShellOption}

/** Enables adding aspects to a design from the commandline, e.g.
  *  sbt> runMain chisel3.stage.ChiselMain --module <module> --with-aspect <aspect>
  */
final class AspectLibrary() extends RegisteredLibrary {
  val name = "AspectLibrary"

  import scala.reflect.runtime.universe._

  private def apply(aspectName: String): Aspect[RawModule] = {
    try {
      // If a regular class, instantiate, otherwise try as a singleton object
      try {
        val x = Class.forName(aspectName).asInstanceOf[Class[_ <: Aspect[RawModule]]]
        x.getDeclaredConstructor().newInstance()
      } catch {
        case e: NoSuchMethodException =>
          val rm = runtimeMirror(getClass.getClassLoader)
          val x = rm.staticModule(aspectName)
          try {
            rm.reflectModule(x).instance.asInstanceOf[Aspect[RawModule]]
          } catch {
            case _: Exception => throw e
          }
      }
    } catch {
      case e: ClassNotFoundException =>
        throw new OptionsException(s"Unable to locate aspect '$aspectName'! (Did you misspell it?)", e)
      case e: InstantiationException =>
        throw new OptionsException(
          s"Unable to create instance of aspect '$aspectName'! (Does this class take parameters?)",
          e
        )
    }
  }

  val options = Seq(
    new ShellOption[String](
      longOption = "with-aspect",
      toAnnotationSeq = {
        case aspectName: String => Seq(apply(aspectName))
      },
      helpText = "The name/class of an aspect to compile with (must be a class/object without arguments!)",
      helpValueName = Some("<package>.<aspect>")
    )
  )
}
