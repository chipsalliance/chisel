// See LICENSE for license details.

package chisel3.aop

import chisel3.RawModule
import firrtl.options.{OptionsException, RegisteredLibrary, ShellOption}

/** Enables adding aspects to a design from the commandline, e.g.
  *  sbt> runMain chisel3.stage.ChiselMain --module <module> --with-aspect <aspect>
  */
case class AspectLibrary() extends RegisteredLibrary  {
  val name = "AspectLibrary"

  import scala.reflect.runtime.universe._

  private def isObject[T](x: T)(implicit tag: TypeTag[T]): Boolean = PartialFunction.cond(tag.tpe) {
    case SingleType(_, _) => true
  }

  def apply(aspectName: String): Aspect[RawModule] = {
    try {
      val x = Class.forName(aspectName).asInstanceOf[Class[_ <: Aspect[RawModule]]]
      if(isObject(x)) {
        val runtimeMirror = scala.reflect.runtime.universe.runtimeMirror(getClass.getClassLoader)
        val x = runtimeMirror.staticModule(aspectName)
        runtimeMirror.reflectModule(x).instance.asInstanceOf[Aspect[RawModule]]
      } else {
        x.newInstance()
      }
    } catch {
      case e: ClassNotFoundException =>
        throw new OptionsException(s"Unable to locate aspect '$aspectName'! (Did you misspell it?)", e)
      case e: InstantiationException =>
        throw new OptionsException(
          s"Unable to create instance of aspect '$aspectName'! (Does this class take parameters?)", e)
    }
  }

  val options = Seq(new ShellOption[String](
    longOption = "with-aspect",
    toAnnotationSeq = {
      case aspectName: String => Seq(apply(aspectName))
    },
    helpText = "The name/class of an aspect to compile with (must be a class/object without arguments!)",
    helpValueName = Some("<package>.<aspect>")
  ))

}
