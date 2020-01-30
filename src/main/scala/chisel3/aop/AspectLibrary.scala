// See LICENSE for license details.

package chisel3.aop

import chisel3.RawModule
import firrtl.options.{OptionsException, RegisteredLibrary, ShellOption}

case class AspectLibrary() extends RegisteredLibrary  {
  val name = "AspectLibrary"

  def apply(aspectName: String): Aspect[RawModule] = {
    try {
      Class.forName(aspectName).asInstanceOf[Class[_ <: Aspect[RawModule]]].newInstance()
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
    helpText = "The name/class of an aspect to compile with (must be a case object!)",
    helpValueName = Some("<package>.<aspect>")
  ))

}
