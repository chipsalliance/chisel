package chisel3.util.experimental

import chisel3._
import chisel3.core.{ChiselAnnotation, CompileOptions, RunFirrtlTransform, annotate, requireIsHardware}
import firrtl.Transform
import firrtl.transforms.{GroupAnnotation, GroupComponents}


object group {
  def apply[T <: Data](components: Seq[T], newModule: String, newInstance: String, outputSuffix: Option[String] = None, inputSuffix: Option[String] = None)(implicit compileOptions: CompileOptions): Unit = {
    if (compileOptions.checkSynthesizable) {
      components.foreach { data =>
        requireIsHardware(data, s"Component ${data.toString} is marked to group, but is not bound.")
      }
    }
    annotate(new ChiselAnnotation with RunFirrtlTransform {
      def toFirrtl = GroupAnnotation(components.map(_.toNamed), newModule, newInstance, outputSuffix, inputSuffix)

      override def transformClass: Class[_ <: Transform] = classOf[GroupComponents]
    })
  }
}

