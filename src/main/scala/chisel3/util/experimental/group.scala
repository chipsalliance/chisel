// See LICENSE for license details.

package chisel3.util.experimental

import chisel3._
import chisel3.experimental.{ChiselAnnotation, RunFirrtlTransform, annotate}
import chisel3.internal.requireIsHardware
import firrtl.Transform
import firrtl.transforms.{GroupAnnotation, GroupComponents}

/** Marks that a module to be ignored in Dedup Transform in Firrtl pass
  *
  * @example {{{
  * class MyModule extends Module {
  *   val io = IO(new Bundle{
  *     val a = Input(Bool())
  *     val b = Output(Bool())
  *   })
  *   val reg1 = RegInit(0.U)
  *   reg1 := io.a
  *   val reg2 = RegNext(reg1)
  *   io.b := reg2
  *   group(Seq(reg1, reg2), "DosRegisters", "doubleReg")
  * }
  * }}}
  *
  * @note Intermediate wires will get pulled into the new instance, but intermediate registers will not
  *       because they are also connected to their module's clock port. This means that if you want
  *       a register to be included in a group, it must be explicitly referred to in the input list.
  */
object group {

  /** Marks a set of components (and their interconnected components) to be included in a new
    * instance hierarchy.
    *
    * @note Intermediate wires will get pulled into the new instance, but intermediate registers will not
    *       because they are also connected to their module's clock port. This means that if you want
    *       a register to be included in a group, it must be explicitly referred to in the input list.
    *
    * @param components components in this group
    * @param newModule suggested name of the new module
    * @param newInstance suggested name of the instance of the new module
    * @param outputSuffix suggested suffix of any output ports of the new module
    * @param inputSuffix suggested suffix of any input ports of the new module
    * @param compileOptions necessary for backwards compatibility
    * @tparam T Parent type of input components
    */
  def apply[T <: Data](
      components: Seq[T],
      newModule: String,
      newInstance: String,
      outputSuffix: Option[String] = None,
      inputSuffix: Option[String] = None
  )(implicit compileOptions: CompileOptions): Unit = {
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

