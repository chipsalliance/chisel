package chisel3.libs.transaction

import chisel3._
import chisel3.core.{BaseModule, ChiselAnnotation, RunFirrtlTransform, dontTouch}
import chisel3.experimental.{MultiIOModule, RawModule, annotate}
import chisel3.libs.aspect.{ModuleAspect, ModuleAspectAnnotation}
import firrtl.{AnnotationSeq, CircuitForm, CircuitState, HighForm, MidForm, RenameMap, Transform}
import firrtl.annotations._
import firrtl.ir.{Input => _, Module => _, Output => _, _}
import firrtl.passes.wiring.WiringInfo

import scala.collection.mutable

abstract class CMR { def apply[T<:Data](ref: T): T }

/**
  *
  */
object TransactionEvent {
  /**
    *
    * @param name Name of the hardware breakpoint instance
    * @param root Location where the breakpoint will live
    * @param f Function to build breakpoint hardware
    * @tparam T Type of the root hardware
    * @return TransactionEvent annotation
    */
  def apply[T<: BaseModule](name: String, root: T, f: (T, CMR) => Data): Seq[ChiselAnnotation] = {

    // Elaborate breakpoint
    val (chiselIR, dut) = Driver.elaborateAndReturn(() => new TransactionEventModule(name, root, f))

    val otherPorts = root match {
      case r: MultiIOModule => Seq((r.clock, dut.clock), (r.reset, dut.reset))
    }

    // Build FIRRTL AST
    val firrtlString = chisel3.internal.firrtl.Emitter.emit(chiselIR)
    val firrtlIR = firrtl.Parser.parse(firrtlString)
    val firrtlModule = firrtlIR.modules.head

    // Build Names for references
    val circuitName = CircuitName(root.circuitName)
    val moduleName = ModuleName(root.name, circuitName)
    def toNamed(ref: Data): ComponentName = ComponentName(ref.pathTo(root).mkString("."), moduleName)

    // Return Annotations
    Seq(
      ModuleAspectAnnotation((dut.cmrs.toSeq ++ otherPorts).map{ case (from, to) => (toNamed(from), ComponentName(name + "." + to.toNamed.name, moduleName))}, root.toNamed, DefInstance(NoInfo, name, firrtlModule.name), firrtlModule)
    ) ++ chiselIR.annotations
  }
}

private class TransactionEventModule[T<:BaseModule](name: String, module: T, f: (T, CMR) => Data) extends MultiIOModule {
  val ios = mutable.ArrayBuffer[Data]()
  val cmrs = mutable.HashMap[Data, Data]()
  val refs = mutable.HashSet[Data]()
  class MyCMR extends CMR {
    def apply[T<:Data](ref: T): T =
      if(!refs.contains(ref)) {
        val x = IO(Input(chiselTypeOf(ref)))
        ios += x
        cmrs += ((ref, x))
        refs += ref
        x
      } else cmrs(ref).asInstanceOf[T]
  }
  object X { val y = f(module, new MyCMR()) }
  val flag = IO(Output(chiselTypeOf(X.y))).suggestName("flag")

  dontTouch(flag)
  flag := X.y
}

