package chisel3.libs

import chisel3._
import chisel3.core.{BaseModule, ChiselAnnotation, RunFirrtlTransform, dontTouch}
import chisel3.experimental.{MultiIOModule, RawModule, annotate}
import BreakPoint.BreakPointAnnotation
import firrtl.{AnnotationSeq, CircuitForm, CircuitState, HighForm, MidForm, RenameMap, Transform}
import firrtl.annotations._
import firrtl.ir.{Input => _, Module => _, Output => _, _}
import firrtl.passes.wiring.WiringInfo

import scala.collection.mutable


abstract class CMR { def apply[T<:Data](ref: T): T }

/**
  *
  */
object BreakPoint {
  class BreakPointTransform extends firrtl.Transform {
    override def inputForm: CircuitForm = MidForm
    override def outputForm: CircuitForm = HighForm

    def onStmt(bps: Seq[BreakPointAnnotation])(s: Statement): Statement = {
      if(bps.isEmpty) s else {
        val newBody = mutable.ArrayBuffer[Statement]()
        newBody += s
        newBody ++= bps.map(_.instance)
        Block(newBody)
      }
    }

    override def execute(state: CircuitState): CircuitState = {
      val bps = state.annotations.collect{ case b: BreakPointAnnotation => b}
      val bpMap = bps.groupBy(_.enclosingModule.name)

      val moduleMap = state.circuit.modules.map(m => m.name -> m).toMap

      val newModules = state.circuit.modules.flatMap {
        case m: firrtl.ir.Module =>
          val myBreakPoints = bpMap.getOrElse(m.name, Nil)
          val newM = m mapStmt onStmt(myBreakPoints)
          newM +: myBreakPoints.map(_.module)
        case x: ExtModule => Seq(x)
      }

      val newCircuit = state.circuit.copy(modules = newModules)
      val newAnnotations = state.annotations.filter(!_.isInstanceOf[BreakPointAnnotation])

      val wiringInfos = bps.flatMap{ case bp@BreakPointAnnotation(refs, enclosingModule, instance, module) =>
        refs.map{ case (from, to) =>
          WiringInfo(from, Seq(to), from.name)
        }
      }

      val transforms = Seq(
        aspect.ToIR,
        new firrtl.ChirrtlToHighFirrtl,
        new firrtl.IRToWorkingIR,
        firrtl.passes.CheckHighForm,
        firrtl.passes.ResolveKinds,
        firrtl.passes.InferTypes,
        firrtl.passes.CheckTypes,
        firrtl.passes.Uniquify,
        firrtl.passes.ResolveKinds,
        firrtl.passes.InferTypes,
        firrtl.passes.ResolveGenders,
        firrtl.passes.CheckGenders,
        firrtl.passes.InferWidths,
        new firrtl.passes.wiring.Wiring(wiringInfos.toSeq),
        new firrtl.IRToWorkingIR,
        new firrtl.ResolveAndCheck
      )

      val finalState = transforms.foldLeft(state.copy(annotations = newAnnotations, circuit = newCircuit)) { (newState, xform) =>
        val x = xform.runTransform(newState)
        //println(s"===${xform.getClass.getSimpleName}===")
        //println(x.circuit.serialize)
        x
      }
      //println(finalState.circuit.serialize)
      finalState
    }
  }

  case class BreakPointAnnotation(refs: Seq[(ComponentName, ComponentName)], enclosingModule: ModuleName, instance: DefInstance, module: firrtl.ir.DefModule) extends Annotation with RunFirrtlTransform {
    override def toFirrtl: Annotation = this
    override def transformClass: Class[_ <: Transform] = classOf[BreakPoint.BreakPointTransform]
    private val errors = mutable.ArrayBuffer[String]()
    private def rename[T<:Named](n: T, renames: RenameMap): T = (n, renames.get(n)) match {
      case (m: ModuleName, Some(Seq(x: ModuleName))) => x.asInstanceOf[T]
      case (c: ComponentName, Some(Seq(x: ComponentName))) => x.asInstanceOf[T]
      case (_, None) => n
      case (_, other) =>
        errors += s"Bad rename in ${this.getClass}: $n to $other"
        n
    }
    override def update(renames: RenameMap): Seq[Annotation] = {
      val newRefs = refs.map { case (from, to) => (rename(from, renames), rename(to, renames)) }
      val newEncl = rename(enclosingModule, renames)
      if(errors.nonEmpty) {
        throw new Exception(errors.mkString("\n"))
      }
      Seq(BreakPointAnnotation(newRefs, newEncl, instance, module))
    }
  }

  private class BreakPointModule[T<:BaseModule](name: String, module: T, f: (T, CMR) => Data) extends MultiIOModule {
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

  /**
    *
    * @param name Name of the hardware breakpoint instance
    * @param root Location where the breakpoint will live
    * @param f Function to build breakpoint hardware
    * @tparam T Type of the root hardware
    * @return BreakPoint annotation
    */
  def apply[T<: BaseModule](name: String, root: T, f: (T, CMR) => Data): Seq[ChiselAnnotation] = {

    // Elaborate breakpoint
    val (chiselIR, dut) = Driver.elaborateAndReturn(() => new BreakPointModule(name, root, f))

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
      BreakPointAnnotation((dut.cmrs.toSeq ++ otherPorts).map{ case (from, to) => (toNamed(from), ComponentName(name + "." + to.toNamed.name, moduleName))}, root.toNamed, DefInstance(NoInfo, name, firrtlModule.name), firrtlModule)
    ) ++ chiselIR.annotations
  }

}

