package chisel3.libs.aspect

import chisel3._
import chisel3.core.{BaseModule, ChiselAnnotation, RunFirrtlTransform, dontTouch}
import chisel3.experimental.{MultiIOModule, RawModule}
import firrtl.annotations._
import firrtl.Mappers._
import firrtl.ir.{Input => _, Module => _, Output => _, _}
import firrtl.passes.Pass
import firrtl.passes.wiring.WiringInfo
import firrtl.{CircuitForm, CircuitState, HighForm, MidForm, RenameMap, Transform, WDefInstance, WRef, WSubAccess, WSubField, WSubIndex}

import scala.collection.mutable


object ToIR extends Pass {
  def toExp(e: Expression): Expression = e map toExp match {
    case ex: WRef => Reference(ex.name, ex.tpe)
    case ex: WSubField => SubField(ex.expr, ex.name, ex.tpe)
    case ex: WSubIndex => SubIndex(ex.expr, ex.value, ex.tpe)
    case ex: WSubAccess => SubAccess(ex.expr, ex.index, ex.tpe)
    case ex => ex // This might look like a case to use case _ => e, DO NOT!
  }

  def toStmt(s: Statement): Statement = s map toExp match {
    case sx: WDefInstance => DefInstance(sx.info, sx.name, sx.module)
    case sx => sx map toStmt
  }

  def run (c:Circuit): Circuit =
    c copy (modules = c.modules map (_ map toStmt))
}

private class ModuleAspectTransform extends firrtl.Transform {
  override def inputForm: CircuitForm = MidForm
  override def outputForm: CircuitForm = HighForm

  def onStmt(bps: Seq[ModuleAspectAnnotation])(s: Statement): Statement = {
    if(bps.isEmpty) s else {
      val newBody = mutable.ArrayBuffer[Statement]()
      newBody += s
      newBody ++= bps.map(_.instance)
      Block(newBody)
    }
  }

  override def execute(state: CircuitState): CircuitState = {
    val bps = state.annotations.collect{ case b: ModuleAspectAnnotation => b}
    val bpMap = bps.groupBy(_.enclosingModule.name)

    val moduleMap = state.circuit.modules.map(m => m.name -> m).toMap

    val newModules = state.circuit.modules.flatMap {
      case m: firrtl.ir.Module =>
        val myModuleAspects = bpMap.getOrElse(m.name, Nil)
        val newM = m mapStmt onStmt(myModuleAspects)
        newM +: myModuleAspects.map(_.module)
      case x: ExtModule => Seq(x)
    }

    val newCircuit = state.circuit.copy(modules = newModules)
    val newAnnotations = state.annotations.filter(!_.isInstanceOf[ModuleAspectAnnotation])

    val wiringInfos = bps.flatMap{ case bp@ModuleAspectAnnotation(refs, enclosingModule, instance, module) =>
      refs.map{ case (from, to) =>
        WiringInfo(from, Seq(to), from.name)
      }
    }


    val transforms = Seq(
      ToIR,
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

case class ModuleAspectAnnotation(connections: Seq[(ComponentName, ComponentName)], enclosingModule: ModuleName, instance: DefInstance, module: firrtl.ir.DefModule) extends Annotation with RunFirrtlTransform {
  override def toFirrtl: Annotation = this
  override def transformClass: Class[_ <: Transform] = classOf[ModuleAspectTransform]
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
    val newConnections = connections.map { case (from, to) => (rename(from, renames), rename(to, renames)) }
    val newEncl = rename(enclosingModule, renames)
    if(errors.nonEmpty) {
      throw new Exception(errors.mkString("\n"))
    }
    Seq(ModuleAspectAnnotation(newConnections, newEncl, instance, module))
  }
}


object ModuleAspect {
  /**
    *
    * @param name Name of the hardware breakpoint instance
    * @param root Location where the breakpoint will live
    * @param f Function to build breakpoint hardware
    * @tparam T Type of the root hardware
    * @return ModuleAspect annotation
    */
  def apply[T<: BaseModule, S<:RawModule](instanceName: String, enclosingModule: T, aspect: () => S, connections: (T, S) => Map[Data, Data]): Seq[ChiselAnnotation] = {

    // Elaborate aspect
    val (chiselIR, dut) = Driver.elaborateAndReturn(aspect)

    val connects = connections(enclosingModule, dut)

    // Build FIRRTL AST for aspect
    val firrtlString = chisel3.internal.firrtl.Emitter.emit(chiselIR)
    val firrtlIR = firrtl.Parser.parse(firrtlString)
    val firrtlModule = firrtlIR.modules.head

    // Build Names for references
    val circuitName = CircuitName(enclosingModule.circuitName)
    val moduleName = ModuleName(enclosingModule.name, circuitName)
    def toNamed(ref: Data): ComponentName = {
      if(ref._parent.get.name == dut.name) {
        ComponentName(instanceName + "." + ref.toNamed.name, moduleName)
      } else {
        ComponentName(ref.pathTo(enclosingModule).mkString("."), moduleName)
      }
    }

    // Return Annotations
    Seq(
      ModuleAspectAnnotation(
        connects.toSeq.map{ case (from, to) => (toNamed(from), toNamed(to)) },
        enclosingModule.toNamed,
        DefInstance(NoInfo, instanceName, firrtlModule.name),
        firrtlModule
      )
    ) ++ chiselIR.annotations
  }

}

