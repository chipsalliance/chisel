package chisel3.aop

import chisel3.core
import chisel3.core.RunFirrtlTransform
import chisel3.internal.{Builder, HasId}
import chisel3.internal.firrtl.DefModule
import firrtl.annotations._
import firrtl._
import firrtl.ir
import firrtl.Mappers._

import scala.collection.mutable
import scala.reflect.runtime.universe.TypeTag

abstract class InjectingConcern[T <: core.RawModule, R <: Aspect[T, _]](implicit tag: TypeTag[T]) extends Concern[T, R] {
  def aspects: Seq[R]
  override def transformClass: Class[_ <: Transform] = classOf[InjectingTransform]
}

case class InjectingAspect[DUT <: core.RawModule, M <: core.RawModule]
    (selectRoot: DUT => M, injection: M => Unit)
    (implicit tag: TypeTag[DUT]) extends Aspect[DUT, M](selectRoot) {

  def toAnnotation(dut: DUT): AnnotationSeq = {
    Seq(toAnnotation(selectRoot(dut), injection))
  }

  def toAnnotation(module: M, inject: M => Unit): InjectStatement = {
    val chiselIR = Builder.build(core.Module(new core.ModuleAspect(module) {
      module match {
        case x: core.MultiIOModule => core.withClockAndReset(x.clock, x.reset) { inject(module) }
        case x: core.RawModule => inject(module)
      }
    }))
    val comps = chiselIR.components.map {
      case x: DefModule if x.name == module.name => x.copy(id = module)
      case other => other
    }

    val annotations = chiselIR.annotations.map(_.toFirrtl).filterNot{ a => a.isInstanceOf[DesignAnnotation[_]] }
    val runFirrtls = annotations.collect {
      case r: RunFirrtlTransform =>
        s"Cannot annotate an aspect with a RunFirrtlTransform annotation: $r"
    }
    assert(runFirrtls.isEmpty, runFirrtls.mkString("\n"))


    val stmts = mutable.ArrayBuffer[ir.Statement]()
    val modules = Aspect.getFirrtl(chiselIR.copy(components = comps)).flatMap {
      case m: firrtl.ir.Module if m.name == module.name =>
        stmts += m.body
        Nil
      case other =>
        Seq(other)
    }

    InjectStatement(module.name, ir.Block(stmts), modules, annotations)
  }
}

case class InjectStatement(module: String, s: firrtl.ir.Statement, modules: Seq[firrtl.ir.DefModule], annotations: Seq[Annotation]) extends NoTargetAnnotation

class InjectingTransform extends ConcernTransform {
  override def inputForm: CircuitForm = ChirrtlForm
  override def outputForm: CircuitForm = ChirrtlForm

  override def execute(state: CircuitState): CircuitState = {

    val addStmtMap = mutable.HashMap[String, Seq[ir.Statement]]()
    val addModules = mutable.ArrayBuffer[ir.DefModule]()
    val newAnnotations = state.annotations.flatMap {
      case InjectStatement(module, s, addedModules, annotations) =>
        addModules ++= addedModules
        addStmtMap(module) = s +: addStmtMap.getOrElse(module, Nil)
        annotations
      case other => Seq(other)
    }
    //addStmtMap.foreach(println)

    val newModules = state.circuit.modules.map { m: ir.DefModule =>
      m match {
        case m: ir.Module if addStmtMap.contains(m.name) =>
          val newM = m.copy(body = ir.Block(m.body +: addStmtMap(m.name)))
          //println(newM.serialize)
          newM
        case m: _root_.firrtl.ir.ExtModule if addStmtMap.contains(m.name) =>
          ir.Module(m.info, m.name, m.ports, ir.Block(addStmtMap(m.name)))
        case other: ir.DefModule => other
      }
    }

    val newCircuit = state.circuit.copy(modules = newModules ++ addModules)

    println("Injecting Transform")
    println("Starting Annotations:")
    state.annotations.foreach(println)
    println("Ending Annotations:")
    newAnnotations.foreach(println)


    state.copy(annotations = newAnnotations, circuit = newCircuit)
  }
}
