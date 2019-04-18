package chisel3.aop.injecting

import chisel3.aop._
import chisel3.core.{RawModule, RunFirrtlTransform}
import chisel3.core
import chisel3.internal.Builder
import chisel3.internal.firrtl.DefModule
import firrtl.annotations.ModuleTarget
import firrtl.{ir, _}

import scala.collection.mutable
import scala.reflect.runtime.universe.TypeTag

/** Aspect to inject Chisel code into a module of type M
  *
  * @param selectRoot Given top-level module, pick the module to apply the aspect (root module)
  * @param injection Function to generate Chisel hardware that will be injected to the end of module m
  *                  Signals in m can be referenced and assigned to as if inside m (yes, it is a bit magical)
  * @param tag Needed to prevent type-erasure of the top-level module type
  * @tparam DUT Type of top-level module
  * @tparam M Type of root module (join point)
  */
case class InjectingAspect[DUT <: RawModule, M <: RawModule](selectRoot: DUT => M, injection: M => Unit)
                                                                (implicit tag: TypeTag[DUT]) extends Aspect[DUT, M](selectRoot) {
  final def toAnnotation(dut: DUT): AnnotationSeq = {
    Seq(toAnnotation(selectRoot(dut), injection, dut.name))
  }

  final def toAnnotation(module: M, inject: M => Unit, circuit: String): InjectStatement = {
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

    InjectStatement(ModuleTarget(circuit, module.name), ir.Block(stmts), modules, annotations)
  }
}

