// SPDX-License-Identifier: Apache-2.0

package chisel3.aop.injecting

import chisel3.{Module, ModuleAspect, RawModule, withClockAndReset}
import chisel3.aop._
import chisel3.experimental.hierarchy.{Definition, Instance, IsInstantiable}
import chisel3.internal.{Builder, DynamicContext}
import chisel3.internal.firrtl.DefModule
import chisel3.stage.DesignAnnotation
import firrtl.annotations.{ModuleTarget, Annotation}
import firrtl.stage.RunFirrtlTransformAnnotation
import firrtl.{ir, AnnotationSeq, _}

import scala.collection.mutable

case class InBody(inputAnnotations: Seq[Annotation], circuit: String, moduleNames: Seq[String]) {

  val injectionAnnotations = mutable.ArrayBuffer[Annotation]()

  def toAnnotations: Seq[Annotation] = RunFirrtlTransformAnnotation(new InjectingTransform) +: injectionAnnotations.toList 

  def apply[A <: RawModule](definition: Definition[A])(injection: Instance[A] => Unit): Unit = {
    val dynamicContext = new DynamicContext(inputAnnotations)
    moduleNames.foreach { n =>
      dynamicContext.globalNamespace.name(n)
    }

    val (chiselIR, _) = Builder.build(Module(new ModuleAspect(definition.getProto) {
      definition.getProto match {
        case x: Module => withClockAndReset(x.clock, x.reset) { injection(definition.toInstance) }
        case x: RawModule => injection(definition.toInstance)
      }
    }), dynamicContext)

    val comps = chiselIR.components.map {
      case x: DefModule if x.name == definition.getProto.name => x.copy(id = definition.getProto)
      case other => other
    }

    val annotations = chiselIR.annotations.map(_.toFirrtl).filterNot{ a => a.isInstanceOf[DesignAnnotation[_]] }

    /** Statements to be injected via aspect. */
    val stmts = mutable.ArrayBuffer[ir.Statement]()
    /** Modules to be injected via aspect. */
    val modules = Aspect.getFirrtl(chiselIR.copy(components = comps)).modules.flatMap {
      // for "container" modules, inject their statements
      case m: firrtl.ir.Module if m.name == definition.getProto.name =>
        stmts += m.body
        Nil
      // for modules to be injected
      case other: firrtl.ir.DefModule =>
        Seq(other)
    }
    injectionAnnotations += InjectStatement(ModuleTarget(circuit, definition.getProto.name), ir.Block(stmts.toSeq), modules, annotations)
  }
}

/** Aspect to inject Chisel code into a module of type M
  *
  * @param selectRoots Given top-level module, pick the instances of a module to apply the aspect (root module)
  * @param injection Function to generate Chisel hardware that will be injected to the end of module m
  *                  Signals in m can be referenced and assigned to as if inside m (yes, it is a bit magical)
  * @tparam T Type of top-level module
  * @tparam M Type of root module (join point)
  */
case class InjectingAspect2[T <: RawModule](
  func: ((T, InBody) => Unit)
) extends InjectorAspect2[T](func)
object InjectingAspect2 {

}

/** Extend to inject Chisel code into a module of type M
  *
  * @param selectRoots Given top-level module, pick the instances of a module to apply the aspect (root module)
  * @param injection Function to generate Chisel hardware that will be injected to the end of module m
  *                  Signals in m can be referenced and assigned to as if inside m (yes, it is a bit magical)
  * @tparam T Type of top-level module
  * @tparam M Type of root module (join point)
  */
abstract class InjectorAspect2[T <: RawModule](
  func: ((T, InBody) => Unit)
) extends Aspect[T] {
  final def toAnnotation(top: T): AnnotationSeq = {
    val moduleNames = Select.collectDeep(top) { case i => i.name }.toSeq
    val inBody = InBody(Nil, top.name, moduleNames)
    func(top, inBody)
    inBody.toAnnotations
  }
}


