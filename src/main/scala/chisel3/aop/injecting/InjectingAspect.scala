// See LICENSE for license details.

package chisel3.aop.injecting

import chisel3.{Module, ModuleAspect, experimental, withClockAndReset, RawModule, MultiIOModule}
import chisel3.aop._
import chisel3.internal.Builder
import chisel3.internal.firrtl.DefModule
import chisel3.stage.DesignAnnotation
import firrtl.annotations.ModuleTarget
import firrtl.stage.RunFirrtlTransformAnnotation
import firrtl.{ir, _}

import scala.collection.mutable
import scala.reflect.runtime.universe.TypeTag

/** Aspect to inject Chisel code into a module of type M
  *
  * @param selectRoots Given top-level module, pick the instances of a module to apply the aspect (root module)
  * @param injection Function to generate Chisel hardware that will be injected to the end of module m
  *                  Signals in m can be referenced and assigned to as if inside m (yes, it is a bit magical)
  * @param tTag Needed to prevent type-erasure of the top-level module type
  * @tparam T Type of top-level module
  * @tparam M Type of root module (join point)
  */
case class InjectingAspect[T <: RawModule,
                           M <: RawModule](selectRoots: T => Iterable[M],
                                           injection: M => Unit
                                          )(implicit tTag: TypeTag[T]) extends Aspect[T] {
  final def toAnnotation(top: T): AnnotationSeq = {
    toAnnotation(selectRoots(top), top.name)
  }

<<<<<<< HEAD
  final def toAnnotation(modules: Iterable[M], circuit: String): AnnotationSeq = {
=======
  /** Returns annotations which contain all injection logic
    *
    * @param modules The modules to inject into
    * @param circuit Top level circuit
    * @param moduleNames The names of all existing modules in the original circuit, to avoid name collisions
    * @return
    */
  final def toAnnotation(modules: Iterable[M], circuit: String, moduleNames: Seq[String]): AnnotationSeq = {
>>>>>>> 2c7264a6... fixing context bug (#1874)
    RunFirrtlTransformAnnotation(new InjectingTransform) +: modules.map { module =>
      val dynamicContext = new DynamicContext(annotationsInAspect)
      // Add existing module names into the namespace. If injection logic instantiates new modules
      //  which would share the same name, they will get uniquified accordingly
      moduleNames.foreach { n =>
        dynamicContext.globalNamespace.name(n)
      }

      val (chiselIR, _) = Builder.build(Module(new ModuleAspect(module) {
        module match {
          case x: MultiIOModule => withClockAndReset(x.clock, x.reset) { injection(module) }
          case x: RawModule => injection(module)
        }
      }))
      val comps = chiselIR.components.map {
        case x: DefModule if x.name == module.name => x.copy(id = module)
        case other => other
      }

      val annotations = chiselIR.annotations.map(_.toFirrtl).filterNot{ a => a.isInstanceOf[DesignAnnotation[_]] }

      /** Statements to be injected via aspect. */
      val stmts = mutable.ArrayBuffer[ir.Statement]()
      /** Modules to be injected via aspect. */
      val modules = Aspect.getFirrtl(chiselIR.copy(components = comps)).modules.flatMap {
        // for "container" modules, inject their statements
        case m: firrtl.ir.Module if m.name == module.name =>
          stmts += m.body
          Nil
<<<<<<< HEAD
        case other =>
=======
        // for modules to be injected
        case other: firrtl.ir.DefModule =>
>>>>>>> 2c7264a6... fixing context bug (#1874)
          Seq(other)
      }

      InjectStatement(ModuleTarget(circuit, module.name), ir.Block(stmts), modules, annotations)
    }.toSeq
  }
}

