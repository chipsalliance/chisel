// SPDX-License-Identifier: Apache-2.0

package chisel3.aop.injecting

import chisel3.{withClockAndReset, Module, ModuleAspect, RawModule}
import chisel3.aop._
import chisel3.internal.{Builder, DynamicContext}
import chisel3.internal.firrtl.DefModule
import chisel3.stage.{ChiselOptions, DesignAnnotation}
import firrtl.annotations.{Annotation, ModuleTarget}
import firrtl.options.Viewer.view
import firrtl.{ir, _}

import scala.collection.mutable

/** Aspect to inject Chisel code into a module of type M
  *
  * @param selectRoots Given top-level module, pick the instances of a module to apply the aspect (root module)
  * @param injection Function to generate Chisel hardware that will be injected to the end of module m
  *                  Signals in m can be referenced and assigned to as if inside m (yes, it is a bit magical)
  * @tparam T Type of top-level module
  * @tparam M Type of root module (join point)
  */
case class InjectingAspect[T <: RawModule, M <: RawModule](
  selectRoots: T => Iterable[M],
  injection:   M => Unit)
    extends InjectorAspect[T, M](
      selectRoots,
      injection
    )

/** Extend to inject Chisel code into a module of type M
  *
  * @param selectRoots Given top-level module, pick the instances of a module to apply the aspect (root module)
  * @param injection Function to generate Chisel hardware that will be injected to the end of module m
  *                  Signals in m can be referenced and assigned to as if inside m (yes, it is a bit magical)
  * @tparam T Type of top-level module
  * @tparam M Type of root module (join point)
  */
abstract class InjectorAspect[T <: RawModule, M <: RawModule](
  selectRoots: T => Iterable[M],
  injection:   M => Unit)
    extends Aspect[T] {
  final def toAnnotation(top: T): AnnotationSeq = {
    val moduleNames =
      Select.allDefinitionsOf[chisel3.experimental.BaseModule](top.toDefinition).map { i => i.toTarget.module }.toSeq
    toAnnotation(selectRoots(top), top.name, moduleNames)
  }

  /** Returns annotations which contain all injection logic
    *
    * @param modules The modules to inject into
    * @param circuit Top level circuit
    * @param moduleNames The names of all existing modules in the original circuit, to avoid name collisions
    * @return
    */
  final def toAnnotation(modules: Iterable[M], circuit: String, moduleNames: Seq[String]): AnnotationSeq = {
    modules.map { module =>
      val chiselOptions = view[ChiselOptions](annotationsInAspect)
      val dynamicContext =
        new DynamicContext(
          annotationsInAspect,
          chiselOptions.throwOnFirstError,
          chiselOptions.warningFilters,
          chiselOptions.sourceRoots
        )
      // Add existing module names into the namespace. If injection logic instantiates new modules
      //  which would share the same name, they will get uniquified accordingly
      moduleNames.foreach { n =>
        dynamicContext.globalNamespace.name(n)
      }

      val (chiselIR, _) = Builder.build(
        Module(new ModuleAspect(module) {
          module match {
            case x: Module    => withClockAndReset(x.clock, x.reset) { injection(module) }
            case x: RawModule => injection(module)
          }
        }),
        dynamicContext
      )

      val comps = chiselIR.components.map {
        case x: DefModule if x.name == module.name => x.copy(id = module)
        case other => other
      }

      val annotations: Seq[Annotation] = chiselIR.firrtlAnnotations.toSeq.filterNot { a =>
        a.isInstanceOf[DesignAnnotation[_]]
      }

      /** Statements to be injected via aspect. */
      val stmts = mutable.ArrayBuffer[ir.Statement]()

      /** Modules to be injected via aspect. */
      val modules = Aspect.getFirrtl(chiselIR.copy(components = comps)).modules.flatMap {
        // for "container" modules, inject their statements
        case m: firrtl.ir.Module if m.name == module.name =>
          stmts += m.body
          Nil
        // for modules to be injected
        case other: firrtl.ir.DefModule =>
          Seq(other)
      }

      InjectStatement(ModuleTarget(circuit, module.name), ir.Block(stmts.toSeq), modules, annotations)
    }.toSeq
  }
}
