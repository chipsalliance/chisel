// SPDX-License-Identifier: Apache-2.0

package chisel3.aop

import chisel3.RawModule
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.Unserializable
import firrtl.AnnotationSeq

/** Represents an aspect of a Chisel module, by specifying
  * what behavior should be done to instance, via the FIRRTL Annotation Mechanism
  * @tparam T Type of top-level module
  */
abstract class Aspect[T <: RawModule] extends Annotation with Unserializable with NoTargetAnnotation {

  /** variable to save [[AnnotationSeq]] from [[chisel3.stage.phases.AspectPhase]]
    * to be used at [[chisel3.aop.injecting.InjectorAspect]], exposes annotations to [[chisel3.internal.DynamicContext]]
    */
  private[aop] var annotationsInAspect: AnnotationSeq = Seq()

  /** Convert this Aspect to a seq of FIRRTL annotation
    * @param top
    * @return
    */
  def toAnnotation(top: T): AnnotationSeq

  /** Called by [[chisel3.stage.phases.AspectPhase]] to resolve this Aspect into annotations
    * @param top
    * @return
    */
  private[chisel3] def resolveAspect(top: RawModule, remainingAnnotations: AnnotationSeq): AnnotationSeq = {
    annotationsInAspect = remainingAnnotations
    toAnnotation(top.asInstanceOf[T])
  }
}

/** Holds utility functions for Aspect stuff */
object Aspect {

  /** Converts elaborated Chisel components to FIRRTL modules
    * @param chiselIR
    * @return
    */
  def getFirrtl(chiselIR: chisel3.internal.firrtl.Circuit): firrtl.ir.Circuit = {
    chisel3.internal.firrtl.Converter.convert(chiselIR)
  }
}
