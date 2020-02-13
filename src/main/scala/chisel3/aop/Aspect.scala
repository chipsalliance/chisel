// See LICENSE for license details.

package chisel3.aop

import chisel3.RawModule
import chisel3.stage.DesignAnnotation
import firrtl.AnnotationSeq
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.{Phase, Unserializable}

import scala.collection.mutable

/** Represents an aspect of a Chisel module, by specifying
  * what behavior should be done to instance, via the FIRRTL Annotation Mechanism
  * @tparam T Type of top-level module
  */
abstract class Aspect[T <: RawModule] extends Annotation with Unserializable with NoTargetAnnotation /*with Phase*/ {

  /*
  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    var dut: Option[RawModule] = None

    val remainingAnnotations = annotations.flatMap {
      case DesignAnnotation(d) =>
        dut = Some(d)
        Nil
      case other => Seq(other)
    }
    if(dut.isDefined) {
      val top = dut.get.asInstanceOf[T]
      val newAnnotations = toAnnotation(top) ++ resolveAspect(top, annotations)
      remainingAnnotations ++ newAnnotations
    } else annotations
  }
   */

  /** Convert this Aspect to a seq of FIRRTL annotation
    * @param top
    * @return
    */
  @deprecated
  def toAnnotation(top: T): AnnotationSeq

  /** Called by [[chisel3.stage.phases.AspectPhase]] to resolve this Aspect into annotations
    * @param top
    * @return
    */
  private[chisel3] def resolveAspect(top: RawModule): AnnotationSeq = {
    //???
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
