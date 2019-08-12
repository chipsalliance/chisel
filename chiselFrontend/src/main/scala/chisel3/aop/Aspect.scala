// See LICENSE for license details.

package chisel3.aop

import chisel3.experimental.{ChiselAnnotation, RawModule}
import firrtl.annotations.Annotation
import firrtl.options.{HasShellOptions, OptionsException, ShellOption, Unserializable}
import firrtl.{AnnotationSeq, RenameMap, Transform}

import scala.reflect.runtime.universe.TypeTag

/** Represents an aspect of a Chisel module, by specifying
  *  what behavior should be done to instance, via the FIRRTL Annotation Mechanism
  * @param dutTag Needed to prevent type-erasure of the top-level module type
  * @tparam T Type of top-level module
  */
abstract class Aspect[T <: RawModule](implicit dutTag: TypeTag[T]) extends Annotation with Unserializable {
  /** Convert this Aspect to a seq of FIRRTL annotation
    * @param top
    * @return
    */
  def toAnnotation(top: T): AnnotationSeq

  /** Called by [[chisel3.stage.AspectStage]] to resolve this Aspect into annotations
    * @param top
    * @return
    */
  private[chisel3] def resolveAspect(top: RawModule): AnnotationSeq = {
    toAnnotation(top.asInstanceOf[T])
  }

  override def update(renames: RenameMap): Seq[Annotation] = Seq(this)
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
