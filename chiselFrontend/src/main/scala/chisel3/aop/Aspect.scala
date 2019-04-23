package chisel3.aop

import chisel3.core.{AdditionalTransforms, RawModule}
import firrtl.annotations.Annotation
import firrtl.{AnnotationSeq, RenameMap, Transform}

import scala.reflect.runtime.universe.TypeTag

/** Represents an aspect of a Chisel module, by specifying
  *  (1) how to select a Chisel instance from the design
  *  (2) what behavior should be done to selected instance, via the FIRRTL Annotation Mechanism
  * @param selectRoots Given top-level module, pick the instances of a module to apply the aspect (root module)
  * @param dutTag Needed to prevent type-erasure of the top-level module type
  * @param mTag Needed to prevent type-erasure of the selected modules' type
  * @tparam DUT Type of top-level module
  * @tparam M Type of root module (join point)
  */
abstract class Aspect[DUT <: RawModule, M <: RawModule](selectRoots: DUT => Seq[M])(implicit dutTag: TypeTag[DUT], mTag: TypeTag[M]) extends Annotation with AdditionalTransforms {
  /** Convert this Aspect to a seq of FIRRTL annotation
    * @param dut
    * @return
    */
  def toAnnotation(dut: DUT): AnnotationSeq

  /** Associated FIRRTL transformation that turns all aspects into their annotations
    * @return
    */
  final def transformClass: Class[_ <: AspectTransform] = classOf[AspectTransform]

  /** Associated FIRRTL transformations, which may be required to modify the design
    *
    * Implemented by Concern library writer
    * @return
    */
  def additionalTransformClasses: Seq[Class[_ <: Transform]]

  /** Called by the FIRRTL transformation that consumes this concern
    * @param dut
    * @return
    */
  def resolveAspect(dut: RawModule): AnnotationSeq = {
    toAnnotation(dut.asInstanceOf[DUT])
  }

  override def update(renames: RenameMap): Seq[Annotation] = Seq(this)
  override def toFirrtl: Annotation = this
}

/** Holds utility functions for Aspect stuff */
object Aspect {

  /** Converts elaborated Chisel components to FIRRTL modules
    * @param chiselIR
    * @return
    */
  def getFirrtl(chiselIR: chisel3.internal.firrtl.Circuit): Seq[firrtl.ir.DefModule] = {
    chisel3.internal.firrtl.Converter.convert(chiselIR).modules
  }
}
