package chisel3.aop

import chisel3.core.{AdditionalTransforms, RawModule}
import firrtl.annotations.Annotation
import firrtl.{AnnotationSeq, RenameMap, Transform}

import scala.reflect.runtime.universe.TypeTag

/** Top-level container for multiple Aspects of a given type A that apply to a given design of type DUT
  *
  * Is optionally linked to a transform, which processes the concern
  *   For example, it can add RTL if needed (see [[chisel3.aop.InjectingConcern]] which triggers
  *   [[chisel3.aop.InjectingTransform]] to inject chisel code into a module via the transform
  * @param tag Necessary to prevent type-erasure of DUT, the type of the top-level chisel module
  * @tparam DUT Type of top-level chisel module
  * @tparam A Type of this concern's aspect
  */
abstract class Concern[DUT <: RawModule, A <: Aspect[DUT, _]](implicit tag: TypeTag[DUT]) extends Annotation with AdditionalTransforms {

  /** All aspects associated with this concern
    *
    * Implemented by user of Concern Library for their design
    * @return
    */
  def aspects: Seq[A]

  /** Associated FIRRTL transformation that turns all aspects into their annotations
    * @return
    */
  final def transformClass: Class[_ <: ConcernTransform] = classOf[ConcernTransform]

  /** Associated FIRRTL transformations, which may be required to modify the design
    *
    * Implemented by Concern library writer
    * @return
    */
  def additionalTransformClasses: Seq[Class[_ <: Transform]]

  /** Convert this concern's aspects into annotations to pass to FIRRTL compilation
    *
    * For now, RunFirrtlTransform will not trigger running the associated transform.
    * The associated transform must be added explicitly
    *
    * @param dut Top-level elaborated design
    * @return
    */
  def toAnnotation(dut: DUT): AnnotationSeq = aspects.flatMap(_.toAnnotation(dut))

  /** Called by the FIRRTL transformation that consumes this concern
    * @param dut
    * @return
    */
  def resolveAspects(dut: RawModule): AnnotationSeq = {
    aspects.flatMap(_.toAnnotation(dut.asInstanceOf[DUT]))
  }

  override def update(renames: RenameMap): Seq[Annotation] = Seq(this)
  override def toFirrtl: Annotation = this
}


