package chisel3.aop.injecting

import chisel3.aop.{Aspect, Concern, ConcernTransform}
import chisel3.core.RawModule
import firrtl.Transform

import scala.reflect.runtime.universe.TypeTag

/** Contains all injecting aspects for a given design-under-test
  *
  * @param tag Necessary to prevent type-erasure of DUT, the type of the top-level chisel module
  * @tparam DUT Type of top-level chisel module
  * @tparam A Type of this concern's aspect
  */
abstract class InjectingConcern[DUT <: RawModule, A <: InjectingAspect[DUT, _]]
    (implicit tag: TypeTag[DUT]) extends Concern[DUT, A] {
  def aspects: Seq[A]
  override def additionalTransformClasses: Seq[Class[_ <: Transform]] = Seq(classOf[InjectingTransform])
}

