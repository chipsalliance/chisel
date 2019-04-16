package chisel3.aop

import chisel3.core.{BaseModule, Data, Module, ModuleAspect, MultiIOModule, RawModule, RunFirrtlTransform, withClockAndReset}
import chisel3.internal.{Builder, HasId}
import chisel3.internal.firrtl.DefModule
import firrtl.{AnnotationSeq, RenameMap, Transform}
import firrtl.annotations._
import firrtl.ir.EmptyStmt

import scala.reflect.runtime.universe.TypeTag

object Aspect {
  def getFirrtl(chiselIR: chisel3.internal.firrtl.Circuit): Seq[firrtl.ir.DefModule] = {
    chisel3.internal.firrtl.Converter.convert(chiselIR).modules
  }
}

/**
  *
  * @param selectRoot Given top-level module, pick the module to apply the aspect (root module)
  * @param tag Needed for reasons
  * @tparam DUT Type of top-level module
  * @tparam M Type of root module (join point)
  */
abstract class Aspect[DUT <: RawModule, M <: RawModule](selectRoot: DUT => M)(implicit tag: TypeTag[DUT]) {
  def toAnnotation(dut: DUT): AnnotationSeq
}


abstract class AnnotatingAspect[DUT <: RawModule, M <: RawModule](selectRoot: DUT => M, selectSignals: M => Seq[Data])(implicit tag: TypeTag[DUT]) extends Aspect[DUT, M](selectRoot) {
  def toTargets(dut: DUT): Seq[ReferenceTarget] = selectSignals(selectRoot(dut)).map(_.toTarget)
  def toAnnotation(dut: DUT): AnnotationSeq
}


abstract class Concern[T <: RawModule, R <: Aspect[T, _]](implicit tag: TypeTag[T]) extends Annotation with RunFirrtlTransform {

  def aspects: Seq[R]

  def transformClass: Class[_ <: Transform]

  def toAnnotation(dut: T): AnnotationSeq = aspects.flatMap(_.toAnnotation(dut))

  def resolveAspects(dut: RawModule): AnnotationSeq = {
    aspects.flatMap(_.toAnnotation(dut.asInstanceOf[T]))
  }

  override def update(renames: RenameMap): Seq[Annotation] = Seq(this)
  override def toFirrtl: Annotation = this
}


