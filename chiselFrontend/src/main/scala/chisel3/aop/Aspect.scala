package chisel3.aop

import chisel3.core.{BaseModule, Data, Module, ModuleAspect, MultiIOModule, RawModule, RunFirrtlTransform, withClockAndReset}
import chisel3.internal.{Builder, HasId}
import chisel3.internal.firrtl.DefModule
import firrtl.{AnnotationSeq, RenameMap, Transform}
import firrtl.annotations._
import firrtl.ir.EmptyStmt

import scala.reflect.runtime.universe.TypeTag

case class AddStatements(module: String, s: firrtl.ir.Statement) extends NoTargetAnnotation

object Aspect {

  def toAnnotation[T <: RawModule](module: T, inject: T => Unit): AddStatements = {
    val chiselIR = Builder.build(Module(new ModuleAspect(module) {
      inject(module)
    }))
    val comps = chiselIR.components.map {
      case x: DefModule if x.name == module.name => x.copy(id = module)
      case other => other
    }
    getFirrtl(chiselIR.copy(components = comps)) match {
      case m: firrtl.ir.Module => AddStatements(module.name, m.body)
      case other => sys.error(s"Got $other, was expected a Module!")
    }
  }

  def getFirrtl(chiselIR: chisel3.internal.firrtl.Circuit): firrtl.ir.DefModule = {
    chisel3.internal.firrtl.Converter.convert(chiselIR).modules.head
  }
}

abstract class Aspect[DUT <: RawModule, M <: RawModule](selectRoot: DUT => M)(implicit tag: TypeTag[DUT]) {
  def toAnnotation(dut: DUT): Annotation
}

case class InjectingAspect[DUT <: RawModule, M <: RawModule](selectRoot: DUT => M, injection: M => Unit)(implicit tag: TypeTag[DUT]) extends Aspect[DUT, M](selectRoot) {
  def toAnnotation(dut: DUT): Annotation = {
    //selectRoot(dut) match {
    //  case x: MultiIOModule => Aspect.toAnnotation(selectRoot(dut), injection)
    //  case x: RawModule => Aspect.toAnnotation(selectRoot(dut), injection)
    //}
    Aspect.toAnnotation(selectRoot(dut), injection)
  }
}

abstract class AnnotatingAspect[DUT <: RawModule, M <: RawModule](selectRoot: DUT => M, selectSignals: M => Seq[Data])(implicit tag: TypeTag[DUT]) extends Aspect[DUT, M](selectRoot) {
  def toTargets(dut: DUT): Seq[ReferenceTarget] = selectSignals(selectRoot(dut)).map(_.toTarget)
  def toAnnotation(dut: DUT): Annotation
}

abstract class Concern[T <: RawModule, R <: Aspect[T, _]](implicit tag: TypeTag[T]) extends Annotation with RunFirrtlTransform {

  def aspects: Seq[R]

  def resolveAspects(dut: RawModule): AnnotationSeq = {
    aspects.map(_.toAnnotation(dut.asInstanceOf[T]))
  }

  def transformClass: Class[_ <: Transform] = classOf[Transform]

  override def update(renames: RenameMap): Seq[Annotation] = Seq(this)
  override def toFirrtl: Annotation = this
}
