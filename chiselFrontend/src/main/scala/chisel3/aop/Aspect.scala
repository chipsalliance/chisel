package chisel3
package aop

import chisel3.core.{BaseModule, Module, MultiIOModule, RawModule, RunFirrtlTransform}
import chisel3.internal.HasId
import chisel3.internal.firrtl.DefModule
import firrtl.{AnnotationSeq, RenameMap, Transform}
import firrtl.annotations._
import firrtl.ir.EmptyStmt

import scala.reflect.runtime.universe.TypeTag

case class AddStatements(module: String, s: firrtl.ir.Statement) extends NoTargetAnnotation

abstract class AspectInfo[DUT <: RawModule](implicit tag: TypeTag[DUT]) {
  def getRot(dut: DUT): RawModule
  def inject(dut: DUT): AnnotationSeq
  def toAnnotation: AnnotationSeq
}

abstract class Aspect[T <: RawModule, R](implicit tag: TypeTag[T]) extends Annotation with RunFirrtlTransform {
  abstract class AbstractModule(circuit: String, moduleName: String) extends MultiIOModule {
    override def circuitName: String = circuit
    override def desiredName:String = moduleName
  }

  def resolveAspectInfo(aspectInfo: R): AnnotationSeq

  def collectAspectInfo(dut: T): R

  def transformClass: Class[_ <: Transform]

  final def untypedResolveAspect(dut: RawModule): AnnotationSeq = {
    val typedDut = dut.asInstanceOf[T]
    //var annotations: AnnotationSeq = Seq.empty[Annotation]
    var aspectInfo: Option[R] = None
    val dutTarget = dut.toTarget
    val chiselIR = internal.Builder.build(Module(new AbstractModule(dutTarget.circuit, dut.name) {
      aspectInfo = Some(collectAspectInfo(typedDut))
    }))
    val annotations = resolveAspectInfo(aspectInfo.get)
    getFirrtl(chiselIR) match {
      case m: firrtl.ir.Module => AddStatements(dut.name, m.body) +: annotations
      case other => annotations
    }
  }

  override def update(renames: RenameMap): Seq[Annotation] = Seq(this)
  override def toFirrtl: Annotation = this

  def getFirrtl(chiselIR: chisel3.internal.firrtl.Circuit): firrtl.ir.DefModule = {
    internal.firrtl.Converter.convert(chiselIR).modules.head
  }
}
