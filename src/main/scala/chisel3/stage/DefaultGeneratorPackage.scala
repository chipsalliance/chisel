package chisel3.stage

import chisel3.aop.Select
import chisel3.experimental.BaseModule
import chisel3.{GeneratorPackage, GeneratorPackageCreator, RawModule, Template}
import firrtl.AnnotationSeq

abstract class DefaultGeneratorPackageCreator[T <: BaseModule] extends GeneratorPackageCreator {
  def gen(): T
  def createPackage(top: T, annos: AnnotationSeq): GeneratorPackage[T]
  override def transform(annos: AnnotationSeq): AnnotationSeq = {
    val retAnnos = new ChiselStage().run(ChiselGeneratorAnnotation(gen _) +: annos)
    val (packges, rest) = retAnnos.partition { _.isInstanceOf[GeneratorPackage[_]] }
    val top = rest.collectFirst {
      case DesignAnnotation(top) => top.asInstanceOf[T]
    }.get
    packges :+ createPackage(top, rest)
  }
}

abstract class MultipleGeneratorPackageCreator[T <: BaseModule] extends GeneratorPackageCreator {
  def createGenerators(packages: Seq[GeneratorPackage[BaseModule]]): Seq[() => T]
  def createPackage(top: T, annos: AnnotationSeq): DefaultGeneratorPackage[T]
  override def transform(annos: AnnotationSeq): AnnotationSeq = {
    val packages = annos.collect { case x: GeneratorPackage[_] => x }
    val gens = createGenerators(packages.map(_.asInstanceOf[GeneratorPackage[BaseModule]]))
    gens.map { gen =>
      val retAnnos = new ChiselStage().run(NoRunFirrtlCompilerAnnotation +: ChiselGeneratorAnnotation(gen) +: annos)
      val rest = retAnnos.filterNot { _.isInstanceOf[GeneratorPackage[_]] }
      val top = rest.collectFirst {
        case DesignAnnotation(top) => top.asInstanceOf[T]
      }.get
      createPackage(top, rest)
    }
  }
}

case class DefaultGeneratorPackage[T <: BaseModule] (
    top: T,
    annotations: AnnotationSeq,
    phase: GeneratorPackageCreator
) extends GeneratorPackage[T] {
  def find[X <: BaseModule](clazz: Class[X]): Seq[Template[X]] =
    Select.collectDeep(top){case x if x.getClass == clazz => Template(x.asInstanceOf[X], Some(this)) }.toList
}

