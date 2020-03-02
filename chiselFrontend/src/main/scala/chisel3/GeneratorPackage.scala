package chisel3

import chisel3.experimental.BaseModule
import firrtl.AnnotationSeq
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.{Dependency, DependencyAPI, DependencyManager, Phase, PreservesAll, Stage, Unserializable}

abstract class GeneratorPackage[+T <: BaseModule] extends RawModule with NoTargetAnnotation with Unserializable {
  val top: T
  val annotations: AnnotationSeq
  val phase: GeneratorPackageCreator
  def find[X <: BaseModule](clazz: Class[X]): Seq[Template[X]]
}

trait GeneratorPackageCreator extends Phase with DependencyAPI[Phase] with PreservesAll[Phase] {
  val packge: String
  val version: String
  override def dependents = Seq(Dependency(GeneratorPackageResolution))
}

object GeneratorPackageResolution extends Phase with DependencyAPI[Phase] with PreservesAll[Phase] {
  override def transform(a: AnnotationSeq): AnnotationSeq = {
    ???
  }
}