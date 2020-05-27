// See LICENSE for license details.

package chiselTests

import chisel3.RawModule
import chisel3.aop.Aspect
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage, NoRunFirrtlCompilerAnnotation}
import firrtl.AnnotationSeq

import scala.reflect.ClassTag

package object naming {
  def run[T <: RawModule](gen: () => T, annotations: AnnotationSeq): AnnotationSeq = {
    new ChiselStage().run(Seq(ChiselGeneratorAnnotation(gen), NoRunFirrtlCompilerAnnotation) ++ annotations)
  }
  def aspectTest[T <: RawModule](gen: () => T)(f: T => Unit)(implicit ctag: ClassTag[T]): Unit = {
    case object BuiltAspect extends Aspect[T] {
      override def toAnnotation(top: T): AnnotationSeq = {f(top); Nil}
    }
    BuiltAspect
    run(gen, Seq(BuiltAspect))
  }
}
