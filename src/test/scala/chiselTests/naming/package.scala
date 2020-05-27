// See LICENSE for license details.

package chiselTests

import chisel3.RawModule
import chisel3.aop.Aspect
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage, NoRunFirrtlCompilerAnnotation}
import firrtl.AnnotationSeq

package object naming {
  private def run[T <: RawModule](gen: () => T, annotations: AnnotationSeq): AnnotationSeq = {
    new ChiselStage().run(Seq(ChiselGeneratorAnnotation(gen), NoRunFirrtlCompilerAnnotation) ++ annotations)
  }

  /** A tester which runs generator and uses an aspect to check the returned object
    * @param gen function to generate a Chisel module
    * @param f a function to check the Chisel module
    * @tparam T the Chisel module class
    */
  def aspectTest[T <: RawModule](gen: () => T)(f: T => Unit): Unit = {
    case object BuiltAspect extends Aspect[T] {
      override def toAnnotation(top: T): AnnotationSeq = {f(top); Nil}
    }
    BuiltAspect
    run(gen, Seq(BuiltAspect))
  }
}
