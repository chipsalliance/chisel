package chiselTests.aop

import chisel3._
import chisel3.aop.{Aspect, Select}
import chisel3.experimental.hierarchy.Definition
import firrtl.AnnotationSeq
import scala.reflect.runtime.universe.TypeTag

object TestAspects {
  case class SelectAspect[T <: RawModule : TypeTag, X](selector: Definition[T] => Seq[X], desired: Definition[T] => Seq[X]) extends Aspect[T] {
    override def toAnnotation(top: T): AnnotationSeq = {
      val definition = top.toDefinition
      val results = selector(definition)
      val desiredSeq = desired(definition)
      assert(results.length == desiredSeq.length, s"Failure! Results $results have different length than desired $desiredSeq!")
      val mismatches = results.zip(desiredSeq).flatMap {
        case (res, des) if res != des => Seq((res, des))
        case other => Nil
      }
      assert(mismatches.isEmpty,s"Failure! The following selected items do not match their desired item:\n" + mismatches.map{
        case (res: Select.Serializeable, des: Select.Serializeable) => s"  ${res.serialize} does not match:\n  ${des.serialize}"
        case (res, des) => s"  $res does not match:\n  $des"
      }.mkString("\n"))
      Nil
    }
  }
}