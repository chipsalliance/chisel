// See LICENSE for license details.

package chisel3.aop.inspecting

import chisel3.RawModule
import chisel3.aop.Aspect
import firrtl.AnnotationSeq

import scala.reflect.runtime.universe.TypeTag

/** Use for inspecting an elaborated design and printing out results
  *
  * @param inspect Given top-level design, print things and return nothing
  * @param tTag
  * @tparam T Type of top-level module
  */
abstract class InspectingAspect[T <: RawModule](inspect: T => Unit)(implicit tTag: TypeTag[T]) extends Aspect[T] {
  override def toAnnotation(top: T): AnnotationSeq = {
    inspect(top)
    Nil
  }
}
