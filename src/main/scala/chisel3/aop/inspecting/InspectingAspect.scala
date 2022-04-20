// SPDX-License-Identifier: Apache-2.0

package chisel3.aop.inspecting

import chisel3.RawModule
import chisel3.aop.Aspect
import firrtl.AnnotationSeq

/** Use for inspecting an elaborated design and printing out results
  *
  * @param inspect Given top-level design, print things and return nothing
  * @tparam T Type of top-level module
  */
case class InspectingAspect[T <: RawModule](inspect: T => Unit) extends InspectorAspect[T](inspect)

/** Extend to make custom inspections of an elaborated design and printing out results
  *
  * @param inspect Given top-level design, print things and return nothing
  * @tparam T Type of top-level module
  */
abstract class InspectorAspect[T <: RawModule](inspect: T => Unit) extends Aspect[T] {
  override def toAnnotation(top: T): AnnotationSeq = {
    inspect(top)
    Nil
  }
}
