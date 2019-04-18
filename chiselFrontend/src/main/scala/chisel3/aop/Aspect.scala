package chisel3.aop

import chisel3.core.RawModule
import firrtl.AnnotationSeq

import scala.reflect.runtime.universe.TypeTag

/** Represents an aspect of a Chisel module, by specifying
  *  (1) how to select a Chisel instance from the design
  *  (2) what behavior should be done to selected instance, via the FIRRTL Annotation Mechanism
  * @param selectRoot Given top-level module, pick the module to apply the aspect (root module)
  * @param tag Needed to prevent type-erasure of the top-level module type
  * @tparam DUT Type of top-level module
  * @tparam M Type of root module (join point)
  */
abstract class Aspect[DUT <: RawModule, M <: RawModule](selectRoot: DUT => M)(implicit tag: TypeTag[DUT]) {
  /** Convert this Aspect to a seq of FIRRTL annotation
    * @param dut
    * @return
    */
  def toAnnotation(dut: DUT): AnnotationSeq
}

/** Holds utility functions for Aspect stuff */
object Aspect {

  /** Converts elaborated Chisel components to FIRRTL modules
    * @param chiselIR
    * @return
    */
  def getFirrtl(chiselIR: chisel3.internal.firrtl.Circuit): Seq[firrtl.ir.DefModule] = {
    chisel3.internal.firrtl.Converter.convert(chiselIR).modules
  }
}
