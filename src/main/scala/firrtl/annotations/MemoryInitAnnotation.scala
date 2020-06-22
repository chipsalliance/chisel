// See LICENSE for license details.

package firrtl.annotations

import firrtl.{MemoryArrayInit, MemoryEmissionOption, MemoryInitValue, MemoryRandomInit, MemoryScalarInit}

/**
 * Represents the initial value of the annotated memory.
 * While not supported on normal ASIC flows, it can be useful for simulation and FPGA flows.
 * This annotation is consumed by the verilog emitter.
 */
sealed trait MemoryInitAnnotation extends SingleTargetAnnotation[ReferenceTarget] with MemoryEmissionOption {
  def isRandomInit: Boolean
}

/** Randomly initialize the `target` memory. This is the same as the default behavior. */
case class MemoryRandomInitAnnotation(target: ReferenceTarget) extends MemoryInitAnnotation {
  override def duplicate(n: ReferenceTarget): Annotation = copy(n)
  override def initValue: MemoryInitValue = MemoryRandomInit
  override def isRandomInit: Boolean = true
}

/** Initialize all entries of the `target` memory with the scalar `value`. */
case class MemoryScalarInitAnnotation(target: ReferenceTarget, value: BigInt) extends MemoryInitAnnotation {
  override def duplicate(n: ReferenceTarget): Annotation = copy(n)
  override def initValue: MemoryInitValue = MemoryScalarInit(value)
  override def isRandomInit: Boolean =  false
}

/** Initialize the `target` memory with the array of `values` which must be the same size as the memory depth. */
case class MemoryArrayInitAnnotation(target: ReferenceTarget, values: Seq[BigInt]) extends MemoryInitAnnotation {
  override def duplicate(n: ReferenceTarget): Annotation = copy(n)
  override def initValue: MemoryInitValue = MemoryArrayInit(values)
  override def isRandomInit: Boolean =  false
}