// SPDX-License-Identifier: Apache-2.0

package firrtl.annotations

import firrtl.{
  MemoryArrayInit,
  MemoryEmissionOption,
  MemoryFileInlineInit,
  MemoryInitValue,
  MemoryRandomInit,
  MemoryScalarInit
}

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
  override def initValue:    MemoryInitValue = MemoryRandomInit
  override def isRandomInit: Boolean = true
}

/** Initialize all entries of the `target` memory with the scalar `value`. */
case class MemoryScalarInitAnnotation(target: ReferenceTarget, value: BigInt) extends MemoryInitAnnotation {
  override def duplicate(n: ReferenceTarget): Annotation = copy(n)
  override def initValue:    MemoryInitValue = MemoryScalarInit(value)
  override def isRandomInit: Boolean = false
}

/** Initialize the `target` memory with the array of `values` which must be the same size as the memory depth. */
case class MemoryArrayInitAnnotation(target: ReferenceTarget, values: Seq[BigInt]) extends MemoryInitAnnotation {
  override def duplicate(n: ReferenceTarget): Annotation = copy(n)
  override def initValue:    MemoryInitValue = MemoryArrayInit(values)
  override def isRandomInit: Boolean = false
}

/** Initialize the `target` memory with inline readmem[hb] statement. */
case class MemoryFileInlineAnnotation(
  target:      ReferenceTarget,
  filename:    String,
  hexOrBinary: MemoryLoadFileType.FileType = MemoryLoadFileType.Hex)
    extends MemoryInitAnnotation {
  require(filename.trim.nonEmpty, "empty filename not allowed in MemoryFileInlineAnnotation")
  override def duplicate(n: ReferenceTarget): Annotation = copy(n)
  override def initValue:    MemoryInitValue = MemoryFileInlineInit(filename, hexOrBinary)
  override def isRandomInit: Boolean = false
}

/** Initializes the memory inside the `ifndef SYNTHESIS` block (default) */
case object MemoryNoSynthInit extends NoTargetAnnotation

/** Initializes the memory outside the `ifndef SYNTHESIS` block */
case object MemorySynthInit extends NoTargetAnnotation
