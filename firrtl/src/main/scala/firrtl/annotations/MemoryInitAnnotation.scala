// SPDX-License-Identifier: Apache-2.0

package firrtl.annotations

/**
  * Represents the initial value of the annotated memory.
  * While not supported on normal ASIC flows, it can be useful for simulation and FPGA flows.
  * This annotation is consumed by the verilog emitter.
  */
sealed trait MemoryInitAnnotation extends SingleTargetAnnotation[ReferenceTarget] {
  def isRandomInit: Boolean
}

/** Initialize the `target` memory with inline readmem[hb] statement. */
case class MemoryFileInlineAnnotation(
  target:      ReferenceTarget,
  filename:    String,
  hexOrBinary: MemoryLoadFileType.FileType = MemoryLoadFileType.Hex)
    extends MemoryInitAnnotation {
  require(filename.trim.nonEmpty, "empty filename not allowed in MemoryFileInlineAnnotation")
  override def duplicate(n: ReferenceTarget): Annotation = copy(n)
  override def isRandomInit: Boolean = false
}
