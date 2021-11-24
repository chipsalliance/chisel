// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.experimental.{RunFirrtlTransform, annotate, ChiselAnnotation}
import firrtl.annotations._
import firrtl.ir.{Module => _, _}
import firrtl.transforms.BlackBoxInlineAnno
import firrtl.Mappers._
import firrtl.{AnnotationSeq, CircuitForm, CircuitState, EmitCircuitAnnotation, LowForm, Transform, VerilogEmitter}

import scala.collection.mutable

/** [[loadMemoryFromFile]] is an annotation generator that helps with loading a memory from a text file inlined in
  * the Verilog module. This relies on Verilator and Verilog's `\$readmemh` or `\$readmemb`.
  * The [[https://github.com/freechipsproject/treadle Treadlebackend]] can also recognize this annotation and load memory at run-time.
  *
  * This annotation, when the FIRRTL compiler runs, triggers the [[MemoryFileInlineAnnotation]] that will add Verilog
  * directives inlined to the module enabling the specified memories to be initialized from files.
  * The module supports both `hex` and `bin` files by passing the appropriate [[MemoryLoadFileType.FileType]] argument with
  * [[MemoryLoadFileType.Hex]] or [[MemoryLoadFileType.Binary]]. Hex is the default.
  *
  * ==Example module==
  *
  * Consider a simple Module containing a memory:
  * {{{
  * import chisel3._
  * class UsesMem(memoryDepth: Int, memoryType: Data) extends Module {
  *   val io = IO(new Bundle {
  *     val address = Input(UInt(memoryType.getWidth.W))
  *     val value   = Output(memoryType)
  *   })
  *   val memory = Mem(memoryDepth, memoryType)
  *   io.value := memory(io.address)
  * }
  * }}}
  *
  * ==Above module with annotation==
  *
  * To load this memory from the file `/workspace/workdir/mem1.hex.txt` just add an import and annotate the memory:
  * {{{
  * import chisel3._
  * import chisel3.util.experimental.loadMemoryFromFile   // <<-- new import here
  * class UsesMem(memoryDepth: Int, memoryType: Data) extends Module {
  *   val io = IO(new Bundle {
  *     val address = Input(UInt(memoryType.getWidth.W))
  *     val value   = Output(memoryType)
  *   })
  *   val memory = Mem(memoryDepth, memoryType)
  *   io.value := memory(io.address)
  *   loadMemoryFromFile(memory, "/workspace/workdir/mem1.hex.txt")  // <<-- Note the annotation here
  * }
  * }}}
  *
  * ==Example file format==
  *
  * A memory file should consist of ASCII text in either hex or binary format. The following example shows such a
  * file formatted to use hex:
  * {{{
  *   0
  *   7
  *   d
  *  15
  * }}}
  *
  * A binary file can be similarly constructed.
  * Chisel does not validate the file format or existence. It is supposed to be in a path accessible by the synthesis
  * tool together with the generated Verilog.
  *
  * @see Chisel3 Wiki entry on
  * [[https://github.com/freechipsproject/chisel3/wiki/Chisel-Memories#loading-memories-in-simulation "Loading Memories
  * in Simulation"]]
  */
object loadMemoryFromFile {


  /** Annotate a memory such that it can be initialized inline using a file
    * @param memory the memory
    * @param fileName the file used for initialization
    * @param hexOrBinary whether the file uses a hex or binary number representation
    */
  def apply[T <: Data](
    memory: MemBase[T],
    fileName: String,
    hexOrBinary: MemoryLoadFileType.FileType = MemoryLoadFileType.Hex
  ): Unit = {
    annotate(new ChiselAnnotation {
      override def toFirrtl = MemoryFileInlineAnnotation(memory.toTarget, fileName, hexOrBinary)
    })
  }
}
