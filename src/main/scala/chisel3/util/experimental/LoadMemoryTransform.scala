// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental

import chisel3._
import chisel3.experimental.{annotate, ChiselAnnotation}
import firrtl.annotations.{ComponentName, LoadMemoryAnnotation, MemoryFileInlineAnnotation, MemoryLoadFileType}

import scala.collection.mutable

/** This is the annotation created when using [[loadMemoryFromFile]], it records the memory, the load file
  * and the format of the file.
  * @param target        memory to load
  * @param fileName      name of input file
  * @param hexOrBinary   use \$readmemh or \$readmemb, i.e. hex or binary text input, default is hex
  */
private case class ChiselLoadMemoryAnnotation[T <: Data](
  target:      MemBase[T],
  fileName:    String,
  hexOrBinary: MemoryLoadFileType.FileType = MemoryLoadFileType.Hex)
    extends ChiselAnnotation {

  if (fileName.isEmpty) {
    throw new Exception(
      s"""LoadMemory from file annotations file empty file name"""
    )
  }

  def toFirrtl: LoadMemoryAnnotation = {
    val tx = target.toNamed.asInstanceOf[ComponentName]
    LoadMemoryAnnotation(tx, fileName, hexOrBinary, Some(tx.name))
  }
}

/** [[loadMemoryFromFile]] is an annotation generator that helps with loading a memory from a text file as a bind module. This relies on
  * Verilator and Verilog's `\$readmemh` or `\$readmemb`.
  *
  * This annotation, when a FIRRTL compiler runs will add Verilog directives to enable the specified memories to be
  * initialized from files.
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
  *
  * @see
  * [[https://github.com/freechipsproject/chisel3/tree/master/src/test/scala/chiselTests/LoadMemoryFromFileSpec.scala
  * LoadMemoryFromFileSpec.scala]] in the test suite for additional examples.
  * @see Chisel3 Wiki entry on
  * [[https://github.com/freechipsproject/chisel3/wiki/Chisel-Memories#loading-memories-in-simulation "Loading Memories
  * in Simulation"]]
  */
object loadMemoryFromFile {

  /** Annotate a memory such that it can be initialized using a file
    * @param memory the memory
    * @param filename the file used for initialization
    * @param hexOrBinary whether the file uses a hex or binary number representation
    */
  def apply[T <: Data](
    memory:      MemBase[T],
    fileName:    String,
    hexOrBinary: MemoryLoadFileType.FileType = MemoryLoadFileType.Hex
  ): Unit = {
    annotate(ChiselLoadMemoryAnnotation(memory, fileName, hexOrBinary))
  }
}

/** [[loadMemoryFromFileInline]] is an annotation generator that helps with loading a memory from a text file inlined in
  * the Verilog module. This relies on Verilator and Verilog's `\$readmemh` or `\$readmemb`.
  *
  * This annotation, when the FIRRTL compiler runs, triggers the `MemoryFileInlineAnnotation` that will add Verilog
  * directives inlined to the module enabling the specified memories to be initialized from files.
  * The module supports both `hex` and `bin` files by passing the appropriate `MemoryLoadFileType.FileType`` argument with
  * `MemoryLoadFileType.Hex` or `MemoryLoadFileType.Binary`. Hex is the default.
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
  * import chisel3.util.experimental.loadMemoryFromFileInline   // <<-- new import here
  * class UsesMem(memoryDepth: Int, memoryType: Data) extends Module {
  *   val io = IO(new Bundle {
  *     val address = Input(UInt(memoryType.getWidth.W))
  *     val value   = Output(memoryType)
  *   })
  *   val memory = Mem(memoryDepth, memoryType)
  *   io.value := memory(io.address)
  *   loadMemoryFromFileInline(memory, "/workspace/workdir/mem1.hex.txt")  // <<-- Note the annotation here
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
object loadMemoryFromFileInline {

  /** Annotate a memory such that it can be initialized inline using a file
    * @param memory the memory
    * @param fileName the file used for initialization
    * @param hexOrBinary whether the file uses a hex or binary number representation
    */
  def apply[T <: Data](
    memory:      MemBase[T],
    fileName:    String,
    hexOrBinary: MemoryLoadFileType.FileType = MemoryLoadFileType.Hex
  ): Unit = {
    annotate(new ChiselAnnotation {
      override def toFirrtl = MemoryFileInlineAnnotation(memory.toTarget, fileName, hexOrBinary)
    })
  }
}
