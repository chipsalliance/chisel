// SPDX-License-Identifier: Apache-2.0

package chiselTests

import java.io.File
import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import circt.stage.{CIRCTTarget, CIRCTTargetAnnotation, ChiselStage, FirtoolOption}
import chisel3.util.experimental.{loadMemoryFromFile, loadMemoryFromFileInline}
import chisel3.util.log2Up
import firrtl.annotations.MemoryLoadFileType
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class Read[A <: Data](depth: Int, tpe: => A) extends Bundle {
  val en = Input(Bool())
  val data = Output(tpe)
  val addr = Input(UInt(log2Up(depth).W))
  def connectToMemory(mem: chisel3.Mem[A]): Unit = {
    data := DontCare
    when(en) {
      data := mem.read(addr)
    }
  }
}

class UsesThreeMems(memoryDepth: Int, memoryType: Data) extends Module {
  val io = Seq.fill(3)(IO(new Read(memoryDepth, memoryType)))

  val memory1 = Mem(memoryDepth, memoryType)
  val memory2 = Mem(memoryDepth, memoryType)
  val memory3 = Mem(memoryDepth, memoryType)
  loadMemoryFromFile(memory1, "./mem1")
  loadMemoryFromFile(memory2, "./mem1")
  loadMemoryFromFile(memory3, "./mem1")

  io(0).connectToMemory(memory1)
  io(1).connectToMemory(memory2)
  io(2).connectToMemory(memory3)
}

class UsesThreeMemsInline(
  memoryDepth: Int,
  memoryType:  Data,
  memoryFile:  String,
  hexOrBinary: MemoryLoadFileType.FileType)
    extends Module {
  val io = Seq.fill(3)(IO(new Read(memoryDepth, memoryType)))

  val memory1 = Mem(memoryDepth, memoryType)
  val memory2 = Mem(memoryDepth, memoryType)
  val memory3 = Mem(memoryDepth, memoryType)
  loadMemoryFromFileInline(memory1, memoryFile, hexOrBinary)
  loadMemoryFromFileInline(memory2, memoryFile, hexOrBinary)
  loadMemoryFromFileInline(memory3, memoryFile, hexOrBinary)

  io(0).connectToMemory(memory1)
  io(1).connectToMemory(memory2)
  io(2).connectToMemory(memory3)
}

class UsesMem(memoryDepth: Int, memoryType: Data) extends Module {
  val io = Seq.fill(3)(IO(new Read(memoryDepth, memoryType)))

  val memory_UsesMem = Mem(memoryDepth, memoryType)
  loadMemoryFromFile(memory_UsesMem, "./mem1")

  io(0).connectToMemory(memory_UsesMem)

  val low1 = Module(new UsesMemLow(memoryDepth, memoryType))
  val low2 = Module(new UsesMemLow(memoryDepth, memoryType))

  low1.io <> io(1)
  low2.io <> io(2)
}

class UsesMemLow(memoryDepth: Int, memoryType: Data) extends Module {
  val io = IO(new Read(memoryDepth, memoryType))

  val memory_UsesMemLow = Mem(memoryDepth, memoryType)

  loadMemoryFromFile(memory_UsesMemLow, "./mem2")

  io.connectToMemory(memory_UsesMemLow)
}

class FileHasSuffix(memoryDepth: Int, memoryType: Data) extends Module {
  val io = Seq.fill(2)(IO(new Read(memoryDepth, memoryType)))

  val memory = Mem(memoryDepth, memoryType)

  loadMemoryFromFile(memory, "./mem1.txt")

  io(0).connectToMemory(memory)

  val low = Module(new UsesMemLow(memoryDepth, memoryType))

  low.io := io(1)
}

class MemoryShape extends Bundle {
  val a = UInt(8.W)
  val b = SInt(8.W)
  val c = Bool()
}

class HasComplexMemory(memoryDepth: Int) extends Module {
  val io = IO(new Read(memoryDepth, new MemoryShape))

  val memory = Mem(memoryDepth, new MemoryShape)

  loadMemoryFromFile(memory, "./mem", MemoryLoadFileType.Hex)

  io.connectToMemory(memory)
}

class HasBinarySupport(memoryDepth: Int, memoryType: Data) extends Module {
  val io = IO(new Read(memoryDepth, memoryType))

  val memory = Mem(memoryDepth, memoryType)

  loadMemoryFromFile(memory, "./mem", MemoryLoadFileType.Binary)

  io.connectToMemory(memory)
}

/**
  * The following tests are a bit incomplete and check that the output verilog is properly constructed
  * For more complete working examples
  * @see <a href="https://github.com/freechipsproject/chisel-testers">Chisel Testers</a> LoadMemoryFromFileSpec.scala
  */
class LoadMemoryFromFileSpec extends AnyFreeSpec with Matchers {
  def fileExistsWithMem(file: File, mem: Option[String] = None): Unit = {
    info(s"$file exists")
    file should exist
    mem.foreach(m => {
      info(s"Memory $m is referenced in $file")
      val found = io.Source.fromFile(file).getLines().exists { _.contains(s"""readmemh("$m"""") }
      found should be(true)
    })
  }

  def deleteRecursively(path: File): Unit = path match {
    case dir if dir.isDirectory() =>
      dir.listFiles().foreach(deleteRecursively)
      dir.delete()
    case file => file.delete()
  }

  def compile(gen: => RawModule, targetDirName: File, splitVerilog: Boolean = true): Unit = {
    deleteRecursively(targetDirName)
    (new ChiselStage).execute(
      args = {
        val args = scala.collection.mutable
          .ArrayBuffer[String]("--target-dir", targetDirName.getPath(), "--target", "systemverilog")
        if (splitVerilog)
          args.append("--split-verilog")
        args.toArray
      },
      annotations = Seq(
        ChiselGeneratorAnnotation(() => gen),
        FirtoolOption("-disable-all-randomization"),
        FirtoolOption("-strip-debug-info")
      )
    )
  }

  "Users can specify a source file to load memory from" in {
    val dir = new File("test_run_dir/load_memory_spec")

    compile(new UsesMem(memoryDepth = 8, memoryType = UInt(16.W)), dir)

    fileExistsWithMem(new File(dir, "memory_UsesMem_8x16_init.sv"), Some("./mem1"))
    fileExistsWithMem(new File(dir, "memory_UsesMemLow_8x16_init.sv"), Some("./mem2"))

  }

  "Calling a module that loads memories from a file more than once should work" in {
    val dir = new File("test_run_dir/load_three_memory_spec")

    compile(new UsesThreeMems(memoryDepth = 8, memoryType = UInt(16.W)), dir)

    fileExistsWithMem(new File(dir, "memory_8x16_init.sv"), Some("./mem1"))

  }

  "In this example the memory has a complex memory type containing a bundle" in {
    val dir = new File("test_run_dir/complex_memory_load")

    compile(new HasComplexMemory(memoryDepth = 8), dir)

    val memoryElements = Seq("a", "b", "c")

    // MFC emits a single memory for a memory of aggregate type.
    fileExistsWithMem(new File(dir, "memory_8x17_init.sv"), Some("./mem"))

  }

  "Has binary format support" in {
    val dir = new File("test_run_dir/binary_memory_load")

    compile(new HasBinarySupport(memoryDepth = 8, memoryType = UInt(16.W)), dir)

    val file = new File(dir, s"memory_8x16_init.sv")
    file should exist

    val fileText = io.Source.fromFile(file).getLines().mkString("\n")
    fileText should include(s"""$$readmemb("./mem", memory_8x16.Memory);""")
  }

  "Module with more than one hex memory inline should work" in {
    val dir = new File("test_run_dir/load_three_memory_spec_inline_hex")

    compile(
      new UsesThreeMemsInline(memoryDepth = 8, memoryType = UInt(16.W), "./testmem.h", MemoryLoadFileType.Hex),
      dir,
      splitVerilog = false
    )

    val file = new File(dir, s"UsesThreeMemsInline.sv")
    file should exist

    val fileText = io.Source.fromFile(file).getLines().mkString("\n")
    fileText should include(s"""$$readmemh("./testmem.h", Memory);""")
  }

  "Module with more than one bin memory inline should work" in {
    val dir = new File("test_run_dir/load_three_memory_spec_inline_bin")

    compile(
      new UsesThreeMemsInline(memoryDepth = 8, memoryType = UInt(16.W), "testmem.bin", MemoryLoadFileType.Binary),
      dir,
      splitVerilog = false
    )

    val file = new File(dir, s"UsesThreeMemsInline.sv")
    file should exist

    val fileText = io.Source.fromFile(file).getLines().mkString("\n")
    fileText should include(s"""$$readmemb("testmem.bin", Memory);""")
  }
}
