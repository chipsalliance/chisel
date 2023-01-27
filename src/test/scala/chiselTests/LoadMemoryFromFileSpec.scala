// SPDX-License-Identifier: Apache-2.0

package chiselTests

import java.io.File
import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import circt.stage.{CIRCTTarget, CIRCTTargetAnnotation, ChiselStage}
import chisel3.util.experimental.loadMemoryFromFile
import chisel3.util.{log2Ceil, Counter}
import firrtl.annotations.MemoryLoadFileType
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers
import firrtl.util.BackendCompilationUtilities.loggingProcessLogger
import scala.sys.process._

class UsesThreeMems(memoryDepth: Int, memoryType: Data) extends Module {
  val io = IO(new Bundle {
    val address = Input(UInt(memoryType.getWidth.W))
    val value1 = Output(memoryType)
    val value2 = Output(memoryType)
    val value3 = Output(memoryType)
  })

  val memory1 = Mem(memoryDepth, memoryType)
  val memory2 = Mem(memoryDepth, memoryType)
  val memory3 = Mem(memoryDepth, memoryType)
  loadMemoryFromFile(memory1, "./mem1")
  loadMemoryFromFile(memory2, "./mem1")
  loadMemoryFromFile(memory3, "./mem1")

  io.value1 := memory1(io.address)
  io.value2 := memory2(io.address)
  io.value3 := memory3(io.address)
}

class UsesMem(memoryDepth: Int, memoryType: Data) extends Module {
  val io = IO(new Bundle {
    val address = Input(UInt(memoryType.getWidth.W))
    val value = Output(memoryType)
    val value1 = Output(memoryType)
    val value2 = Output(memoryType)
  })

  val memory = Mem(memoryDepth, memoryType)
  loadMemoryFromFile(memory, "./mem1")

  io.value := memory(io.address)

  val low1 = Module(new UsesMemLow(memoryDepth, memoryType))
  val low2 = Module(new UsesMemLow(memoryDepth, memoryType))

  low2.io.address := io.address
  low1.io.address := io.address
  io.value1 := low1.io.value
  io.value2 := low2.io.value
}

class UsesMemLow(memoryDepth: Int, memoryType: Data) extends Module {
  val io = IO(new Bundle {
    val address = Input(UInt(memoryType.getWidth.W))
    val value = Output(memoryType)
  })

  val memory = Mem(memoryDepth, memoryType)

  loadMemoryFromFile(memory, "./mem2")

  io.value := memory(io.address)
}

class FileHasSuffix(memoryDepth: Int, memoryType: Data) extends Module {
  val io = IO(new Bundle {
    val address = Input(UInt(memoryType.getWidth.W))
    val value = Output(memoryType)
    val value2 = Output(memoryType)
  })

  val memory = Mem(memoryDepth, memoryType)

  loadMemoryFromFile(memory, "./mem1.txt")

  io.value := memory(io.address)

  val low = Module(new UsesMemLow(memoryDepth, memoryType))

  low.io.address := io.address
  io.value2 := low.io.value
}

class MemoryShape extends Bundle {
  val a = UInt(8.W)
  val b = SInt(8.W)
  val c = Bool()
}

class HasComplexMemory(memoryDepth: Int) extends Module {
  val io = IO(new Bundle {
    val address = Input(UInt(log2Ceil(memoryDepth).W))
    val value = Output(new MemoryShape)
  })

  val memory = Mem(memoryDepth, new MemoryShape)

  loadMemoryFromFile(memory, "./mem", MemoryLoadFileType.Hex)

  io.value := memory(io.address)
}

class HasBinarySupport(memoryDepth: Int, memoryType: Data) extends Module {
  val io = IO(new Bundle {
    val address = Input(UInt(memoryType.getWidth.W))
    val value = Output(memoryType)
  })

  val memory = Mem(memoryDepth, memoryType)

  loadMemoryFromFile(memory, "./mem", MemoryLoadFileType.Binary)

  io.value := memory(io.address)
}

/**
  * The following tests only lint the output and check that the output verilog is properly constructed
  *
  * TODO: Write executable tests, and update after direct CIRCT support
  */
class LoadMemoryFromFileSpec extends AnyFreeSpec with Matchers {
  def fileExistsWithMem(file: File, mem: Option[String] = None, line: Option[String] = None): File = {
    info(s"$file exists")
    file.exists() should be(true)
    mem.foreach(m => {
      info(s"Memory $m is referenced in $file")
      val found = io.Source.fromFile(file).getLines.exists { _.contains(s"""readmemh("$m"""") }
      found should be(true)
    })
    line.foreach(l => {
      info(s"Line $l is referenced in $file")
      val found = io.Source.fromFile(file).getLines.exists { _.contains(l) }
      found should be(true)
    })
    file
  }
  def lints(file: File, filelist: File): Unit = {
    require(file.exists(), s"${file.getName} should be emitted!")
    require(filelist.exists(), s"${file.getName} should be emitted!")
    val cmd = Seq("verilator", "--lint-only", "-sv", file.getAbsolutePath, "-f", filelist.getAbsolutePath())
    assert(cmd.!(loggingProcessLogger) == 0, "Generated Verilog is not valid.")
  }
  def cleanup(files: File*): Unit = files.foreach { _.delete() }
  def compile[T <: Module](testDirName: String, gen: () => T): Unit = {
    (new circt.stage.ChiselStage).execute(
      args = Array("--target", "systemverilog", "--target-dir", testDirName),
      annotations = Seq(ChiselGeneratorAnnotation(gen))
    )
  }

  "Users can specify a source file to load memory from" in {
    val testDirName = "test_run_dir/load_memory_spec"
    compile(testDirName, () => new UsesMem(memoryDepth = 8, memoryType = UInt(16.W)))

    val dir = new File(testDirName)
    val memV = fileExistsWithMem(new File(dir, "UsesMem.UsesMem.memory.v"), Some("./mem1"))
    val memLowV = fileExistsWithMem(new File(dir, "UsesMem.UsesMemLow.memory.v"), Some("./mem2"))
    val dut = new File(dir, "UsesMem.sv")
    val fileList = new File(dir, "load_memories_from_file.f")

    lints(dut, fileList)
    cleanup(
      memV,
      memLowV,
      fileList
    ) // Remove generated files so they don't stick around if iterating, and giving false positives
  }

  "Calling a module that loads memories from a file more than once should work" in {
    val testDirName = "test_run_dir/load_three_memory_spec"
    compile(testDirName, () => new UsesThreeMems(memoryDepth = 8, memoryType = UInt(16.W)))

    val dir = new File(testDirName)
    val mem1 = fileExistsWithMem(new File(dir, "UsesThreeMems.UsesThreeMems.memory1.v"), Some("./mem1"))
    val mem2 = fileExistsWithMem(new File(dir, "UsesThreeMems.UsesThreeMems.memory2.v"), Some("./mem1"))
    val mem3 = fileExistsWithMem(new File(dir, "UsesThreeMems.UsesThreeMems.memory3.v"), Some("./mem1"))
    val dut = new File(dir, "UsesThreeMems.sv")
    val fileList = new File(dir, "load_memories_from_file.f")

    lints(dut, fileList)
    cleanup(
      mem1,
      mem2,
      mem3,
      fileList
    ) // Remove generated files so they don't stick around if iterating, and giving false positives
  }

  "In this example the memory has a complex memory type containing a bundle" in {
    val testDirName = "test_run_dir/complex_memory_load"
    compile(testDirName, () => new HasComplexMemory(memoryDepth = 8))

    val dir = new File(testDirName)
    val mem = fileExistsWithMem(
      new File(dir, s"HasComplexMemory.HasComplexMemory.memory.v"),
      None,
      Some(s"""$$readmemh("./mem", HasComplexMemory.memory_ext.Memory);""")
    )
    val dut = new File(dir, "HasComplexMemory.sv")
    val fileList = new File(dir, "load_memories_from_file.f")

    lints(dut, fileList)
    cleanup(mem, fileList) // Remove generated files so they don't stick around if iterating, and giving false positives
  }

  "Has binary format support" in {
    val testDirName = "test_run_dir/binary_memory_load"
    compile(testDirName, () => new HasBinarySupport(memoryDepth = 8, memoryType = UInt(16.W)))

    val dir = new File(testDirName)
    val mem = fileExistsWithMem(
      new File(dir, s"HasBinarySupport.HasBinarySupport.memory.v"),
      None,
      Some(s"""$$readmemb("./mem", HasBinarySupport.memory_ext.Memory);""")
    )
    val dut = new File(dir, "HasBinarySupport.sv")
    val fileList = new File(dir, "load_memories_from_file.f")
    lints(dut, fileList)
    cleanup(mem, fileList)
  }
}
