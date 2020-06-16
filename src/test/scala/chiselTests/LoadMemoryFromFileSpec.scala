// See LICENSE for license details.

package chiselTests

import java.io.File

import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.util.experimental.loadMemoryFromFile
import chisel3.util.log2Ceil
import firrtl.FirrtlExecutionSuccess
import firrtl.annotations.MemoryLoadFileType
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class UsesThreeMems(memoryDepth: Int, memoryType: Data) extends Module {
  val io = IO(new Bundle {
    val address = Input(UInt(memoryType.getWidth.W))
    val value1  = Output(memoryType)
    val value2  = Output(memoryType)
    val value3  = Output(memoryType)
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
    val value   = Output(memoryType)
    val value1  = Output(memoryType)
    val value2  = Output(memoryType)
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
    val value   = Output(memoryType)
  })

  val memory = Mem(memoryDepth, memoryType)

  loadMemoryFromFile(memory, "./mem2")

  io.value := memory(io.address)
}

class FileHasSuffix(memoryDepth: Int, memoryType: Data) extends Module {
  val io = IO(new Bundle {
    val address = Input(UInt(memoryType.getWidth.W))
    val value   = Output(memoryType)
    val value2  = Output(memoryType)
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
    val value   = Output(new MemoryShape)
  })

  val memory = Mem(memoryDepth, new MemoryShape)

  loadMemoryFromFile(memory, "./mem", MemoryLoadFileType.Hex)

  io.value := memory(io.address)
}


/**
  * The following tests are a bit incomplete and check that the output verilog is properly constructed
  * For more complete working examples
  * @see <a href="https://github.com/freechipsproject/chisel-testers">Chisel Testers</a> LoadMemoryFromFileSpec.scala
  */
class LoadMemoryFromFileSpec extends AnyFreeSpec with Matchers {
  def fileExistsWithMem(file: File, mem: Option[String] = None): Unit = {
    info(s"$file exists")
    file.exists() should be (true)
    mem.foreach( m => {
      info(s"Memory $m is referenced in $file")
      val found = io.Source.fromFile(file).getLines.exists { _.contains(s"""readmemh("$m"""") }
      found should be (true)
    } )
    file.delete()
  }

  "Users can specify a source file to load memory from" in {
    val testDirName = "test_run_dir/load_memory_spec"

    val result = (new ChiselStage).execute(
      args = Array("-X", "verilog", "--target-dir", testDirName),
      annotations = Seq(ChiselGeneratorAnnotation(() => new UsesMem(memoryDepth = 8, memoryType = UInt(16.W))))
    )

    val dir = new File(testDirName)
    fileExistsWithMem(new File(dir, "UsesMem.UsesMem.memory.v"), Some("./mem1"))
    fileExistsWithMem(new File(dir, "UsesMem.UsesMemLow.memory.v"), Some("./mem2"))
    fileExistsWithMem(new File(dir, "firrtl_black_box_resource_files.f"))

  }

  "Calling a module that loads memories from a file more than once should work" in {
    val testDirName = "test_run_dir/load_three_memory_spec"

    val result = (new ChiselStage).execute(
      args = Array("-X", "verilog", "--target-dir", testDirName),
      annotations = Seq(ChiselGeneratorAnnotation(() => new UsesThreeMems(memoryDepth = 8, memoryType = UInt(16.W))))
    )

    val dir = new File(testDirName)
    fileExistsWithMem( new File(dir, "UsesThreeMems.UsesThreeMems.memory1.v"), Some("./mem1"))
    fileExistsWithMem( new File(dir, "UsesThreeMems.UsesThreeMems.memory2.v"), Some("./mem1"))
    fileExistsWithMem( new File(dir, "UsesThreeMems.UsesThreeMems.memory3.v"), Some("./mem1"))
    fileExistsWithMem( new File(dir, "firrtl_black_box_resource_files.f"))

  }

  "In this example the memory has a complex memory type containing a bundle" in {
    val complexTestDirName = "test_run_dir/complex_memory_load"

    val result = (new ChiselStage).execute(
      args = Array("-X", "verilog", "--target-dir", complexTestDirName),
      annotations = Seq(ChiselGeneratorAnnotation(() => new HasComplexMemory(memoryDepth = 8)))
    )

    val dir = new File(complexTestDirName)
    val memoryElements = Seq("a", "b", "c")

    memoryElements.foreach { element =>
      val file = new File(dir, s"HasComplexMemory.HasComplexMemory.memory_$element.v")
      file.exists() should be (true)
      val fileText = io.Source.fromFile(file).getLines().mkString("\n")
      fileText should include (s"""$$readmemh("./mem_$element", HasComplexMemory.memory_$element);""")
      file.delete()
    }

  }

}
