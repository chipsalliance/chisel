// See LICENSE for license details.

package chiselTests

import java.io.{ByteArrayOutputStream, File, PrintStream}

import chisel3._
import chisel3.util.loadMemoryFromFile
import firrtl.FirrtlExecutionSuccess
import org.scalatest.{FreeSpec, Matchers}

//noinspection TypeAnnotation
class UsesMem(memoryDepth: Int, memoryType: Data) extends Module {
  val io = IO(new Bundle {
    val address = Input(UInt(memoryType.getWidth.W))
    val value   = Output(memoryType)
    val value2  = Output(memoryType)
  })

  val memory = Mem(memoryDepth, memoryType)
  loadMemoryFromFile(memory, "./mem1")

  io.value := memory(io.address)

  val low = Module(new UsesMemLow(memoryDepth, memoryType))

  low.io.address := io.address
  io.value2 := low.io.value
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

//noinspection TypeAnnotation
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

class LoadMemoryFromFileSpec extends FreeSpec with Matchers {
  val testDirName = "test_run_dir/load_memory_spec"
  "Users can specify a source file to load memory from" in {
    val result = Driver.execute(
      args = Array("-X", "verilog", "--target-dir", testDirName),
      dut = () => new UsesMem(memoryDepth = 8, memoryType = UInt(16.W))
    )

    result match {
      case ChiselExecutionSuccess(_, _, Some(FirrtlExecutionSuccess(_, _))) =>
        val dir = new File(testDirName)
        new File(dir, "UsesMem.UsesMem.memory.v").exists() should be (true)
        new File(dir, "UsesMem.UsesMemLow.memory.v").exists() should be (true)
        new File(dir, "firrtl_black_box_resource_files.f").exists() should be (true)
      case _=>
        throw new Exception("Failed compile")
    }
  }
}
