// See LICENSE for license details.

package chiselTests

import java.io.File

import chisel3._
import chisel3.util.{loadMemoryFromFile, log2Ceil}
import firrtl.FirrtlExecutionSuccess
import firrtl.annotations.MemoryLoadFileType
import org.scalatest.{FreeSpec, Matchers}

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


class ComplexMemoryLoadingSpec extends FreeSpec with Matchers {
  val testDirName = "test_run_dir/complex_memory_load"

  "Users can specify a source file to load memory from" in {
    val result = Driver.execute(
      args = Array("-X", "verilog", "--target-dir", testDirName),
      dut = () => new HasComplexMemory(memoryDepth = 8)
    )

    result match {
      case ChiselExecutionSuccess(_, emitted, Some(FirrtlExecutionSuccess(emitType, firrtlEmitted))) =>
        val dir = new File(testDirName)
        val memoryElements = Seq("a", "b", "c")

        memoryElements.foreach { element =>
          val file = new File(dir, s"HasComplexMemory.HasComplexMemory.memory_$element.v")
          file.exists() should be (true)
          val fileText = io.Source.fromFile(file).getLines().mkString("\n")
          fileText should include (s"""$$readmemh("./mem_$element", HasComplexMemory.memory_$element);""")
        }

      case _=>
        println(s"Failed compile")
    }
  }
}
