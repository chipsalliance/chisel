// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.util.{ChiselLoadMemoryAnnotation, log2Ceil}
import firrtl.FirrtlExecutionSuccess
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

  chisel3.experimental.annotate(ChiselLoadMemoryAnnotation(memory, "test_run_dir/examples.LoadMemoryFromFileSpec1251342320/mem1.txt"))
  io.value := memory(io.address)
}


class ComplexMemoryLoadingSpec extends FreeSpec with Matchers {
  "Users can specify a source file to load memory from" in {
    val result = Driver.execute(
      args = Array("-X", "verilog", "--target-dir", "test_run_dir/ComplexMemorySpec"),
      dut = () => new HasComplexMemory(memoryDepth = 8)
    )

    result match {
      case ChiselExecutionSuccess(_, emitted, Some(FirrtlExecutionSuccess(emitType, firrtlEmitted))) =>
        //        println(s"emitted code\n$emitted\nType: $emitType\nFirrtl emitted\n$firrtlEmitted")
        println(s"Type: $emitType\nFirrtl emitted\n$firrtlEmitted")
      case _=>
        println(s"Failed compile")
    }
  }
}
