// See LICENSE for license details.

package chiselTests

import chisel3.tutorial.lesson1.AnalyzeModule
import chisel3._
import logger.{LogLevel, Logger}
import firrtl.FirrtlExecutionSuccess
import org.scalatest.{FreeSpec, Matchers}

class MyModule(analyze: Boolean) extends Module
    with AnalyzeModule {
  if(analyze) setAnalyze()
  val io = IO(new Bundle {
    val tval = Input(UInt(16.W))
    val fval = Input(UInt(16.W))
    val cond = Input(Bool())
    val out = Output(UInt(16.W))
  })
  io.out := io.fval
  when(io.cond) {
    io.out := io.tval
  }
}

class AnalyzeModuleTester extends FreeSpec with Matchers {
  "Can run tutorial to print mux count" - {
    "Can turn on Analyze on MyModule" in {
      Logger.makeScope() {
        val captor = new Logger.OutputCaptor
        Logger.setOutput(captor.printStream)
        // logger.Logger.setLevel(LogLevel.Info)
        Logger.setLevel(classOf[chisel3.tutorial.lesson1.AnalyzeCircuit], LogLevel.Info)

        Driver.execute(
          Array("-X", "low", "--target-dir", "test_run_dir"),
          () => new MyModule(true)
        ) match {
          case ChiselExecutionSuccess(_, _, Some(firrtlResult: FirrtlExecutionSuccess)) =>
            val messagesLogged = captor.getOutputAsString
            println(s"messages logged----\n$messagesLogged\n-----")
            messagesLogged.contains("muxes!") should be(true)
        }
      }
    }
  }
}
