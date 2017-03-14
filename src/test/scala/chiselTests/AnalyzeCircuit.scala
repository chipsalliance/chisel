// See LICENSE for license details.

package chisel3tutorialtests

import chisel3.tutorial.lesson1.{AnalyzeModule, AnalyzeCircuit}

import chisel3._
import chisel3.experimental.ChiselAnnotation

import logger.{LazyLogging, LogLevel}

import firrtl.FirrtlExecutionSuccess

import org.scalatest.{FreeSpec, Matchers}
import java.io.{ByteArrayOutputStream, PrintStream}

class OutputCaptor {
  val byteArrayOutputStream = new ByteArrayOutputStream()
  val printStream = new PrintStream(byteArrayOutputStream)
  def getOutputStrings: Seq[String] = {
    byteArrayOutputStream.toString.split("""\n""")
  }
}

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
      val captor = new OutputCaptor
      logger.Logger.setLevel(logger.LogLevel.Info)
      logger.Logger.log2StringBuffer()
      //logger.Logger.setClassLogLevels(Map("chisel3.tutorial.lesson1.AnalyzeCircuit" -> logger.LogLevel.Info))
      Driver.execute(Array("-X", "low", "--target-dir", "test_run_dir"), () => new MyModule(true)) match {
        case ChiselExecutionSuccess(_, _, Some(firrtlResult: FirrtlExecutionSuccess)) =>
          val messagesLogged = logger.Logger.getStringBuffer.get
          println(messagesLogged)
          messagesLogged.contains("muxes!") should be(true)
      }
    }
  }
}
