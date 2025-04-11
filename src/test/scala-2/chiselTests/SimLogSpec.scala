// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.testing.HasTestingDirectory
import chisel3.testing.scalatest.FileCheck
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SimLogSpec extends AnyFlatSpec with Matchers with FileCheck with ChiselSim {

  behavior.of("SimLog")

  it should "support simple printing to files" in {
    class MyModule extends Module {
      val fd = SimLog.file("logfile.log")
      fd.printf(cf"An exact string")
    }
    ChiselStage
      .emitCHIRRTL(new MyModule)
      .fileCheck()(
        """CHECK: fprintf(clock, UInt<1>(0h1), "logfile.log", "An exact string")"""
      )
  }

  it should "support printf with format strings" in {
    class MyModule extends Module {
      val in = IO(Input(UInt(8.W)))
      val fd = SimLog.file("logfile.log")
      fd.printf("in = %d\n", in)
    }
    ChiselStage
      .emitCHIRRTL(new MyModule)
      .fileCheck()(
        """CHECK: fprintf(clock, UInt<1>(0h1), "logfile.log", "in = %d\n", in)"""
      )
  }

  it should "support writing code generic to the FileDescriptor" in {
    class MyModule(fd: SimLog) extends Module {
      val in = IO(Input(UInt(8.W)))
      fd.printf(cf"in = $in%0d\n")
    }
    ChiselStage
      .emitCHIRRTL(new MyModule(SimLog.file("logfile.log")))
      .fileCheck()(
        """CHECK: fprintf(clock, UInt<1>(0h1), "logfile.log", "in = %0d\n", in)"""
      )
    ChiselStage
      .emitCHIRRTL(new MyModule(SimLog.StdErr))
      .fileCheck()(
        """CHECK: printf(clock, UInt<1>(0h1), "in = %0d\n", in)"""
      )
  }

  it should "support writing to a file in simulation" in {
    class MyModule extends Module {
      val (count, done) = chisel3.util.Counter(0 until 4)
      val fd = SimLog.file("logfile.log")
      fd.printf(cf"count = $count%0d\n")
      when(done) { stop() }
    }
    simulate(new MyModule)(RunUntilFinished(5))
    val testdir = implicitly[HasTestingDirectory].getDirectory
    val logfile = testdir.resolve("workdir-verilator").resolve("logfile.log")
    val expected = (0 until 4).map(i => s"count = $i").toList
    val lines = io.Source.fromFile(logfile.toFile).getLines().toList
    lines should be(expected)
  }
}
