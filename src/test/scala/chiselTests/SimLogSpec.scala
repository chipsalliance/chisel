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
      fd.flush()
    }
    ChiselStage
      .emitCHIRRTL(new MyModule)
      .fileCheck()(
        """|CHECK: fprintf(clock, UInt<1>(0h1), "logfile.log", "An exact string")
           |CHECK: fflush(clock, UInt<1>(0h1), "logfile.log")""".stripMargin
      )
  }

  it should "support printf with format strings" in {
    class MyModule extends Module {
      val in = IO(Input(UInt(8.W)))
      val fd = SimLog.file("logfile.log")
      // fd.printf("in = %d\n", in)
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
      fd.flush()
    }
    ChiselStage
      .emitCHIRRTL(new MyModule(SimLog.file("logfile.log")))
      .fileCheck()(
        """|CHECK: fprintf(clock, UInt<1>(0h1), "logfile.log", "in = %0d\n", in)
           |CHECK: fflush(clock, UInt<1>(0h1), "logfile.log")""".stripMargin
      )
    ChiselStage
      .emitCHIRRTL(new MyModule(SimLog.StdErr))
      .fileCheck()(
        """|CHECK: printf(clock, UInt<1>(0h1), "in = %0d\n", in)
           |CHECK: fflush(clock, UInt<1>(0h1))""".stripMargin
      )
  }

  it should "support Printable filenames" in {
    class MyModule extends Module {
      val idx = IO(Input(UInt(8.W)))
      val fd = SimLog.file(cf"logfile_$idx%0d.log")
      fd.printf(cf"An exact string")
      fd.flush()
    }
    ChiselStage
      .emitCHIRRTL(new MyModule)
      .fileCheck()(
        """|CHECK: fprintf(clock, UInt<1>(0h1), "logfile_%0d.log", idx, "An exact string")
           |CHECK: fflush(clock, UInt<1>(0h1), "logfile_%0d.log", idx)""".stripMargin
      )
  }

  it should "check scope for Printable filenames on printf" in {
    class Child(log: SimLog) extends Module {
      val bar = IO(Input(UInt(8.W)))
      log.printf(cf"bar = $bar%0d\n")
    }
    class MyModule extends Module {
      val foo = Wire(UInt(8.W))
      val log = SimLog.file(cf"logfile_$foo%0d.log")
      val child = Module(new Child(log))
      child.bar := foo
    }
    val e = the[ChiselException] thrownBy ChiselStage.emitCHIRRTL(new MyModule, Array("--throw-on-first-error"))
    (e.getMessage should include).regex("SimLog filename operand '.*' is not visible from the current module Child")
  }

  it should "check scope for Printable filenames on flush" in {
    class MyModule extends Module {
      var log: SimLog = null
      when(true.B) {
        val foo = Wire(UInt(8.W))
        log = SimLog.file(cf"logfile_$foo%0d.log")
      }
      log.flush()
    }
    val e = the[ChiselException] thrownBy ChiselStage.emitCHIRRTL(new MyModule, Array("--throw-on-first-error"))
    (e.getMessage should include).regex("SimLog filename operand '.*' has escaped the scope of the block")
  }

  it should "support writing to a file in simulation" in {
    val testdir = implicitly[HasTestingDirectory].getDirectory
    val logfile = testdir.resolve("workdir-verilator").resolve("logfile.log").toFile
    logfile.delete() // delete the log file if it exists
    class MyModule extends Module {
      val (count, done) = chisel3.util.Counter(0 until 4)
      val fd = SimLog.file("logfile.log")
      fd.printf(cf"count = $count%0d\n")
      when(done) { stop() }
    }
    simulate(new MyModule)(RunUntilFinished(5))
    val expected = (0 until 4).map(i => s"count = $i").toList
    val lines = io.Source.fromFile(logfile).getLines().toList
    lines should be(expected)
  }
}
