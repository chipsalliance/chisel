// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.FirrtlFormat.FormatWidth
import circt.stage.ChiselStage
import chisel3.testing.scalatest.FileCheck
import org.scalactic.source.Position
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.annotation.nowarn

class SimLogSpec extends AnyFlatSpec with Matchers with FileCheck {

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
}
