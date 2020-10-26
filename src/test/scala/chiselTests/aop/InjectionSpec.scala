// SPDX-License-Identifier: Apache-2.0

package chiselTests.aop

import chisel3.testers.{BasicTester, TesterDriver}
import chiselTests.{ChiselFlatSpec, Utils}
import chisel3._
import chisel3.aop.injecting.InjectingAspect
import logger.{LogLevel, LogLevelAnnotation}

object InjectionHierarchy {

  class SubmoduleManipulationTester extends BasicTester {
    val moduleSubmoduleA = Module(new SubmoduleA)
  }

  class SubmoduleA extends Module {
    val io = IO(new Bundle {
      val out = Output(Bool())
    })
    io.out := false.B
  }

  class SubmoduleB extends Module {
    val io = IO(new Bundle {
      val in = Input(Bool())
    })
  }

  class AspectTester(results: Seq[Int]) extends BasicTester {
    val values = VecInit(results.map(_.U))
    val counter = RegInit(0.U(results.length.W))
    counter := counter + 1.U
    when(counter >= values.length.U) {
      stop()
    }.otherwise {
      when(reset.asBool() === false.B) {
        assert(counter === values(counter))
      }
    }
  }
}

class InjectionSpec extends ChiselFlatSpec with Utils {
  import InjectionHierarchy._
  val correctValueAspect = InjectingAspect(
    {dut: AspectTester => Seq(dut)},
    {dut: AspectTester =>
      for(i <- 0 until dut.values.length) {
        dut.values(i) := i.U
      }
    }
  )

  val wrongValueAspect = InjectingAspect(
    {dut: AspectTester => Seq(dut)},
    {dut: AspectTester =>
      for(i <- 0 until dut.values.length) {
        dut.values(i) := (i + 1).U
      }
    }
  )

  val manipulateSubmoduleAspect = InjectingAspect(
    {dut: SubmoduleManipulationTester => Seq(dut)},
    {dut: SubmoduleManipulationTester =>
      val moduleSubmoduleB = Module(new SubmoduleB)
      moduleSubmoduleB.io.in := dut.moduleSubmoduleA.io.out
      //if we're here then we've elaborated correctly
      stop()
    }
  )

  val duplicateSubmoduleAspect = InjectingAspect(
    {dut: SubmoduleManipulationTester => Seq(dut)},
    {_: SubmoduleManipulationTester =>
      // By creating a second SubmoduleA, the module names would conflict unless they were uniquified
      val moduleSubmoduleA2 = Module(new SubmoduleA)
      //if we're here then we've elaborated correctly
      stop()
    }
  )

  "Test" should "pass if inserted the correct values" in {
    assertTesterPasses{ new AspectTester(Seq(0, 1, 2)) }
  }
  "Test" should "fail if inserted the wrong values" in {
    assertTesterFails{ new AspectTester(Seq(9, 9, 9)) }
  }
  "Test" should "pass if pass wrong values, but correct with aspect" in {
    assertTesterPasses({ new AspectTester(Seq(9, 9, 9))} , Nil, Seq(correctValueAspect) ++ TesterDriver.verilatorOnly)
  }
  "Test" should "pass if pass wrong values, then wrong aspect, then correct aspect" in {
    assertTesterPasses(
      new AspectTester(Seq(9, 9, 9)), Nil, Seq(wrongValueAspect, correctValueAspect) ++ TesterDriver.verilatorOnly
    )
  }
  "Test" should "fail if pass wrong values, then correct aspect, then wrong aspect" in {
    assertTesterFails({ new AspectTester(Seq(9, 9, 9))} , Nil, Seq(correctValueAspect, wrongValueAspect))
  }

  "Test" should "pass if the submodules in SubmoduleManipulationTester can be manipulated by manipulateSubmoduleAspect" in {
    assertTesterPasses({ new SubmoduleManipulationTester} , Nil, Seq(manipulateSubmoduleAspect) ++ TesterDriver.verilatorOnly)
  }

  "Module name collisions when adding a new module" should "be resolved" in {
    assertTesterPasses(
      { new SubmoduleManipulationTester},
      Nil,
      Seq(duplicateSubmoduleAspect) ++ TesterDriver.verilatorOnly
    )
  }
}
