//// SPDX-License-Identifier: Apache-2.0
//
//package chiselTests.aop
//
//import chisel3.testers.{BasicTester, TesterDriver}
//import chiselTests.{ChiselFlatSpec, Utils}
//import chisel3._
//import chisel3.aop.Select
//import chisel3.aop.injecting.{InjectingAspect2, InBody}
//import chisel3.experimental.hierarchy._
//import logger.{LogLevel, LogLevelAnnotation}
//import chisel3.internal.instantiable
//
//object InjectionHierarchy2 {
//
//  @instantiable
//  class SubmoduleManipulationTester extends BasicTester {
//    @public val moduleSubmoduleA = Module(new SubmoduleA)
//  }
//
//  @instantiable
//  class MultiModuleInjectionTester extends BasicTester {
//    @public val subA0 = Module(new SubmoduleA)
//    @public val subA1 = Module(new SubmoduleA)
//  }
//
//  @instantiable
//  class SubmoduleA extends Module {
//    @public val io = IO(new Bundle {
//      val out = Output(Bool())
//    })
//    io.out := false.B
//  }
//
//  @instantiable
//  class SubmoduleB extends Module {
//    @public val io = IO(new Bundle {
//      val in = Input(Bool())
//    })
//  }
//
//  @instantiable
//  class SubmoduleC extends experimental.ExtModule with util.HasExtModuleInline {
//    @public val io = IO(new Bundle {
//      val in = Input(Bool())
//    })
//    //scalastyle:off regex
//    setInline("SubmoduleC.v", s"""
//                                   |module SubmoduleC(
//                                   |    input  io_in
//                                   |);
//                                   |endmodule
//    """.stripMargin)
//  }
//
//  @instantiable
//  class AspectTester(results: Seq[Int]) extends BasicTester {
//    @public val values = VecInit(results.map(_.U))
//    @public val counter = RegInit(0.U(results.length.W))
//    counter := counter + 1.U
//    when(counter >= values.length.U) {
//      stop()
//    }.otherwise {
//      when(reset.asBool() === false.B) {
//        assert(counter === values(counter))
//      }
//    }
//  }
//}
//
//class InjectionSpec2 extends ChiselFlatSpec with Utils {
//  import InjectionHierarchy2._
//  val correctValueAspect = InjectingAspect2 { (dut: AspectTester, inBody: InBody) =>
//    inBody(dut.toDefinition) { instance =>
//      for(i <- 0 until instance.values.length) {
//        instance.values(i) := i.U
//      }
//    }
//  }
//
//  val wrongValueAspect = InjectingAspect2 { (dut: AspectTester, inBody: InBody) =>
//    inBody(dut.toDefinition) { instance =>
//      for(i <- 0 until instance.values.length) {
//        instance.values(i) := (i + 1).U
//      }
//    }
//  }
//
//  val manipulateSubmoduleAspect = InjectingAspect2 { (dut: SubmoduleManipulationTester, inBody: InBody) =>
//    inBody(dut.toDefinition) { instance =>
//      val moduleSubmoduleB = Module(new SubmoduleB)
//      moduleSubmoduleB.io.in := instance.moduleSubmoduleA.io.out
//      //if we're here then we've elaborated correctly
//      stop()
//    }
//  }
//
//  val duplicateSubmoduleAspect = InjectingAspect2 { (dut: SubmoduleManipulationTester, inBody: InBody) =>
//    inBody(dut.toDefinition) { _ =>
//      // By creating a second SubmoduleA, the module names would conflict unless they were uniquified
//      val moduleSubmoduleA2 = Module(new SubmoduleA)
//      //if we're here then we've elaborated correctly
//      stop()
//    }
//  }
//
//  val addingExternalModules = InjectingAspect2 { (dut: SubmoduleManipulationTester, inBody: InBody) =>
//    inBody(dut.toDefinition) { _ =>
//      // By creating a second SubmoduleA, the module names would conflict unless they were uniquified
//      val moduleSubmoduleC = Module(new SubmoduleC)
//      //if we're here then we've elaborated correctly
//      stop()
//    }
//  }
//
//  /*
//  val multiModuleInjectionAspect = InjectingAspect2(
//    { top: MultiModuleInjectionTester =>
//      Select.collectDeep(top) { case m: SubmoduleA => m }
//    },
//    { m: Module =>
//      val wire = Wire(Bool())
//      wire := m.reset.asBool()
//      dontTouch(wire)
//      stop()
//    }
//  )
//  */
//
//  "Test" should "pass if inserted the correct values" in {
//    assertTesterPasses{ new AspectTester(Seq(0, 1, 2)) }
//  }
//  "Test" should "fail if inserted the wrong values" in {
//    assertTesterFails{ new AspectTester(Seq(9, 9, 9)) }
//  }
//  "Test" should "pass if pass wrong values, but correct with aspect" in {
//    assertTesterPasses({ new AspectTester(Seq(9, 9, 9))} , Nil, Seq(correctValueAspect) ++ TesterDriver.verilatorOnly)
//  }
//  "Test" should "pass if pass wrong values, then wrong aspect, then correct aspect" in {
//    assertTesterPasses(
//      new AspectTester(Seq(9, 9, 9)), Nil, Seq(wrongValueAspect, correctValueAspect) ++ TesterDriver.verilatorOnly
//    )
//  }
//  "Test" should "fail if pass wrong values, then correct aspect, then wrong aspect" in {
//    assertTesterFails({ new AspectTester(Seq(9, 9, 9))} , Nil, Seq(correctValueAspect, wrongValueAspect))
//  }
//
//  "Test" should "pass if the submodules in SubmoduleManipulationTester can be manipulated by manipulateSubmoduleAspect" in {
//    assertTesterPasses({ new SubmoduleManipulationTester} , Nil, Seq(manipulateSubmoduleAspect) ++ TesterDriver.verilatorOnly)
//  }
//
//  "Module name collisions when adding a new module" should "be resolved" in {
//    assertTesterPasses(
//      { new SubmoduleManipulationTester},
//      Nil,
//      Seq(duplicateSubmoduleAspect) ++ TesterDriver.verilatorOnly
//    )
//  }
//
//  "Adding external modules" should "work" in {
//    assertTesterPasses(
//      { new SubmoduleManipulationTester},
//      Nil,
//      Seq(addingExternalModules) ++ TesterDriver.verilatorOnly
//    )
//  }
//
//  /*
//  "Injection into multiple submodules of the same class" should "work" in {
//    assertTesterPasses(
//      {new MultiModuleInjectionTester},
//      Nil,
//      Seq(multiModuleInjectionAspect) ++ TesterDriver.verilatorOnly
//    )
//  }
//  */
//}
//