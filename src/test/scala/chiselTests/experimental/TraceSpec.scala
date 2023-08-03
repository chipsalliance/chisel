// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.Trace._
import chisel3.stage.{ChiselGeneratorAnnotation, DesignAnnotation}
import chisel3.util.experimental.InlineInstance
import circt.stage.ChiselStage
import firrtl.AnnotationSeq
import firrtl.annotations.TargetToken.{Instance, OfModule, Ref}
import firrtl.annotations.{CompleteTarget, InstanceTarget, ReferenceTarget}
import firrtl.util.BackendCompilationUtilities.createTestDirectory

import org.scalatest.matchers.should.Matchers

class TraceSpec extends ChiselFlatSpec with Matchers {

  def refTarget(topName: String, ref: String, path: Seq[(Instance, OfModule)] = Seq()) =
    ReferenceTarget(topName, topName, path, ref, Seq())

  def instTarget(topName: String, instance: String, ofModule: String, path: Seq[(Instance, OfModule)] = Seq()) =
    InstanceTarget(topName, topName, path, instance, ofModule)

  def compile(testName: String, gen: () => Module): (os.Path, AnnotationSeq) = {
    val testDir = os.Path(createTestDirectory(testName).getAbsolutePath)
    val annos = (new ChiselStage).execute(
      Array("--target-dir", s"$testDir", "--target", "systemverilog", "--split-verilog"),
      Seq(
        ChiselGeneratorAnnotation(gen)
      )
    )
    (testDir, annos)
  }

  "TraceFromAnnotations" should "be able to get nested name." in {
    class Bundle0 extends Bundle {
      val a = UInt(8.W)
      val b = Bool()
      val c = Enum0.Type
    }

    class Bundle1 extends Bundle {
      val a = new Bundle0
      val b = Vec(4, Vec(4, Bool()))
    }

    class Module0 extends Module {
      val i = IO(Input(new Bundle1))
      val o = IO(Output(new Bundle1))
      val r = Reg(new Bundle1)
      o := r
      r := i

      traceName(r)
      dontTouch(r)
      traceName(i)
      dontTouch(i)
      traceName(o)
      dontTouch(o)
    }

    class Module1 extends Module {
      val i = IO(Input(new Bundle1))
      val m0 = Module(new Module0)
      m0.i := i
      m0.o := DontCare
    }

    object Enum0 extends ChiselEnum {
      val s0, s1, s2 = Value
    }

    val (testDir, annos) = compile("TraceFromAnnotations", () => new Module1)
    val dut = annos.collectFirst { case DesignAnnotation(dut) => dut }.get.asInstanceOf[Module1]
    // out of Builder.

    val oneTarget = finalTarget(annos)(dut.m0.r.a.a).head
    val ioTarget = finalTarget(annos)(dut.m0.i.b(1)(2)).head

    val topName = "Module1"
    oneTarget should be(refTarget(topName, "r_a_a", Seq(Instance("m0") -> OfModule("Module0"))))

    ioTarget should be(refTarget(topName, "i_b_1_2", Seq(Instance("m0") -> OfModule("Module0"))))

    // Below codes doesn't needs to be a FIRRTL Transform.
    def generateVerilatorConfigFile(data: Seq[Data], annos: AnnotationSeq): String =
      """`verilator_config
        |lint_off -rule unused
        |lint_off -rule declfilename
        |""".stripMargin +
        data
          .flatMap(finalTarget(annos))
          .toSet
          .map { target: CompleteTarget =>
            s"""public_flat_rd -module "${target.tokens.collectFirst {
              case OfModule(m) => m
            }.get}" -var "${target.tokens.collectFirst { case Ref(r) => r }.get}""""
          }
          .mkString("\n") + "\n"

    def verilatorTemplate(data: Seq[Data], annos: AnnotationSeq): String = {
      val vpiNames = data.flatMap(finalTarget(annos)).map { ct =>
        s"""TOP.${ct.circuit}.${ct.path.map { case (Instance(i), _) => i }.mkString(".")}.${ct.tokens.collectFirst {
          case Ref(r) => r
        }.get}"""
      }
      s"""
         |#include "V${topName}.h"
         |#include "verilated_vpi.h"
         |#include <memory>
         |#include <verilated.h>
         |
         |int vpiGetInt(const char name[]) {
         |  vpiHandle vh1 = vpi_handle_by_name((PLI_BYTE8 *)name, NULL);
         |  if (!vh1)
         |    vl_fatal(__FILE__, __LINE__, "sim_main", "No handle found");
         |  s_vpi_value v;
         |  v.format = vpiIntVal;
         |  vpi_get_value(vh1, &v);
         |  return v.value.integer;
         |}
         |
         |int main(int argc, char **argv) {
         |  const std::unique_ptr<VerilatedContext> contextp{new VerilatedContext};
         |  contextp->commandArgs(argc, argv);
         |  const std::unique_ptr<V$topName> top{new V$topName{contextp.get(), "TOP"}};
         |  top->reset = 0;
         |  top->clock = 0;
         |  int a_b = 1;
         |  top->i_a_b = a_b;
         |  bool started = false;
         |  int ticks = 20;
         |  while (ticks--) {
         |    contextp->timeInc(1);
         |    top->clock = !top->clock;
         |    if (!top->clock) {
         |      if (contextp->time() > 1 && contextp->time() < 10) {
         |        top->reset = 1;
         |      } else {
         |        top->reset = 0;
         |        started = true;
         |      }
         |      a_b = a_b ? 0 : 1;
         |      top->i_a_b = a_b;
         |    }
         |    top->eval();
         |    VerilatedVpi::callValueCbs();
         |    if (started && !top->clock) {
         |      const int i = top->i_a_b;
         |      const int o = vpiGetInt("${vpiNames.head}");
         |      if (i == o)
         |        vl_fatal(__FILE__, __LINE__, "sim_main", "${vpiNames.head} should be the old value of Module1.i_a_b");
         |      printf("${vpiNames.head}=%d Module1.m0.o_a_b=%d\\n", i, o);
         |    }
         |  }
         |  top->final();
         |  return 0;
         |}
         |""".stripMargin
    }

    val config = os.temp(dir = testDir, contents = generateVerilatorConfigFile(Seq(dut.m0.o.a.b), annos))
    val verilog = testDir / s"$topName.sv"
    val cpp = os.temp(dir = testDir, suffix = ".cpp", contents = verilatorTemplate(Seq(dut.m0.o.a.b), annos))
    val exe = testDir / "obj_dir" / s"V$topName"
    os.proc(
      "verilator",
      "--cc",
      "--exe",
      "--build",
      "--vpi",
      s"-I$testDir",
      s"$cpp",
      s"$verilog",
      s"$config"
    ).call(stdout = os.Inherit, stderr = os.Inherit, cwd = testDir)
    assert(
      os.proc(s"$exe").call(stdout = os.Inherit, stderr = os.Inherit).exitCode == 0,
      "verilator should exit peacefully"
    )
  }

  "TraceFromCollideBundle" should "work" in {
    class CollideModule extends Module {
      val a = IO(
        Input(
          Vec(
            2,
            new Bundle {
              val b = Flipped(Bool())
              val c = Vec(
                2,
                new Bundle {
                  val d = UInt(2.W)
                  val e = Flipped(UInt(3.W))
                }
              )
              val c_1_e = UInt(4.W)
            }
          )
        )
      )
      val a_0_c = IO(Output(UInt(5.W)))
      val a__0 = IO(Output(UInt(5.W)))
      a_0_c := DontCare
      a__0 := DontCare

      traceName(a)
      dontTouch(a)
      traceName(a_0_c)
      dontTouch(a_0_c)
      traceName(a__0)
      dontTouch(a__0)
    }

    val (_, annos) = compile("TraceFromCollideBundle", () => new CollideModule)
    val dut = annos.collectFirst { case DesignAnnotation(dut) => dut }.get.asInstanceOf[CollideModule]

    val topName = "CollideModule"

    val a0 = finalTarget(annos)(dut.a(0))
    val a__0 = finalTarget(annos)(dut.a__0).head
    val a__0_ref = refTarget(topName, "a__0")
    a0.foreach(_ shouldNot be(a__0_ref))
    a__0 should be(a__0_ref)

    val a0_c = finalTarget(annos)(dut.a(0).c)
    val a_0_c = finalTarget(annos)(dut.a_0_c).head
    val a_0_c_ref = refTarget(topName, "a_0_c")
    a0_c.foreach(_ shouldNot be(a_0_c_ref))
    a_0_c should be(a_0_c_ref)

    val a0_c1_e = finalTarget(annos)(dut.a(0).c(1).e).head
    val a0_c_1_e = finalTarget(annos)(dut.a(0).c_1_e).head
    println(dut.a(0).c(1).e.toTarget)
    println(a0_c1_e)
    println(dut.a(0).c_1_e.toTarget)
    println(a0_c_1_e)
    a0_c1_e should be(refTarget(topName, "a_0_c_1_e"))
    a0_c_1_e should be(refTarget(topName, "a_0_c_1_e_0"))
  }

  "Inline should work" should "work" in {
    class Module0 extends Module {
      val i = IO(Input(Bool()))
      val o = IO(Output(Bool()))
      traceName(i)
      dontTouch(i)
      o := !i
    }

    class Module1 extends Module {
      val i = IO(Input(Bool()))
      val o = IO(Output(Bool()))
      val m0 = Module(new Module0 with InlineInstance)
      m0.i := i
      o := m0.o
    }

    val (_, annos) = compile("Inline", () => new Module1)
    val dut = annos.collectFirst { case DesignAnnotation(dut) => dut }.get.asInstanceOf[Module1]

    val m0_i = finalTarget(annos)(dut.m0.i).head
    m0_i should be(refTarget("Module1", "m0_i"))
  }

  "Constant Propagation" should "be turned off by traceName" in {
    class Module0 extends Module {
      val i = WireDefault(1.U)
      val i0 = i + 1.U
      val o = IO(Output(UInt(2.W)))
      traceName(i0)
      dontTouch(i0)
      o := i0
    }

    val (_, annos) = compile("ConstantProp", () => new Module0)
    val dut = annos.collectFirst { case DesignAnnotation(dut) => dut }.get.asInstanceOf[Module0]

    val i0 = finalTarget(annos)(dut.i0).head
    i0 should be(refTarget("Module0", "i0"))
  }

  "Nested Module" should "work" in {
    class Io extends Bundle {
      val i = Input(Bool())
      val o = Output(Bool())
    }

    class Not extends Module {
      val io = IO(new Io)
      io.o := !io.i
    }

    class M1 extends Module {
      val io = IO(new Io)
      val bar = Module(new Not)
      bar.io <> io
    }

    class M2 extends Module {
      val io = IO(new Io)
      val m1 = Module(new M1 with InlineInstance)
      val foo = Module(new Not)

      m1.io.i := io.i
      foo.io.i := io.i

      io.o := m1.io.o && foo.io.o
    }

    class M3 extends Module {
      val io = IO(new Io)
      val m2 = Module(new M2)
      io <> m2.io
      traceName(m2.foo)
      traceName(m2.m1.bar)
    }

    val (_, annos) = compile("NestedModule", () => new M3)
    val m3 = annos.collectFirst { case DesignAnnotation(dut) => dut }.get.asInstanceOf[M3]

    val m2_m1_not = finalTarget(annos)(m3.m2.m1.bar).head
    val m2_not = finalTarget(annos)(m3.m2.foo).head

    m2_m1_not should be(instTarget("M3", "m1_bar", "Not", Seq(Instance("m2") -> OfModule("M2"))))
    m2_not should be(instTarget("M3", "foo", "Not", Seq(Instance("m2") -> OfModule("M2"))))
  }

  "All traced signal" should "generate" in {
    class M extends Module {
      val a = Wire(Bool())
      val b = Wire(Vec(2, Bool()))
      a := DontCare
      b := DontCare
      Seq(a, b).foreach { a => traceName(a); dontTouch(a) }
    }
    val (_, annos) = compile("NestedModule", () => new M)
    val dut = annos.collectFirst { case DesignAnnotation(dut) => dut }.get.asInstanceOf[M]
    val allTargets = finalTargetMap(annos)
    allTargets(dut.a.toAbsoluteTarget) should be(Seq(refTarget("M", "a")))
    allTargets(dut.b.toAbsoluteTarget) should be(
      Seq(
        refTarget("M", "b_0"),
        refTarget("M", "b_1")
      )
    )
    allTargets(dut.b(0).toAbsoluteTarget) should be(Seq(refTarget("M", "b_0")))
    allTargets(dut.b(1).toAbsoluteTarget) should be(Seq(refTarget("M", "b_1")))
  }
}
