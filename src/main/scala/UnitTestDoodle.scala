import chisel3._
import chisel3.test.RootTest
import chisel3.util.HasBlackBoxInline
import chisel3.experimental.{IntrinsicModule, ExtModule}
import chisel3.experimental.hierarchy._
import chisel3.ltl._
import circt.stage.ChiselStage
import scala.language.reflectiveCalls
import scala.collection.mutable.ArrayBuffer

import Sequence._

// Some configuration object passed to Foo that is hard to figure out otherwise.
// (Imagine it's hard to enumerate properly.)
case class FooConfig(
  val forbiddenInputValues: Seq[UInt] = Seq(),
)

// Some module that we're instantiating multiple times with different
// parameters. Also contains inline unit tests.
@instantiable
class Foo(width: Int, config: FooConfig) extends RawModule {
  override def desiredName = s"Foo${width}"
  val parent = this // allows tests to capture the parent module

  @public val dataIn = IO(Input(UInt(width.W)))
  @public val dataOut = IO(Output(UInt(width.W)))

  dataOut := dataIn +% 1.U

  // Some asserts to check that the input is not one of the forbidden values.
  config.forbiddenInputValues.foreach {
    value => AssertProperty(dataIn =/= value)
  }

  // Perform a sanity check with the input tied to zero.
  if (width < 4) {
    test {
      class TestProgramA(width: Int, expectedValue: Int) extends BlackBox(
        Map("W" -> width, "K" -> expectedValue)
      ) with HasBlackBoxInline {
        val io = IO(new Bundle {
          val dut_dataIn = UInt(width.W)
          val dut_dataOut = Flipped(UInt(width.W))
        })
        setInline("TestProgramA.sv", """
        |program TestProgramA #(
        |  parameter int W,
        |  parameter int K
        |)(
        |  output [W-1:0] dut_dataIn,
        |  input  [W-1:0] dut_dataOut
        |);
        |  initial begin
        |    dut_dataIn = 0;
        |    #1ns;
        |    repeat(10) assert(dut_dataOut == K);
        |    $display("Zero input sanity check successful.");
        |  end
        |endprogram
        """.stripMargin)
      }

      val dut = Instance(this.toDefinition)
      val testProgram = Module(new TestProgramA(width, 1))
      dut.dataIn := testProgram.io.dut_dataIn
      testProgram.io.dut_dataOut := dut.dataOut
    }
  }

  // Formally check that hardware does what it promises.
  test {
    val dut = Instance(this.toDefinition)
    val some_input = IO(Input(chiselTypeOf(dataIn)))
    dut.dataIn := some_input
    AssertProperty(dut.dataOut === some_input +% 1.U)

    // The config specified some forbidden input values. Exclude those with
    // assumptions such that the asserts in the DUT don't fire.
    config.forbiddenInputValues.foreach {
      value => AssumeProperty(some_input =/= value)
    }
  }

  // Check that the module works for random inputs.
  if (width > 4) {
    test {
      class TestProgramC(width: Int) extends BlackBox(Map("W" -> width)) with HasBlackBoxInline {
        val io = IO(new Bundle {
          val clock = Clock()
          val randomValue = UInt(width.W)
        })
        setInline("TestProgramC.sv", """
        |program TestProgramC #(parameter int W) (
        |  output clock,
        |  output [W-1:0] randomValue,
        |);
        |  initial begin
        |    clock = 0;
        |    randomValue = 0;
        |    #10;
        |    repeat(100) begin
        |      clock = 1;
        |      randomValue = $random();
        |      #1;
        |      clock = 1;
        |      #1;
        |    end
        |    $display("Random input check successful.");
        |  end
        |endprogram
        """.stripMargin)
      }

      val dut = Instance(this.toDefinition)
      val testProgram = Module(new TestProgramC(width))
      dut.dataIn := testProgram.io.randomValue
      AssertProperty(
        dut.dataOut === testProgram.io.randomValue +% 1.U,
        clock=Some(testProgram.io.clock)
      )
    }
  }

  // Create a DPI testbench.
  test {
    class DPIIntrinsic extends ExtModule(Map("func_name" -> "module_Foo_test_driver")) {
      val dut_dataIn = IO(Output(UInt(width.W)))
      val dut_dataOut = IO(Input(UInt(width.W)))
    }
    val dut = Instance(this.toDefinition)
    val dpi = Module(new DPIIntrinsic)
    dut.dataIn := dpi.dut_dataIn
    dpi.dut_dataOut := dut.dataOut
  }

  // Check that two instances of `Foo` can play together nicely.
  test {
    val dut1 = Instance(this.toDefinition)
    val dut2 = Instance(this.toDefinition)
    dut1.dataIn := 3.U
    dut2.dataIn := dut1.dataOut
    AssertProperty(dut2.dataOut === 5.U)
  }
}

class Top extends RawModule {
  val f1 = Instantiate(new Foo(3, FooConfig()))
  val f2 = Instantiate(new Foo(3, FooConfig()))
  val f3 = Instantiate(new Foo(4, FooConfig()))
  val f4 = Instantiate(new Foo(4, FooConfig()))
  val f5 = Instantiate(new Foo(5, FooConfig(forbiddenInputValues = Seq(3.U, 4.U))))
  f1.dataIn := 0.U
  f2.dataIn := 0.U
  f3.dataIn := 0.U
  f4.dataIn := 0.U
  f5.dataIn := 0.U
}

object UnitTestDoodle {
  def main(args: Array[String]): Unit = {
    println(ChiselStage.emitCHIRRTL(new Top))
    ChiselStage.emitSystemVerilogFile(new Top)
  }
}

object MyRootTest1 extends RootTest {
  Definition(new Foo(6, FooConfig(forbiddenInputValues = Seq(3.U, 4.U))))
}

class MyRootTest2 extends RootTest {
  Definition(new Foo(13, FooConfig()))
  Definition(new Foo(14, FooConfig()))
}
