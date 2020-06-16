// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.CompileOptions._
import chisel3.stage.ChiselStage

class CompileOptionsSpec extends ChiselFlatSpec with Utils {

  abstract class StrictModule extends Module()(chisel3.ExplicitCompileOptions.Strict)
  abstract class NotStrictModule extends Module()(chisel3.ExplicitCompileOptions.NotStrict)

  class SmallBundle extends Bundle {
    val f1 = UInt(4.W)
    val f2 = UInt(5.W)
    override def cloneType: this.type = (new SmallBundle).asInstanceOf[this.type]
  }
  class BigBundle extends SmallBundle {
    val f3 = UInt(6.W)
    override def cloneType: this.type = (new BigBundle).asInstanceOf[this.type]
  }

  // scalastyle:off line.size.limit
  "A Module with missing bundle fields when compiled with implicit Strict.CompileOption " should "throw an exception" in {
    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      import chisel3.ExplicitCompileOptions.Strict

      class ConnectFieldMismatchModule extends Module {
        val io = IO(new Bundle {
          val in = Input(new SmallBundle)
          val out = Output(new BigBundle)
        })
        io.out := io.in
      }
      ChiselStage.elaborate { new ConnectFieldMismatchModule() }
    }
  }

  "A Module with missing bundle fields when compiled with implicit NotStrict.CompileOption " should "not throw an exception" in {
    import chisel3.ExplicitCompileOptions.NotStrict

    class ConnectFieldMismatchModule extends Module {
      val io = IO(new Bundle {
        val in = Input(new SmallBundle)
        val out = Output(new BigBundle)
      })
      io.out := io.in
    }
    ChiselStage.elaborate { new ConnectFieldMismatchModule() }
  }

  "A Module in which a Reg is created with a bound type when compiled with implicit Strict.CompileOption " should "throw an exception" in {
    a [BindingException] should be thrownBy extractCause[BindingException] {
      import chisel3.ExplicitCompileOptions.Strict

      class CreateRegFromBoundTypeModule extends Module {
        val io = IO(new Bundle {
          val in = Input(new SmallBundle)
          val out = Output(new BigBundle)
        })
        val badReg = Reg(7.U(4.W))
      }
      ChiselStage.elaborate { new CreateRegFromBoundTypeModule() }
    }
  }

  "A Module in which a Reg is created with a bound type when compiled with implicit NotStrict.CompileOption " should "not throw an exception" in {
    import chisel3.ExplicitCompileOptions.NotStrict

    class CreateRegFromBoundTypeModule extends Module {
      val io = IO(new Bundle {
        val in = Input(new SmallBundle)
        val out = Output(new BigBundle)
      })
      val badReg = Reg(7.U(4.W))
    }
    ChiselStage.elaborate { new CreateRegFromBoundTypeModule() }
  }

  "A Module with wrapped IO when compiled with implicit Strict.CompileOption " should "not throw an exception" in {
    import chisel3.ExplicitCompileOptions.Strict

    class RequireIOWrapModule extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(32.W))
        val out = Output(Bool())
      })
      io.out := io.in(1)
    }
    ChiselStage.elaborate { new RequireIOWrapModule() }
  }

  "A Module with unwrapped IO when compiled with implicit Strict.CompileOption " should "throw an exception" in {
    a [BindingException] should be thrownBy extractCause[BindingException] {
      import chisel3.ExplicitCompileOptions.Strict

      class RequireIOWrapModule extends Module {
        val io = new Bundle {
          val in = Input(UInt(32.W))
          val out = Output(Bool())
        }
        io.out := io.in(1)
      }
      ChiselStage.elaborate {
        new RequireIOWrapModule()
      }
    }
  }

  "A Module connecting output as source to input as sink when compiled with implicit Strict.CompileOption " should "throw an exception" in {
    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      import chisel3.ExplicitCompileOptions.Strict

      class SimpleModule extends Module {
        val io = IO(new Bundle {
          val in = Input(UInt(3.W))
          val out = Output(UInt(4.W))
        })
      }
      class SwappedConnectionModule extends SimpleModule {
        val child = Module(new SimpleModule)
        io.in := child.io.out
      }
      ChiselStage.elaborate { new SwappedConnectionModule() }
    }
  }

  "A Module connecting output as source to input as sink when compiled with implicit NotStrict.CompileOption " should "not throw an exception" in {
    import chisel3.ExplicitCompileOptions.NotStrict

    class SimpleModule extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(3.W))
        val out = Output(UInt(4.W))
      })
    }
    class SwappedConnectionModule extends SimpleModule {
      val child = Module(new SimpleModule)
      io.in := child.io.out
    }
    ChiselStage.elaborate { new SwappedConnectionModule() }
  }

  "A Module with directionless connections when compiled with implicit Strict.CompileOption " should "throw an exception" in {
    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      // Verify we can suppress the inclusion of default compileOptions
      import Chisel.{defaultCompileOptions => _}
      import chisel3.ExplicitCompileOptions.Strict

      class SimpleModule extends Module {
        val io = IO(new Bundle {
          val in = Input(UInt(3.W))
          val out = Output(UInt(4.W))
        })
        val noDir = Wire(UInt(3.W))
      }

      class DirectionLessConnectionModule extends SimpleModule {
        val a = 0.U(3.W)
        val b = Wire(UInt(3.W))
        val child = Module(new SimpleModule)
        b := child.noDir
      }
      ChiselStage.elaborate { new DirectionLessConnectionModule() }
    }
  }

  "A Module with directionless connections when compiled with implicit NotStrict.CompileOption " should "not throw an exception" in {
    import chisel3.ExplicitCompileOptions.NotStrict

    class SimpleModule extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(3.W))
        val out = Output(UInt(4.W))
      })
      val noDir = Wire(UInt(3.W))
    }

    class DirectionLessConnectionModule extends SimpleModule {
      val a = 0.U(3.W)
      val b = Wire(UInt(3.W))
      val child = Module(new SimpleModule)
      b := child.noDir
    }
    ChiselStage.elaborate { new DirectionLessConnectionModule() }
  }
  // scalastyle:on line.size.limit
}
