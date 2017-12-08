// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.core.Binding.BindingException
import chisel3.core.CompileOptions._

class CompileOptionsSpec extends ChiselFlatSpec {

  abstract class StrictModule extends Module()(chisel3.core.ExplicitCompileOptions.Strict)
  abstract class NotStrictModule extends Module()(chisel3.core.ExplicitCompileOptions.NotStrict)

  class SmallBundle extends Bundle {
    val f1 = UInt(4.W)
    val f2 = UInt(5.W)
    override def cloneType: this.type = (new SmallBundle).asInstanceOf[this.type]
  }
  class BigBundle extends SmallBundle {
    val f3 = UInt(6.W)
    override def cloneType: this.type = (new BigBundle).asInstanceOf[this.type]
  }

  "A Module with missing bundle fields when compiled with implicit Strict.CompileOption " should "throw an exception" in {
    a [ChiselException] should be thrownBy {
      import chisel3.core.ExplicitCompileOptions.Strict

      class ConnectFieldMismatchModule extends Module {
        val io = IO(new Bundle {
          val in = Input(new SmallBundle)
          val out = Output(new BigBundle)
        })
        io.out := io.in
      }
      elaborate { new ConnectFieldMismatchModule() }
    }
  }

  "A Module with missing bundle fields when compiled with implicit NotStrict.CompileOption " should "not throw an exception" in {
    import chisel3.core.ExplicitCompileOptions.NotStrict

    class ConnectFieldMismatchModule extends Module {
      val io = IO(new Bundle {
        val in = Input(new SmallBundle)
        val out = Output(new BigBundle)
      })
      io.out := io.in
    }
    elaborate { new ConnectFieldMismatchModule() }
  }

  "A Module in which a Reg is created with a bound type when compiled with implicit Strict.CompileOption " should "throw an exception" in {
    a [BindingException] should be thrownBy {
      import chisel3.core.ExplicitCompileOptions.Strict

      class CreateRegFromBoundTypeModule extends Module {
        val io = IO(new Bundle {
          val in = Input(new SmallBundle)
          val out = Output(new BigBundle)
        })
        val badReg = Reg(7.U(4.W))
      }
      elaborate { new CreateRegFromBoundTypeModule() }
    }
  }

  "A Module in which a Reg is created with a bound type when compiled with implicit NotStrict.CompileOption " should "not throw an exception" in {
    import chisel3.core.ExplicitCompileOptions.NotStrict

    class CreateRegFromBoundTypeModule extends Module {
      val io = IO(new Bundle {
        val in = Input(new SmallBundle)
        val out = Output(new BigBundle)
      })
      val badReg = Reg(7.U(4.W))
    }
    elaborate { new CreateRegFromBoundTypeModule() }
  }

  "A Module with wrapped IO when compiled with implicit Strict.CompileOption " should "not throw an exception" in {
    import chisel3.core.ExplicitCompileOptions.Strict

    class RequireIOWrapModule extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(32.W))
        val out = Output(Bool())
      })
      io.out := io.in(1)
    }
    elaborate { new RequireIOWrapModule() }
  }

  "A Module with unwrapped IO when compiled with implicit Strict.CompileOption " should "throw an exception" in {
    a [BindingException] should be thrownBy {
      import chisel3.core.ExplicitCompileOptions.Strict

      class RequireIOWrapModule extends Module {
        val io = new Bundle {
          val in = Input(UInt(32.W))
          val out = Output(Bool())
        }
        io.out := io.in(1)
      }
      elaborate {
        new RequireIOWrapModule()
      }
    }
  }

  "A Module connecting output as source to input as sink when compiled with implicit Strict.CompileOption " should "throw an exception" in {
    a [ChiselException] should be thrownBy {
      import chisel3.core.ExplicitCompileOptions.Strict

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
      elaborate { new SwappedConnectionModule() }
    }
  }

  "A Module connecting output as source to input as sink when compiled with implicit NotStrict.CompileOption " should "not throw an exception" in {
    import chisel3.core.ExplicitCompileOptions.NotStrict

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
    elaborate { new SwappedConnectionModule() }
  }

  "A Module with directionless connections when compiled with implicit Strict.CompileOption " should "throw an exception" in {
    a [ChiselException] should be thrownBy {
      // Verify we can suppress the inclusion of default compileOptions
      import Chisel.{defaultCompileOptions => _, _}
      import chisel3.core.ExplicitCompileOptions.Strict

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
      elaborate { new DirectionLessConnectionModule() }
    }
  }

  "A Module with directionless connections when compiled with implicit NotStrict.CompileOption " should "not throw an exception" in {
    import chisel3.core.ExplicitCompileOptions.NotStrict

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
    elaborate { new DirectionLessConnectionModule() }
  }
}
