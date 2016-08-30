// See LICENSE for license details.

package chiselTests

import org.scalatest._
import chisel3._
import chisel3.core.Binding.BindingException
import chisel3.testers.BasicTester

class CompileOptionsSpec extends ChiselFlatSpec {

  class SmallBundle extends Bundle {
    val f1 = UInt(width = 4)
    val f2 = UInt(width = 5)
    override def cloneType: this.type = (new SmallBundle).asInstanceOf[this.type]
  }
  class BigBundle extends SmallBundle {
    val f3 = UInt(width = 6)
    override def cloneType: this.type = (new BigBundle).asInstanceOf[this.type]
  }

  "A Module with missing bundle fields when compiled with Strict.CompileOption " should "throw an exception" in {
    a [ChiselException] should be thrownBy {
      import chisel3.Strict.CompileOptions

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

  "A Module with missing bundle fields when compiled with NotStrict.CompileOption " should "not throw an exception" in {
    import chisel3.NotStrict.CompileOptions

    class ConnectFieldMismatchModule extends Module {
      val io = IO(new Bundle {
        val in = Input(new SmallBundle)
        val out = Output(new BigBundle)
      })
      io.out := io.in
    }
    elaborate { new ConnectFieldMismatchModule() }
  }

  "A Module in which a Reg is created with a bound type when compiled with Strict.CompileOption " should "throw an exception" in {
    a [BindingException] should be thrownBy {
      import chisel3.Strict.CompileOptions

      class CreateRegFromBoundTypeModule extends Module {
        val io = IO(new Bundle {
          val in = Input(new SmallBundle)
          val out = Output(new BigBundle)
        })
        val badReg = Reg(UInt(7, width=4))
      }
      elaborate { new CreateRegFromBoundTypeModule() }
    }
  }

  "A Module in which a Reg is created with a bound type when compiled with NotStrict.CompileOption " should "not throw an exception" in {
    import chisel3.NotStrict.CompileOptions

    class CreateRegFromBoundTypeModule extends Module {
      val io = IO(new Bundle {
        val in = Input(new SmallBundle)
        val out = Output(new BigBundle)
      })
      val badReg = Reg(UInt(7, width=4))
    }
    elaborate { new CreateRegFromBoundTypeModule() }
  }


  "A Module with wrapped IO when compiled with Strict.CompileOption " should "not throw an exception" in {
    import chisel3.Strict.CompileOptions

    class RequireIOWrapModule extends Module {
      val io = IO(new Bundle {
        val in = UInt(width = 32).asInput
        val out = Bool().asOutput
      })
      io.out := io.in(1)
    }
    elaborate { new RequireIOWrapModule() }
}

  "A Module with unwrapped IO when compiled with NotStrict.CompileOption " should "not throw an exception" in {
    import chisel3.NotStrict.CompileOptions

    class RequireIOWrapModule extends Module {
      val io = new Bundle {
        val in = UInt(width = 32).asInput
        val out = Bool().asOutput
      }
      io.out := io.in(1)
    }
      elaborate { new RequireIOWrapModule() }
  }

  "A Module connecting output as source to input as sink when compiled with Strict.CompileOption " should "throw an exception" in {
    a [ChiselException] should be thrownBy {
      import chisel3.Strict.CompileOptions

      class SimpleModule extends Module {
        val io = IO(new Bundle {
          val in = Input(UInt(width = 3))
          val out = Output(UInt(width = 4))
        })
      }
      class SwappedConnectionModule extends SimpleModule {
        val child = Module(new SimpleModule)
        io.in := child.io.out
      }
      elaborate { new SwappedConnectionModule() }
    }
  }

  "A Module connecting output as source to input as sink when compiled with NotStrict.CompileOption " should "not throw an exception" in {
    import chisel3.NotStrict.CompileOptions

    class SimpleModule extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(width = 3))
        val out = Output(UInt(width = 4))
      })
    }
    class SwappedConnectionModule extends SimpleModule {
      val child = Module(new SimpleModule)
      io.in := child.io.out
    }
    elaborate { new SwappedConnectionModule() }
  }

  "A Module with directionless connections when compiled with Strict.CompileOption " should "throw an exception" in {
    a [ChiselException] should be thrownBy {
      import chisel3.Strict.CompileOptions

      class SimpleModule extends Module {
        val io = IO(new Bundle {
          val in = Input(UInt(width = 3))
          val out = Output(UInt(width = 4))
        })
        val noDir = Wire(UInt(width = 3))
      }

      class DirectionLessConnectionModule extends SimpleModule {
        val a = UInt(0, width = 3)
        val b = Wire(UInt(width = 3))
        val child = Module(new SimpleModule)
        b := child.noDir
      }
      elaborate { new DirectionLessConnectionModule() }
    }
  }

  "A Module with directionless connections when compiled with NotStrict.CompileOption " should "not throw an exception" in {
    import chisel3.NotStrict.CompileOptions

    class SimpleModule extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(width = 3))
        val out = Output(UInt(width = 4))
      })
      val noDir = Wire(UInt(width = 3))
    }

    class DirectionLessConnectionModule extends SimpleModule {
      val a = UInt(0, width = 3)
      val b = Wire(UInt(width = 3))
      val child = Module(new SimpleModule)
      b := child.noDir
    }
    elaborate { new DirectionLessConnectionModule() }
  }
}
