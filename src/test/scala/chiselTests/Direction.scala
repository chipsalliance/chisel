// See LICENSE for license details.

package chiselTests

import org.scalatest._
import chisel3._
import chisel3.util.Decoupled

class DirectionedBundle extends Bundle {
  val in = Input(UInt(32.W))
  val out = Output(UInt(32.W))
}

class DirectionHaver extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(32.W))
    val out = Output(UInt(32.W))
    val inBundle = Input(new DirectionedBundle)  // should override elements
    val outBundle = Output(new DirectionedBundle)  // should override elements
  })
}

class GoodDirection extends DirectionHaver {
  io.out := 0.U
  io.outBundle.in := 0.U
  io.outBundle.out := 0.U
}

class BadDirection extends DirectionHaver {
  io.in := 0.U
}

class BadSubDirection extends DirectionHaver {
  io.inBundle.out := 0.U
}

class TopDirectionOutput extends Module {
  val io = IO(Output(new DirectionedBundle))
  io.in := 42.U
  io.out := 117.U
}

class DirectionSpec extends ChiselPropSpec with Matchers {

  //TODO: In Chisel3 these are actually FIRRTL errors. Remove from tests?

  property("Outputs should be assignable") {
    elaborate(new GoodDirection)
  }

  property("Inputs should not be assignable") {
    a[Exception] should be thrownBy {
     elaborate(new BadDirection)
    }
    a[Exception] should be thrownBy {
     elaborate(new BadSubDirection)
    }
  }

  property("Top-level forced outputs should be assignable") {
    elaborate(new TopDirectionOutput)
  }

  property("Empty Vecs with directioned sample_element should not cause direction errors") {
    elaborate(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = Vec(0, Output(UInt(8.W)))
      })
    })
    elaborate(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = Flipped(Vec(0, Output(UInt(8.W))))
      })
    })
    elaborate(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = Output(Vec(0, UInt(8.W)))
      })
    })
  }

  property("Empty Vecs with no direction on the sample_element *should* cause direction errors") {
    an [Exception] should be thrownBy {
      elaborate(new Module {
        val io = IO(new Bundle {
          val foo = Input(UInt(8.W))
          val x = Vec(0, UInt(8.W))
        })
      })
    }
  }

  property("Empty Bundles should not cause direction errors") {
    elaborate(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = new Bundle {}
      })
    })
    elaborate(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = Flipped(new Bundle {})
      })
    })
    elaborate(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = new Bundle {
          val y = if (false) Some(Input(UInt(8.W))) else None
        }
      })
    })
  }

  property("Explicitly directioned but empty Bundles should cause direction errors") {
    an [Exception] should be thrownBy {
      elaborate(new Module {
        val io = IO(new Bundle {
          val foo = UInt(8.W)
          val x = Input(new Bundle {})
        })
      })
    }
  }

  import chisel3.experimental.{MultiIOModule, DataMirror, Direction}
  import chisel3.core.SpecifiedDirection

  property("Directions should be preserved through cloning and binding of Bundles") {
    elaborate(new MultiIOModule {
      class MyBundle extends Bundle {
        val foo = Input(UInt(8.W))
        val bar = Output(UInt(8.W))
      }
      class MyOuterBundle extends Bundle {
        val fizz = new MyBundle
        val buzz = Flipped(new MyBundle)
      }
      val a = new MyOuterBundle
      val b = IO(a)
      val specifiedDirs = Seq(
        a.fizz.foo -> SpecifiedDirection.Input,
        a.fizz.bar -> SpecifiedDirection.Output,
        a.fizz -> SpecifiedDirection.Unspecified,
        a.buzz.foo -> SpecifiedDirection.Input,
        a.buzz.bar -> SpecifiedDirection.Output,
        a.buzz -> SpecifiedDirection.Flip
      )
      val actualDirs = Seq(
        b.fizz.foo -> Direction.Input,
        b.fizz.bar -> Direction.Output,
        b.fizz -> Direction.Bidirectional(Direction.Default),
        b.buzz.foo -> Direction.Output,
        b.buzz.bar -> Direction.Input,
        b.buzz -> Direction.Bidirectional(Direction.Flipped)
      )
      for ((data, dir) <- specifiedDirs) {
        DataMirror.specifiedDirectionOf(data) shouldBe (dir)
      }
      for ((data, dir) <- actualDirs) {
        DataMirror.directionOf(data) shouldBe (dir)
      }
    }.asInstanceOf[MultiIOModule]) // The cast works around weird reflection behavior (bug?)
  }

  property("Directions should be preserved through cloning and binding of Vecs") {
    elaborate(new MultiIOModule {
      val a = Vec(1, Input(UInt(8.W)))
      val b = Vec(1, a)
      val c = Vec(1, Flipped(a))
      val io0 = IO(b)
      val io1 = IO(c)
      val specifiedDirs = Seq(
        a(0) -> SpecifiedDirection.Input,
        b(0)(0) -> SpecifiedDirection.Input,
        a -> SpecifiedDirection.Unspecified,
        b -> SpecifiedDirection.Unspecified,
        c(0) -> SpecifiedDirection.Flip,
        c(0)(0) -> SpecifiedDirection.Input,
        c -> SpecifiedDirection.Unspecified
      )
      val actualDirs = Seq(
        io0(0)(0) -> Direction.Input,
        io0(0) -> Direction.Input,
        io0 -> Direction.Input,
        io1(0)(0) -> Direction.Output,
        io1(0) -> Direction.Output,
        io1 -> Direction.Output
      )
      for ((data, dir) <- specifiedDirs) {
        DataMirror.specifiedDirectionOf(data) shouldBe (dir)
      }
      for ((data, dir) <- actualDirs) {
        DataMirror.directionOf(data) shouldBe (dir)
      }
    }.asInstanceOf[MultiIOModule]) // The cast works around weird reflection behavior (bug?)
  }
}
