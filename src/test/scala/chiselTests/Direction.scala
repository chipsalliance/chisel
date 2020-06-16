// See LICENSE for license details.

package chiselTests

import org.scalatest._
import chisel3._
import chisel3.stage.ChiselStage
import org.scalatest.matchers.should.Matchers

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

class DirectionSpec extends ChiselPropSpec with Matchers with Utils {

  //TODO: In Chisel3 these are actually FIRRTL errors. Remove from tests?

  property("Outputs should be assignable") {
    ChiselStage.elaborate(new GoodDirection)
  }

  property("Inputs should not be assignable") {
    a[Exception] should be thrownBy extractCause[Exception] {
     ChiselStage.elaborate(new BadDirection)
    }
    a[Exception] should be thrownBy extractCause[Exception] {
     ChiselStage.elaborate(new BadSubDirection)
    }
  }

  property("Top-level forced outputs should be assignable") {
    ChiselStage.elaborate(new TopDirectionOutput)
  }

  property("Empty Vecs with directioned sample_element should not cause direction errors") {
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = Vec(0, Output(UInt(8.W)))
      })
    })
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = Flipped(Vec(0, Output(UInt(8.W))))
      })
    })
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = Output(Vec(0, UInt(8.W)))
      })
    })
  }

  property("Empty Vecs with no direction on the sample_element *should* cause direction errors") {
    an [Exception] should be thrownBy extractCause[Exception] {
      ChiselStage.elaborate(new Module {
        val io = IO(new Bundle {
          val foo = Input(UInt(8.W))
          val x = Vec(0, UInt(8.W))
        })
      })
    }
  }

  property("Empty Bundles should not cause direction errors") {
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = new Bundle {}
      })
    })
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = Flipped(new Bundle {})
      })
    })
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = new Bundle {
          val y = if (false) Some(Input(UInt(8.W))) else None
        }
      })
    })
  }

  property("Explicitly directioned but empty Bundles should cause direction errors") {
    an [Exception] should be thrownBy extractCause[Exception] {
      ChiselStage.elaborate(new Module {
        val io = IO(new Bundle {
          val foo = UInt(8.W)
          val x = Input(new Bundle {})
        })
      })
    }
  }

  import chisel3.experimental.{DataMirror, Direction}

  property("Directions should be preserved through cloning and binding of Bundles") {
    ChiselStage.elaborate(new MultiIOModule {
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
    ChiselStage.elaborate(new MultiIOModule {
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

  property("Using Vec and Flipped together should calculate directions properly") {
    class MyModule extends RawModule {
      class MyBundle extends Bundle {
        val a = Input(Bool())
        val b = Output(Bool())
      }

      val index = IO(Input(UInt(1.W)))

      // Check all permutations of Vec and Flipped.
      val regularVec = IO(Vec(2, new MyBundle))
      regularVec <> DontCare
      assert(DataMirror.directionOf(regularVec.head.a) == Direction.Input)
      assert(DataMirror.directionOf(regularVec.head.b) == Direction.Output)
      assert(DataMirror.directionOf(regularVec(index).a) == Direction.Input)
      assert(DataMirror.directionOf(regularVec(index).b) == Direction.Output)

      val vecFlipped = IO(Vec(2, Flipped(new MyBundle)))
      vecFlipped <> DontCare
      assert(DataMirror.directionOf(vecFlipped.head.a) == Direction.Output)
      assert(DataMirror.directionOf(vecFlipped.head.b) == Direction.Input)
      assert(DataMirror.directionOf(vecFlipped(index).a) == Direction.Output)
      assert(DataMirror.directionOf(vecFlipped(index).b) == Direction.Input)

      val flippedVec = IO(Flipped(Vec(2, new MyBundle)))
      flippedVec <> DontCare
      assert(DataMirror.directionOf(flippedVec.head.a) == Direction.Output)
      assert(DataMirror.directionOf(flippedVec.head.b) == Direction.Input)
      assert(DataMirror.directionOf(flippedVec(index).a) == Direction.Output)
      assert(DataMirror.directionOf(flippedVec(index).b) == Direction.Input)

      // Flipped(Vec(Flipped())) should be equal to non-flipped.
      val flippedVecFlipped = IO(Flipped(Vec(2, Flipped(new MyBundle))))
      flippedVecFlipped <> DontCare
      assert(DataMirror.directionOf(flippedVecFlipped.head.a) == Direction.Input)
      assert(DataMirror.directionOf(flippedVecFlipped.head.b) == Direction.Output)
      assert(DataMirror.directionOf(flippedVecFlipped(index).a) == Direction.Input)
      assert(DataMirror.directionOf(flippedVecFlipped(index).b) == Direction.Output)
    }

    val emitted: String = (new ChiselStage).emitChirrtl(new MyModule)
    val firrtl: String = ChiselStage.convert(new MyModule).serialize

    // Check that emitted directions are correct.
    Seq(emitted, firrtl).foreach { o => {
      // Chisel Emitter formats spacing a little differently than the
      // FIRRTL Emitter :-(
      val s = o.replace("{flip a", "{ flip a")
      assert(s.contains("output regularVec : { flip a : UInt<1>, b : UInt<1>}[2]"))
      assert(s.contains("input vecFlipped : { flip a : UInt<1>, b : UInt<1>}[2]"))
      assert(s.contains("input flippedVec : { flip a : UInt<1>, b : UInt<1>}[2]"))
      assert(s.contains("output flippedVecFlipped : { flip a : UInt<1>, b : UInt<1>}[2]"))
    } }
  }

  property("Vec with Input/Output should calculate directions properly") {
    class MyModule extends RawModule {
      class MyBundle extends Bundle {
        val a = Input(Bool())
        val b = Output(Bool())
      }

      val index = IO(Input(UInt(1.W)))

      val inputVec = IO(Vec(2, Input(new MyBundle)))
      inputVec <> DontCare
      assert(DataMirror.directionOf(inputVec.head.a) == Direction.Input)
      assert(DataMirror.directionOf(inputVec.head.b) == Direction.Input)
      assert(DataMirror.directionOf(inputVec(index).a) == Direction.Input)
      assert(DataMirror.directionOf(inputVec(index).b) == Direction.Input)

      val vecInput = IO(Input(Vec(2, new MyBundle)))
      vecInput <> DontCare
      assert(DataMirror.directionOf(vecInput.head.a) == Direction.Input)
      assert(DataMirror.directionOf(vecInput.head.b) == Direction.Input)
      assert(DataMirror.directionOf(vecInput(index).a) == Direction.Input)
      assert(DataMirror.directionOf(vecInput(index).b) == Direction.Input)

      val vecInputFlipped = IO(Input(Vec(2, Flipped(new MyBundle))))
      vecInputFlipped <> DontCare
      assert(DataMirror.directionOf(vecInputFlipped.head.a) == Direction.Input)
      assert(DataMirror.directionOf(vecInputFlipped.head.b) == Direction.Input)
      assert(DataMirror.directionOf(vecInputFlipped(index).a) == Direction.Input)
      assert(DataMirror.directionOf(vecInputFlipped(index).b) == Direction.Input)

      val outputVec = IO(Vec(2, Output(new MyBundle)))
      outputVec <> DontCare
      assert(DataMirror.directionOf(outputVec.head.a) == Direction.Output)
      assert(DataMirror.directionOf(outputVec.head.b) == Direction.Output)
      assert(DataMirror.directionOf(outputVec(index).a) == Direction.Output)
      assert(DataMirror.directionOf(outputVec(index).b) == Direction.Output)

      val vecOutput = IO(Output(Vec(2, new MyBundle)))
      vecOutput <> DontCare
      assert(DataMirror.directionOf(vecOutput.head.a) == Direction.Output)
      assert(DataMirror.directionOf(vecOutput.head.b) == Direction.Output)
      assert(DataMirror.directionOf(vecOutput(index).a) == Direction.Output)
      assert(DataMirror.directionOf(vecOutput(index).b) == Direction.Output)

      val vecOutputFlipped = IO(Output(Vec(2, Flipped(new MyBundle))))
      vecOutputFlipped <> DontCare
      assert(DataMirror.directionOf(vecOutputFlipped.head.a) == Direction.Output)
      assert(DataMirror.directionOf(vecOutputFlipped.head.b) == Direction.Output)
      assert(DataMirror.directionOf(vecOutputFlipped(index).a) == Direction.Output)
      assert(DataMirror.directionOf(vecOutputFlipped(index).b) == Direction.Output)
    }

    val emitted: String = (new ChiselStage).emitChirrtl(new MyModule)
    val firrtl: String = ChiselStage.convert(new MyModule).serialize

    // Check that emitted directions are correct.
    Seq(emitted, firrtl).foreach { o => {
      // Chisel Emitter formats spacing a little differently than the
      // FIRRTL Emitter :-(
      val s = o.replace("{a", "{ a")
      assert(s.contains("input inputVec : { a : UInt<1>, b : UInt<1>}[2]"))
      assert(s.contains("input vecInput : { a : UInt<1>, b : UInt<1>}[2]"))
      assert(s.contains("input vecInputFlipped : { a : UInt<1>, b : UInt<1>}[2]"))
      assert(s.contains("output outputVec : { a : UInt<1>, b : UInt<1>}[2]"))
      assert(s.contains("output vecOutput : { a : UInt<1>, b : UInt<1>}[2]"))
      assert(s.contains("output vecOutputFlipped : { a : UInt<1>, b : UInt<1>}[2]"))
    } }
  }
}
