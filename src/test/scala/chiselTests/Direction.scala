// SPDX-License-Identifier: Apache-2.0

package chiselTests

import org.scalatest._
import org.scalatest.matchers.should.Matchers
import chisel3._
import chisel3.experimental.{ExtModule, OpaqueType}

import circt.stage.ChiselStage
import scala.collection.immutable.SeqMap

class DirectionedBundle extends Bundle {
  val in = Input(UInt(32.W))
  val out = Output(UInt(32.W))
}

class DirectionHaver extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(32.W))
    val out = Output(UInt(32.W))
    val inBundle = Input(new DirectionedBundle) // should override elements
    val outBundle = Output(new DirectionedBundle) // should override elements
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
    ChiselStage.emitCHIRRTL(new GoodDirection)
  }

  property("Inputs should not be assignable") {
    a[Exception] should be thrownBy extractCause[Exception] {
      ChiselStage.emitCHIRRTL(new BadDirection)
    }
    a[Exception] should be thrownBy extractCause[Exception] {
      ChiselStage.emitCHIRRTL(new BadSubDirection)
    }
  }

  property("Top-level forced outputs should be assignable") {
    ChiselStage.emitCHIRRTL(new TopDirectionOutput)
  }

  property("Empty Vecs with directioned sample_element should not cause direction errors") {
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = Vec(0, Output(UInt(8.W)))
      })
    })
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = Flipped(Vec(0, Output(UInt(8.W))))
      })
    })
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = Output(Vec(0, UInt(8.W)))
      })
    })
  }

  property(
    "Empty Vecs with no direction on the sample_element should not cause direction errors, as Chisel and chisel3 directions are merged"
  ) {
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = Vec(0, UInt(8.W))
      })
    })
  }

  property("Empty Bundles should not cause direction errors") {
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = new Bundle {}
      })
    })
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = Flipped(new Bundle {})
      })
    })
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = new Bundle {
          val y = if (false) Some(Input(UInt(8.W))) else None
        }
      })
    })
  }

  property(
    "Explicitly directioned but empty Bundles should not cause direction errors because Chisel and chisel3 directionality are merged"
  ) {
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val foo = UInt(8.W)
        val x = Input(new Bundle {})
      })
    })
  }

  import chisel3.experimental.Direction
  import chisel3.reflect.DataMirror

  property("Flipped should flip the specified direction of a Bundle") {
    class MyBundle extends Bundle {
      val out = Output(UInt(8.W))
      val in = Input(UInt(8.W))
    }
    class Top extends Module {
      val foo = IO(Flipped(new MyBundle))
      // Where I come from, referential transparency is a good thing
      val fooType = chiselTypeOf(foo)
      val fizz = IO(Flipped(fooType))
      val buzz = IO(Flipped(chiselTypeOf(foo)))

      DataMirror.specifiedDirectionOf(foo) should be(SpecifiedDirection.Flip)
      DataMirror.specifiedDirectionOf(fizz) should be(SpecifiedDirection.Unspecified)
      DataMirror.specifiedDirectionOf(buzz) should be(SpecifiedDirection.Unspecified)
      DataMirror.directionOf(foo) should be(Direction.Bidirectional(Direction.Flipped))
      DataMirror.directionOf(fizz) should be(Direction.Bidirectional(Direction.Default))
      DataMirror.directionOf(buzz) should be(Direction.Bidirectional(Direction.Default))
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new Top)
    chirrtl should include("input foo")
    chirrtl should include("output fizz")
    chirrtl should include("output buzz")
  }

  property("Directions should be preserved through cloning and binding of Bundles") {
    ChiselStage.emitCHIRRTL(new Module {
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
    }.asInstanceOf[Module]) // The cast works around weird reflection behavior (bug?)
  }

  property("Directions should be preserved through cloning and binding of Vecs") {
    ChiselStage.emitCHIRRTL(new Module {
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
    }.asInstanceOf[Module]) // The cast works around weird reflection behavior (bug?)
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

      val flippedVecVecFlipped = IO(Flipped(Vec(2, Vec(1, Flipped(new MyBundle)))))
      flippedVecVecFlipped <> DontCare
      assert(DataMirror.directionOf(flippedVecVecFlipped.head.head.a) == Direction.Input)
      assert(DataMirror.directionOf(flippedVecVecFlipped.head.head.b) == Direction.Output)
      assert(DataMirror.directionOf(flippedVecVecFlipped(index).head.a) == Direction.Input)
      assert(DataMirror.directionOf(flippedVecVecFlipped(index).head.b) == Direction.Output)
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)

    assert(chirrtl.contains("output regularVec : { flip a : UInt<1>, b : UInt<1>}[2]"))
    assert(chirrtl.contains("input vecFlipped : { flip a : UInt<1>, b : UInt<1>}[2]"))
    assert(chirrtl.contains("input flippedVec : { flip a : UInt<1>, b : UInt<1>}[2]"))
    assert(chirrtl.contains("output flippedVecFlipped : { flip a : UInt<1>, b : UInt<1>}[2]"))
    assert(chirrtl.contains("output flippedVecVecFlipped : { flip a : UInt<1>, b : UInt<1>}[1][2]"))
  }

  property("Using Vec and Flipped together should calculate directions properly for an ExtModule") {
    class MyBundle extends Bundle {
      val a = Input(Bool())
      val b = Output(Bool())
    }
    class MyBlackBox extends ExtModule {
      val regularVec = IO(Vec(2, new MyBundle))
      assert(DataMirror.directionOf(regularVec.head.a) == Direction.Input)
      assert(DataMirror.directionOf(regularVec.head.b) == Direction.Output)

      val vecFlipped = IO(Vec(2, Flipped(new MyBundle)))
      assert(DataMirror.directionOf(vecFlipped.head.a) == Direction.Output)
      assert(DataMirror.directionOf(vecFlipped.head.b) == Direction.Input)

      val flippedVec = IO(Flipped(Vec(2, new MyBundle)))
      assert(DataMirror.directionOf(flippedVec.head.a) == Direction.Output)
      assert(DataMirror.directionOf(flippedVec.head.b) == Direction.Input)

      // Flipped(Vec(Flipped())) should be equal to non-flipped.
      val flippedVecFlipped = IO(Flipped(Vec(2, Flipped(new MyBundle))))
      assert(DataMirror.directionOf(flippedVecFlipped.head.a) == Direction.Input)
      assert(DataMirror.directionOf(flippedVecFlipped.head.b) == Direction.Output)

      val flippedVecVecFlipped = IO(Flipped(Vec(2, Vec(1, Flipped(new MyBundle)))))
      assert(DataMirror.directionOf(flippedVecVecFlipped.head.head.a) == Direction.Input)
      assert(DataMirror.directionOf(flippedVecVecFlipped.head.head.b) == Direction.Output)
    }
    class MyModule extends RawModule {
      val child = Module(new MyBlackBox)
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)

    assert(chirrtl.contains("output regularVec : { flip a : UInt<1>, b : UInt<1>}[2]"))
    assert(chirrtl.contains("input vecFlipped : { flip a : UInt<1>, b : UInt<1>}[2]"))
    assert(chirrtl.contains("input flippedVec : { flip a : UInt<1>, b : UInt<1>}[2]"))
    assert(chirrtl.contains("output flippedVecFlipped : { flip a : UInt<1>, b : UInt<1>}[2]"))
    assert(chirrtl.contains("output flippedVecVecFlipped : { flip a : UInt<1>, b : UInt<1>}[1][2]"))
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

    val emitted: String = ChiselStage.emitCHIRRTL(new MyModule)
    val firrtl:  String = ChiselStage.convert(new MyModule).serialize

    // Check that emitted directions are correct.
    Seq(emitted, firrtl).foreach { o =>
      {
        // Chisel Emitter formats spacing a little differently than the
        // FIRRTL Emitter :-(
        val s = o.replace("{a", "{ a")
        assert(s.contains("input inputVec : { a : UInt<1>, b : UInt<1>}[2]"))
        assert(s.contains("input vecInput : { a : UInt<1>, b : UInt<1>}[2]"))
        assert(s.contains("input vecInputFlipped : { a : UInt<1>, b : UInt<1>}[2]"))
        assert(s.contains("output outputVec : { a : UInt<1>, b : UInt<1>}[2]"))
        assert(s.contains("output vecOutput : { a : UInt<1>, b : UInt<1>}[2]"))
        assert(s.contains("output vecOutputFlipped : { a : UInt<1>, b : UInt<1>}[2]"))
      }
    }
  }

  property("Using OpaqueTypes and Flipped together should calculate directions properly") {
    import chiselTests.experimental.OpaqueTypeSpec.{Boxed, Unboxed}
    class MyModule extends RawModule {
      val unboxedFlipped = IO(new Unboxed(Flipped(UInt(8.W))))
      assert(DataMirror.directionOf(unboxedFlipped.underlying) == Direction.Input)

      val flippedUnboxedFlipped = IO(Flipped(new Unboxed(Flipped(UInt(8.W)))))
      assert(DataMirror.directionOf(flippedUnboxedFlipped.underlying) == Direction.Output)

      // It needs to be recursive
      val unboxedUnboxedFlipped = IO(new Unboxed(new Unboxed(Flipped(UInt(8.W)))))
      assert(DataMirror.directionOf(unboxedUnboxedFlipped.underlying.underlying) == Direction.Input)

      val flippedUnboxedUnboxedFlipped = IO(Flipped(new Unboxed(new Unboxed(Flipped(UInt(8.W))))))
      assert(DataMirror.directionOf(flippedUnboxedUnboxedFlipped.underlying.underlying) == Direction.Output)

      // It should also work when nested inside of another Bundle
      val boxedUnboxedFlipped = IO(new Boxed(new Unboxed(Flipped(UInt(8.W)))))
      assert(DataMirror.directionOf(boxedUnboxedFlipped.underlying.underlying) == Direction.Input)

      val flippedBoxedUnboxedFlipped = IO(Flipped(new Boxed(new Unboxed(Flipped(UInt(8.W))))))
      assert(DataMirror.directionOf(flippedBoxedUnboxedFlipped.underlying.underlying) == Direction.Output)

      // It also needs to be recursive when inside of another bundle
      val boxedUnboxedUnboxedFlipped = IO(new Boxed(new Unboxed(new Unboxed(Flipped(UInt(8.W))))))
      assert(DataMirror.directionOf(boxedUnboxedUnboxedFlipped.underlying.underlying.underlying) == Direction.Input)

      val flippedBoxedUnboxedUnboxedFlipped = IO(Flipped(new Boxed(new Unboxed(new Unboxed(Flipped(UInt(8.W)))))))
      assert(
        DataMirror.directionOf(flippedBoxedUnboxedUnboxedFlipped.underlying.underlying.underlying) == Direction.Output
      )

    }

    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    assert(chirrtl.contains("input unboxedFlipped : UInt<8>"))
    assert(chirrtl.contains("output flippedUnboxedFlipped : UInt<8>"))
    assert(chirrtl.contains("input unboxedUnboxedFlipped : UInt<8>"))
    assert(chirrtl.contains("output flippedUnboxedUnboxedFlipped : UInt<8>"))
    assert(chirrtl.contains("output boxedUnboxedFlipped : { flip underlying : UInt<8>}"))
    assert(chirrtl.contains("input flippedBoxedUnboxedFlipped : { flip underlying : UInt<8>}"))
    assert(chirrtl.contains("output boxedUnboxedUnboxedFlipped : { flip underlying : UInt<8>}"))
    assert(chirrtl.contains("input flippedBoxedUnboxedUnboxedFlipped : { flip underlying : UInt<8>}"))
  }

  property("Can now describe a Decoupled bundle using Flipped, not Input/Output in chisel3") {
    class Decoupled extends Bundle {
      val bits = UInt(3.W)
      val valid = Bool()
      val ready = Flipped(Bool())
    }
    class MyModule extends RawModule {
      val incoming = IO(Flipped(new Decoupled))
      val outgoing = IO(new Decoupled)

      outgoing <> incoming
    }

    val emitted: String = ChiselStage.emitCHIRRTL(new MyModule)

    // Check that emitted directions are correct.
    assert(emitted.contains("input incoming : { bits : UInt<3>, valid : UInt<1>, flip ready : UInt<1>}"))
    assert(emitted.contains("output outgoing : { bits : UInt<3>, valid : UInt<1>, flip ready : UInt<1>}"))
    assert(emitted.contains("connect outgoing, incoming"))
  }
  property("Can now mix Input/Output and Flipped within the same bundle") {
    class Decoupled extends Bundle {
      val bits = UInt(3.W)
      val valid = Bool()
      val ready = Flipped(Bool())
    }
    class DecoupledAndMonitor extends Bundle {
      val producer = new Decoupled()
      val consumer = Flipped(new Decoupled())
      val monitor = Input(new Decoupled()) // Same as Flipped(stripFlipsIn(..))
      val driver = Output(new Decoupled()) // Same as stripFlipsIn(..)
    }
    class MyModule extends RawModule {
      val io = IO(Flipped(new DecoupledAndMonitor()))
      io.consumer <> io.producer
      io.monitor.bits := io.driver.bits
      io.monitor.valid := io.driver.valid
      io.monitor.ready := io.driver.ready
    }

    val emitted: String = ChiselStage.emitCHIRRTL(new MyModule)

    assert(
      emitted.contains(
        "input io : { producer : { bits : UInt<3>, valid : UInt<1>, flip ready : UInt<1>}, flip consumer : { bits : UInt<3>, valid : UInt<1>, flip ready : UInt<1>}, flip monitor : { bits : UInt<3>, valid : UInt<1>, ready : UInt<1>}, driver : { bits : UInt<3>, valid : UInt<1>, ready : UInt<1>}}"
      )
    )
    assert(emitted.contains("connect io.consumer, io.producer"))
    assert(emitted.contains("connect io.monitor.bits, io.driver.bits"))
    assert(emitted.contains("connect io.monitor.valid, io.driver.valid"))
    assert(emitted.contains("connect io.monitor.ready, io.driver.ready"))
  }
  property("Bugfix: marking Vec fields with mixed directionality as Output/Input clears inner directions") {
    class Decoupled extends Bundle {
      val bits = UInt(3.W)
      val valid = Bool()
      val ready = Flipped(Bool())
    }
    class Coercing extends Bundle {
      val source = Output(Vec(1, new Decoupled()))
      val sink = Input(Vec(1, new Decoupled()))
    }
    class MyModule extends RawModule {
      val io = IO(new Coercing())
      val source = IO(Output(Vec(1, new Decoupled())))
      val sink = IO(Input(Vec(1, new Decoupled())))
    }

    val emitted: String = ChiselStage.emitCHIRRTL(new MyModule)

    assert(
      emitted.contains(
        "output io : { source : { bits : UInt<3>, valid : UInt<1>, ready : UInt<1>}[1], flip sink : { bits : UInt<3>, valid : UInt<1>, ready : UInt<1>}[1]}"
      )
    )
    assert(
      emitted.contains(
        "output source : { bits : UInt<3>, valid : UInt<1>, ready : UInt<1>}[1]"
      )
    )
    assert(
      emitted.contains(
        "input sink : { bits : UInt<3>, valid : UInt<1>, ready : UInt<1>}[1]"
      )
    )
  }
  property("Bugfix: clearing all flips inside an opaque type") {

    class Decoupled extends Bundle {
      val bits = UInt(3.W)
      val valid = Bool()
      val ready = Flipped(Bool())
    }
    class MyOpaqueType extends Record with OpaqueType {
      val k = new Decoupled()
      val elements = SeqMap("" -> k)
    }
    class MyModule extends RawModule {
      val w = Wire(new MyOpaqueType())
    }

    val emitted: String = ChiselStage.emitCHIRRTL(new MyModule)

    assert(
      emitted.contains(
        "wire w : { bits : UInt<3>, valid : UInt<1>, flip ready : UInt<1>}"
      )
    )
  }
}
