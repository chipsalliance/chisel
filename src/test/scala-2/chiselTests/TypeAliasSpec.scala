package chiselTests

import chisel3._
import circt.stage.ChiselStage
import chisel3.experimental.{HasTypeAlias, RecordAlias}

class TypeAliasSpec extends ChiselFlatSpec with Utils {
  "Bundles with opt-in alias names" should "have an emitted FIRRTL type alias" in {
    class Test extends Module {
      class FooBundle extends Bundle with HasTypeAlias {
        override def aliasName = RecordAlias(typeName)

        val x = UInt(8.W)
        val y = UInt(8.W)
      }

      val io = IO(new Bundle {
        val in = Input(new FooBundle)
        val out = Output(new FooBundle)
      })

      val w = Wire(new FooBundle)

      w :#= io.in
      io.out :#= w
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Test)

    chirrtl should include("type FooBundle = { x : UInt<8>, y : UInt<8>}")
    chirrtl should include("flip in : FooBundle")
    chirrtl should include("out : FooBundle")
    chirrtl should include("wire w : FooBundle")
  }

  "Bundles with opt-in FIRRTL type aliases" should "support nesting" in {
    class Test extends Module {
      class BarBundle extends Bundle with HasTypeAlias {
        override def aliasName = RecordAlias("Buzz")

        val y = UInt(8.W)
      }
      class FooBundle extends Bundle with HasTypeAlias {
        override def aliasName = RecordAlias("Fizz")

        val x = UInt(8.W)
        val bar = new BarBundle
      }

      val io = IO(new Bundle {
        val in = Input(new FooBundle)
        val out = Output(new FooBundle)
      })

      val w = Wire(new FooBundle)

      w :#= io.in
      io.out :#= w
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Test)

    chirrtl should include("type Fizz = { x : UInt<8>, bar : Buzz}")
    chirrtl should include("flip in : Fizz")
    chirrtl should include("out : Fizz")
    chirrtl should include("wire w : Fizz")
  }

  "Bundles with opt-in alias names" should "allow user-specified stripped suffixing" in {
    class Test extends Module {
      class FooBundle extends Bundle with HasTypeAlias {
        override def aliasName = RecordAlias(typeName, "_testSuffix")

        val x = Flipped(UInt(8.W))
        val y = UInt(8.W)
      }

      val io = IO(new Bundle {
        val in = Input(new FooBundle)
        val out = Output(new FooBundle)
      })

      io.out :#= io.in
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Test)

    chirrtl should include("type FooBundle_testSuffix = { x : UInt<8>, y : UInt<8>}")
    chirrtl should include("flip in : FooBundle_testSuffix")
    chirrtl should include("out : FooBundle_testSuffix")
  }

  "Bundles with opt-in FIRRTL type aliases" should "generate normal aliases for coerced, monodirectional/unflipped bundles" in {
    class MonoStrippedTest extends Module {
      class BarBundle extends Bundle with HasTypeAlias {
        override def aliasName = RecordAlias("Buzz")

        val x = UInt(8.W)
        val y = UInt(8.W)
      }

      val io = IO(new Bundle {
        val in = Input(new BarBundle)
        val out = Output(new BarBundle)
      })

      io.out :#= io.in
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new MonoStrippedTest)
    // Alias for the bundle since it wouldn't be stripped
    chirrtl should include("type Buzz = { x : UInt<8>, y : UInt<8>}")
  }

  "Bundles with opt-in FIRRTL type aliases" should "generate modified aliases for coerced, bidirectional bundles" in {
    class StandardStrippedTest extends Module {
      class BarBundle extends Bundle with HasTypeAlias {
        override def aliasName = RecordAlias("Buzz")

        val x = UInt(8.W)
        // This flip gets stripped by both Input() and Output()
        val y = Flipped(UInt(8.W))
      }

      val io = IO(new Bundle {
        val in = Input(new BarBundle)
        val out = Output(new BarBundle)
      })

      io.out :#= io.in
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new StandardStrippedTest)
    // No unmodified alias for the bidirectional bundle since it isn't actually bound
    chirrtl shouldNot include("type Buzz = { x : UInt<8>, flip y : UInt<8>}")
    // No unmodified alias for the stripped bundle
    chirrtl shouldNot include("type Buzz = { x : UInt<8>, y : UInt<8>}")
    // Modified alias for the stripped bundle
    chirrtl should include("type Buzz_stripped = { x : UInt<8>, y : UInt<8>}")
  }

  "Bundles with opt-in FIRRTL type aliases" should "generate modified aliases for coerced, monodirectional/flipped bundles" in {
    class FlippedMonoStrippedTest extends Module {
      class BarBundle extends Bundle with HasTypeAlias {
        override def aliasName = RecordAlias("Buzz")

        // These flips get stripped by both Input() and Output()
        val x = Flipped(UInt(8.W))
        val y = Flipped(UInt(8.W))
      }

      val io = IO(new Bundle {
        val in = Input(new BarBundle)
        val out = Output(new BarBundle)
      })

      io.out :#= io.in
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new FlippedMonoStrippedTest)
    // No unmodified alias for the flipped monodirectional bundle since it isn't actually bound
    chirrtl shouldNot include("type Buzz = { flip x : UInt<8>, flip y : UInt<8>}")
    // No unmodified alias for the stripped bundle
    chirrtl shouldNot include("type Buzz = { x : UInt<8>, y : UInt<8>}")
    // Modified alias for the stripped bundle
    chirrtl should include("type Buzz_stripped = { x : UInt<8>, y : UInt<8>}")
  }

  "Duplicate bundle type aliases with same structures" should "compile" in {
    class Test extends Module {
      // All three of these bundles are structurally equivalent in FIRRTL and thus
      // are equivalent, substitutable aliases for each other. Merge/dedup them into one
      class FooBundle extends Bundle with HasTypeAlias {
        override def aliasName = RecordAlias("IdenticalBundle")

        val x = UInt(8.W)
        val y = UInt(8.W)
      }
      class FooBarBundle extends Bundle with HasTypeAlias {
        override def aliasName = RecordAlias("IdenticalBundle")

        val x = UInt(8.W)
        val y = UInt(8.W)
      }
      class BarBundle extends Bundle with HasTypeAlias {
        override def aliasName = RecordAlias("IdenticalBundle")

        val x = UInt(8.W)
        val y = UInt(8.W)
      }

      val io = IO(new Bundle {
        val in = Input(new FooBundle)
        val out = Output(new BarBundle)
      })

      val w = Wire(new FooBarBundle)

      w.x :#= io.in.x
      w.y :#= io.in.y

      io.out.x :#= w.x
      io.out.y :#= w.y
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Test)

    chirrtl should include("type IdenticalBundle = { x : UInt<8>, y : UInt<8>}")
    chirrtl should include("output io : { flip in : IdenticalBundle, out : IdenticalBundle}")
    chirrtl should include("wire w : IdenticalBundle")
  }

  "Duplicate bundle type aliases with differing structures" should "error" in {
    val msg = (the[ChiselException] thrownBy extractCause[ChiselException] {
      class Test extends Module {
        // These bundles are structurally unequivalent and so must be aliased with different names.
        // Error if they share the same name
        class FooBundle extends Bundle with HasTypeAlias {
          override def aliasName = RecordAlias("DifferentBundle")

          val x = Bool()
          val y = UInt(8.W)
        }
        class BarBundle extends Bundle with HasTypeAlias {
          override def aliasName = RecordAlias("DifferentBundle")

          val x = SInt(8.W)
          val y = Bool()
        }

        val io = IO(new Bundle {
          val in = Input(new FooBundle)
          val out = Output(new BarBundle)
        })

        io.out.x := io.in.y
        io.out.y :#= io.in.x
      }

      val args = Array("--throw-on-first-error", "--full-stacktrace")
      val chirrtl = ChiselStage.emitCHIRRTL(new Test, args)
    }).getMessage

    msg should include(
      "Attempted to redeclare an existing type alias 'DifferentBundle' with a new Record structure"
    )
    msg should include("The alias was previously defined as:")
    msg should include("@[src/test/scala/chiselTests/TypeAliasSpec.scala")
  }

  "Bundles with unsanitary names" should "properly sanitize" in {
    class Test extends Module {
      class FooBundle extends Bundle with HasTypeAlias {
        // Sanitizes to '_'; the sanitized alias needs to be used in the aliasing algorithm
        // instead of the direct user-defined alias.
        override def aliasName = RecordAlias("")

        val x = UInt(8.W)
        val y = UInt(8.W)
      }

      val io = IO(new Bundle {
        val in = Input(new FooBundle)
        val out = Output(new FooBundle)
      })

      val w = Wire(new FooBundle)

      w :#= io.in
      io.out :#= w
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Test)

    chirrtl should include("type _ = { x : UInt<8>, y : UInt<8>}")
    chirrtl should include("output io : { flip in : _, out : _}")
    chirrtl should include("wire w : _")
  }

  "Bundle type aliases overriding an existing FIRRTL type" should "error" in {
    // Special keywords/types specified in the FIRRTL spec.
    // These result in parser errors and should not be allowed by Chisel
    // Duplicated from the same list in the `internal` package object, which is private to chisel3
    val firrtlKeywords = Seq(
      "FIRRTL",
      "Clock",
      "UInt",
      "Reset",
      "AsyncReset",
      "Analog",
      "Probe",
      "RWProbe",
      "version",
      "type",
      "circuit",
      "parameter",
      "input",
      "output",
      "extmodule",
      "module",
      "intmodule",
      "intrinsic",
      "defname",
      "const",
      "flip",
      "reg",
      "smem",
      "cmem",
      "mport",
      "define",
      "attach",
      "inst",
      "of",
      "reset",
      "printf",
      "skip",
      "node"
    )

    // Prevent statements like type Clock = { ... }
    firrtlKeywords.map { tpe =>
      (the[ChiselException] thrownBy extractCause[ChiselException] {
        class Test(val firrtlType: String) extends Module {
          class FooBundle extends Bundle with HasTypeAlias {
            override def aliasName = RecordAlias(firrtlType)

            val x = UInt(8.W)
          }

          val io = IO(new Bundle {
            val in = Input(new FooBundle)
            val out = Output(new FooBundle)
          })

          io.out.x :#= io.in.x
        }

        val args = Array("--throw-on-first-error", "--full-stacktrace")
        val chirrtl = ChiselStage.emitCHIRRTL(new Test(tpe), args)
      }).getMessage should include(
        s"Attempted to use an illegal word '$tpe' for a type alias. Chisel does not automatically disambiguate these aliases at this time."
      )
    }
  }

  "Nested bundles with nested FIRRTL type aliases" should "generate with previously-defined aliases" in {
    class Test extends Module {
      class FooBundle(width: Int) extends Bundle with HasTypeAlias {
        override def aliasName = RecordAlias("FooBundle")
        val x = UInt(width.W)
      }
      class BarBundle(width: Int) extends Bundle with HasTypeAlias {
        override def aliasName = RecordAlias("BarBundle")
        val y = SInt(width.W)
      }
      // All three of these bundles are structurally equivalent in FIRRTL and thus
      // are equivalent, substitutable aliases for each other. Merge/dedup them into one
      class ParentBundle extends Bundle with HasTypeAlias {
        override def aliasName = RecordAlias("ParentBundle")

        val foo = new FooBundle(8)
        val bar = new BarBundle(3)
      }

      val io = IO(new Bundle {
        val in = Input(new ParentBundle)
        val out = Output(new ParentBundle)
      })

      io.out :#= io.in
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Test)

    chirrtl should include("type FooBundle = { x : UInt<8>}")
    chirrtl should include("type BarBundle = { y : SInt<3>}")
    chirrtl should include("type ParentBundle = { foo : FooBundle, bar : BarBundle}")
  }

  "Nested bundles with nested FIRRTL type aliases which have flipped values" should "generate properly" in {
    // Similar to the previous test, but also checks for proper hierarchal propagation of directionality to child
    // alias bundles
    class Test extends Module {
      class ChildBundle(width: Int) extends Bundle with HasTypeAlias {
        override def aliasName = RecordAlias("ChildBundle")
        val y = Flipped(SInt(width.W))
      }
      class ParentBundle extends Bundle with HasTypeAlias {
        override def aliasName = RecordAlias("ParentBundle")

        val bar = new ChildBundle(3)
      }

      val io = IO(new Bundle {
        val in = Input(new ParentBundle)
        val out = Output(new ParentBundle)
      })

      io.out :#= io.in
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Test)

    // Check for bad stripped alias: Whenever there is no strip suffix, or the flip is still present
    chirrtl shouldNot include("type ChildBundle = { y : SInt<3>}")
    chirrtl shouldNot include("type ChildBundle = { flip y : SInt<3>}")
    chirrtl shouldNot include("type ChildBundle_stripped = { flip y : SInt<3>}")
    chirrtl should include("type ChildBundle_stripped = { y : SInt<3>}")

    // Stripped parent alias should also be present
    chirrtl should include("type ParentBundle_stripped = { bar : ChildBundle_stripped}")
  }

  "Type alises" should "work on Records" in {
    import scala.collection.immutable.SeqMap
    class Test extends Module {
      class FooRecord extends Record with HasTypeAlias {
        override def aliasName = RecordAlias("FooRecord")

        val elements = SeqMap("foo" -> UInt(8.W), "bar" -> Bool())
      }

      val io = IO(new Bundle {
        val in = Input(new FooRecord)
        val out = Output(new FooRecord)
      })

      io.out :#= io.in
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Test)

    chirrtl should include("type FooRecord = { bar : UInt<1>, foo : UInt<8>}")

    chirrtl should include("flip in : FooRecord")
    chirrtl should include("out : FooRecord")
  }

  "Type alises" should "work on Records with flipped, coerced Elements" in {
    import scala.collection.immutable.SeqMap
    class Test extends Module {
      class FooRecord extends Record with HasTypeAlias {
        override def aliasName = RecordAlias("FooRecord")

        val elements = SeqMap("foo" -> Flipped(UInt(8.W)), "bar" -> Bool())
      }

      val io = IO(new Bundle {
        val in = Input(new FooRecord)
        val out = Output(new FooRecord)
      })

      io.out :#= io.in
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Test)

    chirrtl should include("type FooRecord_stripped = { bar : UInt<1>, foo : UInt<8>}")

    chirrtl should include("flip in : FooRecord_stripped")
    chirrtl should include("out : FooRecord_stripped")
  }

  "Type alises" should "work on Records with flipped, uncoerced Elements" in {
    import scala.collection.immutable.SeqMap
    class Test extends Module {
      class FooRecord extends Record with HasTypeAlias {
        override def aliasName = RecordAlias("FooRecord")

        val elements = SeqMap("foo" -> Flipped(UInt(8.W)), "bar" -> Bool())
      }

      val io = IO(new Bundle {
        val in = Flipped(new FooRecord)
        val out = new FooRecord
      })

      io.out :<>= io.in
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Test)

    chirrtl should include("type FooRecord = { bar : UInt<1>, flip foo : UInt<8>}")

    chirrtl should include("flip in : FooRecord")
    chirrtl should include("out : FooRecord")
  }

  "Type alises" should "work on Records with monodirectional/flipped/uncoerced Elements" in {
    import scala.collection.immutable.SeqMap
    class Test extends Module {
      class FooRecord extends Record with HasTypeAlias {
        override def aliasName = RecordAlias("FooRecord")

        val elements = SeqMap("foo" -> Flipped(UInt(8.W)), "bar" -> Flipped(Bool()))
      }

      val io = IO(new Bundle {
        val in = Flipped(new FooRecord)
        val out = new FooRecord
      })

      io.out :<>= io.in
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Test)

    chirrtl should include("type FooRecord = { flip bar : UInt<1>, flip foo : UInt<8>}")

    chirrtl should include("flip in : FooRecord")
    chirrtl should include("out : FooRecord")
  }
}
