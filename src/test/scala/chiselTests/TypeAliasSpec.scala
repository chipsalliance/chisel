package chiselTests

import chisel3._
import circt.stage.ChiselStage

class TypeAliasSpec extends ChiselFlatSpec with Utils {
  "Bundles with opt-in alias names" should "have an emitted FIRRTL type alias" in {
    class Test extends Module {
      class FooBundle extends Bundle {
        override def aliasName = Some(typeName)

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
      class BarBundle extends Bundle {
        override def aliasName = Some("Buzz")

        val y = UInt(8.W)
      }
      class FooBundle extends Bundle {
        override def aliasName = Some("Fizz")

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

  "Duplicate bundle type aliases with same structures" should "compile" in {
    class Test extends Module {
      // All three of these bundles are structurally equivalent in FIRRTL and thus
      // are equivalent, substitutable aliases for each other. Merge/dedup them into one
      class FooBundle extends Bundle {
        override def aliasName = Some("IdenticalBundle")

        val x = UInt(8.W)
        val y = UInt(8.W)
      }
      class FooBarBundle extends Bundle {
        override def aliasName = Some("IdenticalBundle")

        val x = UInt(8.W)
        val y = UInt(8.W)
      }
      class BarBundle extends Bundle {
        override def aliasName = Some("IdenticalBundle")

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
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      class Test extends Module {
        // These bundles are structurally unequivalent and so must be aliased with different names.
        // Error if they share the same name
        class FooBundle extends Bundle {
          override def aliasName = Some("DifferentBundle")

          val x = Bool()
          val y = UInt(8.W)
        }
        class BarBundle extends Bundle {
          override def aliasName = Some("DifferentBundle")

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
    }).getMessage should include(
      "Attempted to redeclare an existing type alias 'DifferentBundle' with a new bundle structure"
    )
  }

  "Bundles with unsanitary names" should "properly sanitize" in {
    class Test extends Module {
      class FooBundle extends Bundle {
        // Sanitizes to '_'; the sanitized alias needs to be used in the aliasing algorithm
        // instead of the direct user-defined alias.
        override def aliasName = Some("")

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
          class FooBundle extends Bundle {
            override def aliasName = Some(firrtlType)

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
        s"Attempted to override a FIRRTL keyword '$tpe' with a type alias!"
      )
    }
  }
}
