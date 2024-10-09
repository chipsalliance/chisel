// SPDX-License-Identifier: Apache-2.0

package chiselTests.naming

import chisel3._
import chisel3.aop.Select
import chisel3.experimental.prefix
import chisel3.experimental.AffectsChiselName
import chiselTests.{ChiselFlatSpec, MatchesAndOmits, Utils}
import circt.stage.ChiselStage

class NamePluginSpec extends ChiselFlatSpec with MatchesAndOmits with Utils {
  implicit val minimumScalaVersion: Int = 12

  "Scala plugin" should "name internally scoped components" in {
    class Test extends Module {
      { val mywire = Wire(UInt(3.W)) }
    }
    ChiselStage.emitCHIRRTL(new Test) should include("wire mywire :")
  }

  "Scala plugin" should "name internally scoped instances" in {
    class Inner extends Module {}
    class Test extends Module {
      { val myinstance = Module(new Inner) }
    }
    ChiselStage.emitCHIRRTL(new Test) should include("inst myinstance of Inner")
  }

  "Scala plugin" should "interact with prefixing" in {
    class Test extends Module {
      def builder() = {
        val wire = Wire(UInt(3.W))
      }
      prefix("first") {
        builder()
      }
      prefix("second") {
        builder()
      }
    }
    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))("wire first_wire :", "wire second_wire :")()
  }

  "Scala plugin" should "interact with prefixing so last val name wins" in {
    class Test extends Module {
      def builder() = {
        val wire1 = Wire(UInt(3.W))
        val wire2 = Wire(UInt(3.W))
        wire2
      }
      {
        val x1 = prefix("first") {
          builder()
        }
      }
      {
        val x2 = prefix("second") {
          builder()
        }
      }
    }
    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))(
      "wire x1_first_wire1",
      "wire x1",
      "wire x2_second_wire1",
      "wire x2"
    )()
  }

  "Scala plugin" should "name verification ops" in {
    class Test extends Module {
      val foo, bar = IO(Input(UInt(8.W)))

      {
        val x2 = cover(foo =/= bar)
        val x3 = chisel3.assume(foo =/= 123.U)
        val x4 = printf("foo = %d\n", foo)
      }
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new Test)
    (chirrtl should include).regex("cover.*: x2")
    (chirrtl should include).regex("assume.*: x3")
    (chirrtl should include).regex("printf.*: x4")
  }

  "Naming on option" should "work" in {

    class Test extends Module {
      def builder(): Option[UInt] = {
        val a = Wire(UInt(3.W))
        Some(a)
      }

      { val blah = builder() }
    }
    ChiselStage.emitCHIRRTL(new Test) should include("wire blah :")
  }

  "Naming on iterables" should "work" in {

    class Test extends Module {
      def builder(): Seq[UInt] = {
        val a = Wire(UInt(3.W))
        val b = Wire(UInt(3.W))
        Seq(a, b)
      }
      {
        val blah = {
          builder()
        }
      }
    }
    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))("wire blah_0 :", "wire blah_1 :")()
  }

  "Naming on nested iterables" should "work" in {

    class Test extends Module {
      def builder(): Seq[Seq[UInt]] = {
        val a = Wire(UInt(3.W))
        val b = Wire(UInt(3.W))
        val c = Wire(UInt(3.W))
        val d = Wire(UInt(3.W))
        Seq(Seq(a, b), Seq(c, d))
      }
      {
        val blah = {
          builder()
        }
      }
    }
    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))(
      "wire blah_0_0 :",
      "wire blah_0_1 :",
      "wire blah_1_0 :",
      "wire blah_1_1 :"
    )()
  }

  "Naming on custom case classes" should "not work" in {
    case class Container(a: UInt, b: UInt)

    class Test extends Module {
      def builder(): Container = {
        val a = Wire(UInt(3.W))
        val b = Wire(UInt(3.W))
        Container(a, b)
      }

      { val blah = builder() }
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))("wire a :", "wire b :")()
  }

  "Multiple names on an IO within a module" should "get the first name" in {
    class Test extends RawModule {
      {
        val a = IO(Output(UInt(3.W)))
        val b = a
      }
    }

    ChiselStage.emitCHIRRTL(new Test) should include("output a :")
  }

  "Multiple names on a non-IO" should "get the first name" in {
    class Test extends Module {
      {
        val a = Wire(UInt(3.W))
        val b = a
      }
    }

    ChiselStage.emitCHIRRTL(new Test) should include("wire a :")
  }

  "Outer Expression, First Statement naming" should "apply to IO" in {
    class Test extends RawModule {
      {
        val widthOpt: Option[Int] = Some(4)
        val out = widthOpt.map { w =>
          val port = IO(Output(UInt(w.W)))
          port
        }
        val foo = out
        val bar = out.get
      }
    }

    ChiselStage.emitCHIRRTL(new Test) should include("output out :")
  }

  "Outer Expression, First Statement naming" should "apply to non-IO" in {
    class Test extends RawModule {
      {
        val widthOpt: Option[Int] = Some(4)
        val fizz = widthOpt.map { w =>
          val wire = Wire(UInt(w.W))
          wire
        }
        val foo = fizz
        val bar = fizz.get
      }
    }

    ChiselStage.emitCHIRRTL(new Test) should include("wire fizz :")
  }

  "autoSeed" should "NOT override automatic naming for IO" in {
    class Test extends RawModule {
      {
        val a = IO(Output(UInt(3.W)))
        a.autoSeed("b")
      }
    }

    ChiselStage.emitCHIRRTL(new Test) should include("output a :")
  }

  "autoSeed" should "override automatic naming for non-IO" in {
    class Test extends Module {
      {
        val a = Wire(UInt(3.W))
        a.autoSeed("b")
      }
    }

    ChiselStage.emitCHIRRTL(new Test) should include("wire b :")
  }

  "Unapply assignments" should "still be named" in {
    class Test extends Module {
      {
        val (a, b) = (Wire(UInt(3.W)), Wire(UInt(3.W)))
      }
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))("wire a :", "wire b :")()
  }

  "Unapply assignments" should "name (but not prefix) local vals on the RHS" in {
    class Test extends Module {
      {
        val (a, b) = {
          val x, y = Wire(UInt(3.W))
          val sum = WireInit(x + y)
          (x, y)
        }
      }
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))("wire a :", "wire b :", "wire sum :")()
  }

  "Unapply assignments" should "not override already named things" in {
    class Test extends Module {
      {
        val x = Wire(UInt(3.W))
        val (a, b) = (x, Wire(UInt(3.W)))
      }
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))("wire x :", "wire b :")()
  }

  "Case class unapply assignments" should "be named" in {
    case class Foo(x: UInt, y: UInt)
    class Test extends Module {
      {
        def func() = Foo(Wire(UInt(3.W)), Wire(UInt(3.W)))
        val Foo(a, b) = func()
      }
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))("wire a :", "wire b :")()
  }

  "Complex unapply assignments" should "be named" in {
    case class Foo(x: UInt, y: UInt)
    class Test extends Module {
      {
        val w = Wire(UInt(3.W))
        def func() = {
          val x = Foo(Wire(UInt(3.W)), Wire(UInt(3.W)))
          (x, w) :: Nil
        }
        val ((Foo(a, _), c) :: Nil) = func()
      }
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))("wire w :", "wire a :", "wire _WIRE :")()
  }

  "Recursive types" should "not infinitely loop" in {
    // When this fails, it causes a StackOverflow when compiling the tests
    // Unfortunately, this doesn't seem to work with assertCompiles(...), it probably ignores the
    // custom project scalacOptions
    def func(x: String) = {
      // We only check types of vals, we don't actually want to run this code though
      val y = scala.xml.XML.loadFile(x)
      y
    }
  }

  "Nested val declarations" should "all be named" in {
    class Test extends Module {
      {
        val a = {
          val b = {
            val c = Wire(UInt(3.W))
            Wire(UInt(3.W))
          }
          Wire(UInt(3.W))
        }
      }
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))("wire a_b_c :", "wire a_b :", "wire a :")()
  }

  behavior.of("Unnamed values (aka \"Temporaries\")")

  they should "be declared by starting the name with '_'" in {
    class Test extends Module {
      {
        val a = {
          val b = {
            val _c = Wire(UInt(3.W))
            4.U // literal so there is no name
          }
          b
        }
      }
    }
    ChiselStage.emitCHIRRTL(new Test) should include("wire _a_b_c :")
  }

  "tuples" should "be named" in {
    class Test extends Module {
      val x = (Wire(UInt(3.W)), Wire(UInt(3.W)))
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))("wire x_1 :", "wire x_2 :")()
  }

  "nested tuples" should "be named" in {
    class Test extends Module {
      val x = (
        (Wire(UInt(3.W)), Wire(UInt(3.W))),
        (Wire(UInt(3.W)), Wire(UInt(3.W)))
      )
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))("wire x_1_1 :", "wire x_1_2 :", "wire x_2_1 :", "wire x_2_2 :")()
  }

  "tuples containing non-Data" should "be named" in {
    class Test extends Module {
      val x = (Wire(UInt(3.W)), "foobar", Wire(UInt(3.W)))
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))("wire x_1 :", "wire x_3 :")()
  }

  "tuples nested in options" should "be named" in {
    class Test extends Module {
      val x = Option((Wire(UInt(3.W)), Wire(UInt(3.W))))
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))("wire x_1 :", "wire x_2 :")()
  }

  "tuple assignment" should "name IOs and registers" in {
    class Test extends Module {
      def myFunc(): (UInt, String) = {
        val out = IO(Output(UInt(3.W)))
        val in = IO(Input(UInt(3.W)))
        out := Mux(in(0), RegNext(in + 2.U), in << 3)
        (out, "Hi!")
      }

      val foo = myFunc()
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))(
      "input clock :",
      "input reset :",
      "output foo_1 :",
      "input foo_in :",
      "reg foo_out_REG :"
    )("wire")
  }

  "identity views" should "forward names to their targets" in {
    import chisel3.experimental.dataview._
    class Test extends Module {
      val x = {
        val _w = Wire(UInt(3.W))
        _w.viewAs[UInt]
      }
      val y = Wire(UInt(3.W)).readOnly // readOnly is implemented with views

      val z = Wire(UInt(3.W))
      val zz = z.viewAs[UInt] // But don't accidentally override names
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Test))(
      "wire x :",
      "wire y :",
      "wire z :"
    )()
  }

  "AffectsChiselName" should "name the user-defined type" in {
    case class SomeClass(d: UInt) extends AffectsChiselName
    class Test extends Module {
      val x = SomeClass(Wire(UInt(8.W)))
    }
    ChiselStage.emitCHIRRTL(new Test) should include("wire x_d :")
  }

  "AffectsChiselName with a user-defined Product" should "give an empty name" in {
    case class SomeClass(d: UInt) extends AffectsChiselName {
      override def productElementName(n: Int): String = ""
    }
    class Test extends Module {
      val x = SomeClass(Wire(UInt(8.W)))
    }
    ChiselStage.emitCHIRRTL(new Test) should include("wire x :")
  }
}
