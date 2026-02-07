// SPDX-License-Identifier: Apache-2.0

package chiselTests.naming

import chisel3._
import chisel3.experimental.prefix
import chisel3.experimental.AffectsChiselName
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NamePluginSpec extends AnyFlatSpec with Matchers with FileCheck {
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
    ChiselStage.emitCHIRRTL {
      new Module {
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
    }.fileCheck()(
      """|CHECK: wire first_wire :
         |CHECK: wire second_wire :
         |""".stripMargin
    )
  }

  "Scala plugin" should "interact with prefixing so last val name wins" in {
    ChiselStage.emitCHIRRTL {
      new Module {
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
    }.fileCheck()(
      """|CHECK: wire x1_first_wire1
         |CHECK: wire x1
         |CHECK: wire x2_second_wire1
         |CHECK: wire x2
         |""".stripMargin
    )
  }

  "Scala plugin" should "name verification ops" in {
    ChiselStage.emitCHIRRTL {
      new Module {
        val foo, bar = IO(Input(UInt(8.W)))

        {
          val x2 = cover(foo =/= bar)
          val x3 = chisel3.assume(foo =/= 123.U)
          val x4 = printf("foo = %d\n", foo)
        }
      }
    }.fileCheck()(
      """|CHECK: cover{{.*}}: x2
         |CHECK: assume{{.*}}: x3
         |CHECK: printf{{.*}}: x4
         |""".stripMargin
    )
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

    ChiselStage.emitCHIRRTL {
      new Module {
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
    }.fileCheck()(
      """|CHECK: wire blah_0 :
         |CHECK: wire blah_1 :
         |""".stripMargin
    )
  }

  "Naming on nested iterables" should "work" in {

    ChiselStage.emitCHIRRTL {
      new Module {
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
    }.fileCheck()(
      """|CHECK: wire blah_0_0 :
         |CHECK: wire blah_0_1 :
         |CHECK: wire blah_1_0 :
         |CHECK: wire blah_1_1 :
         |""".stripMargin
    )
  }

  "Naming on custom case classes" should "not work" in {
    ChiselStage.emitCHIRRTL {
      case class Container(a: UInt, b: UInt)

      new Module {
        def builder(): Container = {
          val a = Wire(UInt(3.W))
          val b = Wire(UInt(3.W))
          Container(a, b)
        }

        { val blah = builder() }
      }
    }.fileCheck()(
      """|CHECK: wire a :
         |CHECK: wire b :
         |""".stripMargin
    )
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
    ChiselStage.emitCHIRRTL {
      new Module {
        {
          val (a, b) = (Wire(UInt(3.W)), Wire(UInt(3.W)))
        }
      }
    }.fileCheck()(
      """|CHECK: wire a :
         |CHECK: wire b :
         |""".stripMargin
    )
  }

  "Unapply assignments" should "name (but not prefix) local vals on the RHS" in {
    ChiselStage.emitCHIRRTL {
      new Module {
        {
          val (a, b) = {
            val x, y = Wire(UInt(3.W))
            val sum = WireInit(x + y)
            (x, y)
          }
        }
      }
    }.fileCheck()(
      """|CHECK: wire a :
         |CHECK: wire b :
         |CHECK: wire sum :
         |"""
    )
  }

  "Unapply assignments" should "not override already named things" in {
    ChiselStage.emitCHIRRTL {
      new Module {
        {
          val x = Wire(UInt(3.W))
          val (a, b) = (x, Wire(UInt(3.W)))
        }
      }
    }.fileCheck()(
      """|CHECK: wire x :
         |CHECK: wire b :
         |""".stripMargin
    )
  }

  "Case class unapply assignments" should "be named" in {
    ChiselStage.emitCHIRRTL {
      case class Foo(x: UInt, y: UInt)
      new Module {
        {
          def func() = Foo(Wire(UInt(3.W)), Wire(UInt(3.W)))
          val Foo(a, b) = func()
        }
      }
    }.fileCheck()(
      """|CHECK: wire a :
         |CHECK: wire b :
         |""".stripMargin
    )
  }

  "Complex unapply assignments" should "be named" in {
    ChiselStage.emitCHIRRTL {
      case class Foo(x: UInt, y: UInt)
      new Module {
        {
          val w = Wire(UInt(3.W))
          def func() = {
            val x = Foo(Wire(UInt(3.W)), Wire(UInt(3.W)))
            (x, w) :: Nil
          }
          val ((Foo(a, _), c) :: Nil) = func()
        }
      }
    }.fileCheck()(
      """|CHECK: wire w :
         |CHECK: wire a :
         |CHECK: wire _WIRE :
         |""".stripMargin
    )
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
    ChiselStage.emitCHIRRTL {
      new Module {
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
    }.fileCheck()(
      """|CHECK: wire a_b_c :
         |CHECK: wire a_b :
         |CHECK: wire a :
         |"""
    )
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
    ChiselStage.emitCHIRRTL {
      new Module {
        val x = (Wire(UInt(3.W)), Wire(UInt(3.W)))
      }
    }.fileCheck()(
      """|CHECK: wire x_1 :
         |CHECK: wire x_2 :
         |""".stripMargin
    )
  }

  "nested tuples" should "be named" in {
    ChiselStage.emitCHIRRTL {
      new Module {
        val x = (
          (Wire(UInt(3.W)), Wire(UInt(3.W))),
          (Wire(UInt(3.W)), Wire(UInt(3.W)))
        )
      }
    }.fileCheck()(
      """|CHECK: wire x_1_1 :
         |CHECK: wire x_1_2 :
         |CHECK: wire x_2_1 :
         |CHECK: wire x_2_2 :
         |""".stripMargin
    )
  }

  "tuples containing non-Data" should "be named" in {
    ChiselStage.emitCHIRRTL {
      new Module {
        val x = (Wire(UInt(3.W)), "foobar", Wire(UInt(3.W)))
      }
    }.fileCheck()(
      """|CHECK: wire x_1 :
         |CHECK: wire x_3 :
         |""".stripMargin
    )
  }

  "tuples nested in options" should "be named" in {
    ChiselStage.emitCHIRRTL {
      new Module {
        val x = Option((Wire(UInt(3.W)), Wire(UInt(3.W))))
      }
    }.fileCheck()(
      """|CHECK: wire x_1 :
         |CHECK: wire x_2 :
         |""".stripMargin
    )
  }

  "tuple assignment" should "name IOs and registers" in {
    ChiselStage
      .emitCHIRRTL(
        new Module {
          def myFunc(): (UInt, String) = {
            val out = IO(Output(UInt(3.W)))
            val in = IO(Input(UInt(3.W)))
            out := Mux(in(0), RegNext(in + 2.U), in << 3)
            (out, "Hi!")
          }

          val foo = myFunc()
        }
      )
      .fileCheck("--implicit-check-not=wire")(
        """|CHECK: input clock :
           |CHECK: input reset :
           |CHECK: output foo_1 :
           |CHECK: input foo_in :
           |CHECK: reg foo_out_REG :
           |""".stripMargin
      )
  }

  "identity views" should "forward names to their targets" in {
    ChiselStage.emitCHIRRTL {
      import chisel3.experimental.dataview._
      new Module {
        val x = {
          val _w = Wire(UInt(3.W))
          _w.viewAs[UInt]
        }
        val y = Wire(UInt(3.W)).readOnly // readOnly is implemented with views

        val z = Wire(UInt(3.W))
        val zz = z.viewAs[UInt] // But don't accidentally override names
      }
    }.fileCheck()(
      """|CHECK: wire x :
         |CHECK: wire y :
         |CHECK: wire z :
         |""".stripMargin
    )
  }

  "FlatIOs" should "forward names to their targets" in {
    ChiselStage.emitCHIRRTL {
      new Module {
        val io = FlatIO(new Bundle {
          val in = Input(UInt(3.W))
          val out = Output(UInt(3.W))
        })
        io.out := {
          val w = Wire(UInt(3.W))
          w := io.in + 1.U
          w
        }
      }
    }.fileCheck()(
      """|CHECK: input in :
         |CHECK: output out :
         |CHECK: wire out_w :
         |""".stripMargin
    )
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

  "AffectsChiselName" should "work in unapply" in {
    case class SomeClass(d: UInt) extends AffectsChiselName
    class Test extends Module {
      val (x, y) = {
        val bad = SomeClass(Wire(UInt(8.W)))
        (3, bad)
      }
    }
    ChiselStage
      .emitCHIRRTL(new Test)
      .fileCheck()(
        """|CHECK-NOT: bad
           |CHECK:     wire y_d :
           |""".stripMargin
      )
  }

  "withNames" should "require the same number of names as fields" in {
    case class Foo(x: Int, y: Int, z: Int)
    an[IllegalArgumentException] should be thrownBy {
      chisel3.withNames("a", "b")(Foo(1, 2, 3))
    }
  }
}
