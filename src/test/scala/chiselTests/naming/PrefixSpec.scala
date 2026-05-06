// SPDX-License-Identifier: Apache-2.0

package chiselTests.naming

import chisel3._
import chisel3.experimental.{noPrefix, prefix, skipPrefix, AffectsChiselPrefix}
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.matchers.should.Matchers
import org.scalatest.propspec.AnyPropSpec

class PrefixSpec extends AnyPropSpec with Matchers with FileCheck {
  implicit val minimumMajorVersion: Int = 12
  property("Scala plugin should interact with prefixing so last plugin name wins?") {
    ChiselStage.emitCHIRRTL {
      new Module {
        def builder(): UInt = {
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
      """|CHECK: wire x1_first_wire1 :
         |CHECK: wire x2_second_wire1 :
         |CHECK: wire x2 :
         |""".stripMargin
    )
  }

  property("Nested prefixes should work") {
    ChiselStage.emitCHIRRTL {
      new Module {
        def builder2(): UInt = {
          val wire1 = Wire(UInt(3.W))
          val wire2 = Wire(UInt(3.W))
          wire2
        }
        def builder(): UInt = {
          val wire1 = Wire(UInt(3.W))
          val wire2 = Wire(UInt(3.W))
          prefix("foo") {
            builder2()
          }
        }
        { val x1 = builder() }
        { val x2 = builder() }
      }
    }.fileCheck()(
      """|CHECK: wire x1_wire1 :
         |CHECK: wire x1_wire2 :
         |CHECK: wire x1_foo_wire1 :
         |CHECK: wire x1 :
         |CHECK: wire x2_wire1 :
         |CHECK: wire x2_wire2 :
         |CHECK: wire x2_foo_wire1 :
         |CHECK: wire x2 :""".stripMargin
    )
  }

  property("Skipping a prefix should work") {
    ChiselStage.emitCHIRRTL {
      new Module {
        def builder2(): UInt = {
          skipPrefix {
            val wire1 = Wire(UInt(3.W))
            val wire2 = Wire(UInt(3.W))
            wire2
          }
        }
        def builder(): UInt = {
          prefix("foo") {
            builder2()
          }
        }
        { val x1 = builder() }
        { val x2 = builder2() }
        { builder2() }
      }
    }.fileCheck()(
      """|CHECK: wire x1_wire1 :
         |CHECK: wire x1 :
         |CHECK: wire wire1 :
         |CHECK: wire x2 :
         |CHECK: wire wire1_1 :
         |CHECK: wire wire2 :
         |""".stripMargin
    )
  }

  property("Prefixing seeded with signal") {
    ChiselStage.emitCHIRRTL {
      new Module {
        def builder(): UInt = {
          val wire = Wire(UInt(3.W))
          wire := 3.U
          wire
        }
        {
          val x1 = Wire(UInt(3.W))
          x1 := {
            builder()
          }
          val x2 = Wire(UInt(3.W))
          x2 := {
            builder()
          }
        }
      }
    }.fileCheck()(
      """|CHECK: wire x1 :
         |CHECK: wire x1_wire :
         |CHECK: wire x2 :
         |CHECK: wire x2_wire :
         |""".stripMargin
    )
  }

  property("Automatic prefixing should work") {

    ChiselStage.emitCHIRRTL {
      new Module {
        def builder(): UInt = {
          val a = Wire(UInt(3.W))
          val b = Wire(UInt(3.W))
          b
        }

        {
          val ADAM = builder()
          val JACOB = builder()
        }
      }
    }.fileCheck()(
      """|CHECK: wire ADAM_a :
         |CHECK: wire ADAM :
         |CHECK: wire JACOB_a :
         |CHECK: wire JACOB :
         |""".stripMargin
    )
  }

  property("No prefixing annotation on defs should work") {

    ChiselStage.emitCHIRRTL {
      new Module {
        def builder(): UInt = noPrefix {
          val a = Wire(UInt(3.W))
          val b = Wire(UInt(3.W))
          b
        }

        { val noprefix = builder() }
      }
    }.fileCheck()(
      """|CHECK: wire a :
         |CHECK: wire noprefix :
         |""".stripMargin
    )
  }

  property("Prefixing on temps should work") {

    ChiselStage.emitCHIRRTL {
      new Module {
        def builder(): UInt = {
          val a = Wire(UInt(3.W))
          val b = Wire(UInt(3.W))
          a +& (b * a)
        }

        { val blah = builder() }
      }
    }.fileCheck()(
      """|CHECK: node _blah_T = mul
         |CHECK: node blah = add
         |""".stripMargin
    )
  }

  property("Prefixing should not leak into child modules") {
    class Child extends Module {
      {
        val wire = Wire(UInt())
      }
    }

    class Test extends Module {
      {
        val child = prefix("InTest") {
          Module(new Child)
        }
      }
    }
    ChiselStage.emitCHIRRTL(new Test) should include("wire wire :")
  }

  property("Prefixing should not leak into child modules, example 2") {
    class Child extends Module {
      {
        val wire = Wire(UInt())
      }
    }

    class Test extends Module {
      val x = IO(Input(UInt(3.W)))
      val y = {
        lazy val module = new Child
        val child = Module(module)
      }
    }
    ChiselStage.emitCHIRRTL(new Test) should include("wire wire")
  }

  property("Instance names should not be added to prefix") {
    ChiselStage.emitCHIRRTL {
      class Child(tpe: UInt) extends Module {
        {
          val io = IO(Input(tpe))
        }
      }

      new Module {
        {
          lazy val module = {
            val x = UInt(3.W)
            new Child(x)
          }
          val child = Module(module)
        }
      }
    }.fileCheck()(
      """|CHECK: input clock :
         |CHECK: input reset :
         |CHECK: input io :
         |""".stripMargin
    )
  }

  property("Prefixing should not be caused by nested Iterable[Iterable[Any]]") {
    class Test extends Module {
      {
        val iia = {
          val wire = Wire(UInt(3.W))
          List(List("Blah"))
        }
      }
    }
    ChiselStage.emitCHIRRTL(new Test) should include("wire wire :")
  }

  property("Prefixing should be caused by nested Iterable[Iterable[Data]]") {
    class Test extends Module {
      {
        val iia = {
          val wire = Wire(UInt(3.W))
          List(List(3.U))
        }
      }
    }
    ChiselStage.emitCHIRRTL(new Test) should include("wire iia_wire :")
  }

  property("Prefixing should NOT be influenced by suggestName") {
    ChiselStage.emitCHIRRTL {
      new Module {
        {
          val wire = {
            val x = Wire(UInt(3.W)) // wire_x
            Wire(UInt(3.W)).suggestName("foo")
          }
        }
      }
    }.fileCheck()(
      """|CHECK: wire wire_x :
         |CHECK: wire foo :
         |""".stripMargin
    )
  }

  property("Prefixing should be influenced by the \"current name\" of the signal") {
    ChiselStage.emitCHIRRTL {
      new Module {
        {
          val wire = {
            val y = Wire(UInt(3.W)).suggestName("foo")
            val x = Wire(UInt(3.W)) // wire_x
            y
          }

          val wire2 = Wire(UInt(3.W))
          wire2 := {
            val x = Wire(UInt(3.W)) // wire2_x
            x + 1.U
          }
          wire2.suggestName("bar")

          val wire3 = Wire(UInt(3.W))
          wire3.suggestName("fizz")
          wire3 := {
            val x = Wire(UInt(3.W)) // fizz_x
            x + 1.U
          }
        }
      }
    }.fileCheck()(
      """|CHECK: wire foo :
         |CHECK: wire wire_x :
         |CHECK: wire bar :
         |CHECK: wire wire2_x :
         |CHECK: wire fizz :
         |CHECK: wire fizz_x :
         |""".stripMargin
    )
  }

  property("Prefixing have intuitive behavior") {
    ChiselStage.emitCHIRRTL {
      new Module {
        {
          val wire = {
            val x = Wire(UInt(3.W)).suggestName("mywire")
            val y = Wire(UInt(3.W)).suggestName("mywire2")
            y := x
            y
          }
        }
      }
    }.fileCheck()(
      """|CHECK: wire wire_mywire :
         |CHECK: wire mywire2 :
         |""".stripMargin
    )
  }

  property("Prefixing on connection to subfields work") {
    ChiselStage.emitCHIRRTL {
      new Module {
        {
          val wire = Wire(new Bundle {
            val x = UInt(3.W)
            val y = UInt(3.W)
            val vec = Vec(4, UInt(3.W))
          })
          wire.x := RegNext(3.U)
          wire.y := RegNext(3.U)
          wire.vec(0) := RegNext(3.U)
          wire.vec(wire.x) := RegNext(3.U)
          wire.vec(1.U) := RegNext(3.U)
        }
      }
    }.fileCheck()(
      """|CHECK: reg wire_x_REG :
         |CHECK: reg wire_y_REG :
         |CHECK: reg wire_vec_0_REG :
         |CHECK: reg wire_vec_REG :
         |CHECK: reg wire_vec_1_REG :
         |""".stripMargin
    )
  }

  property("Prefixing on connection to IOs should work") {
    ChiselStage.emitCHIRRTL {
      class Child extends Module {
        val in = IO(Input(UInt(3.W)))
        val out = IO(Output(UInt(3.W)))
        out := RegNext(in)
      }
      new Module {
        {
          val child = Module(new Child)
          child.in := RegNext(3.U)
        }
      }
    }.fileCheck()(
      """|CHECK: reg out_REG :
         |CHECK: reg child_in_REG :
         |""".stripMargin
    )
  }

  property("Prefixing on bulk connects should work") {
    ChiselStage.emitCHIRRTL {
      class Child extends Module {
        val in = IO(Input(UInt(3.W)))
        val out = IO(Output(UInt(3.W)))
        out := RegNext(in)
      }
      new Module {
        {
          val child = Module(new Child)
          child.in <> RegNext(3.U)
        }
      }
    }.fileCheck()(
      """|CHECK: reg out_REG :
         |CHECK: reg child_in_REG :
         |""".stripMargin
    )

  }

  property("Connections should use the non-prefixed name of the connected Data") {
    ChiselStage.emitCHIRRTL {
      new Module {
        prefix("foo") {
          val x = Wire(UInt(8.W))
          x := {
            val w = Wire(UInt(8.W))
            w := 3.U
            w + 1.U
          }
        }
      }
    }.fileCheck()(
      """|CHECK: wire foo_x :
         |CHECK: wire foo_x_w :
         |""".stripMargin
    )
  }

  property("Connections to aggregate fields should use the non-prefixed aggregate name") {
    ChiselStage.emitCHIRRTL {
      new Module {
        prefix("foo") {
          val x = Wire(new Bundle { val bar = UInt(8.W) })
          x.bar := {
            val w = Wire(new Bundle { val fizz = UInt(8.W) })
            w.fizz := 3.U
            w.fizz + 1.U
          }
        }
      }
    }.fileCheck()(
      """|CHECK: wire foo_x :
         |CHECK: wire foo_x_bar_w :
         |""".stripMargin
    )
  }

  property("Prefixing with wires in recursive functions should grow linearly") {
    ChiselStage.emitCHIRRTL {
      new Module {
        def func(bools: Seq[Bool]): Bool = {
          if (bools.isEmpty) true.B
          else {
            val w = Wire(Bool())
            w := bools.head && func(bools.tail)
            w
          }
        }
        val in = IO(Input(Vec(4, Bool())))
        val x = func(in)
      }
    }.fileCheck()(
      """|CHECK: wire x :
         |CHECK: wire x_w_w :
         |CHECK: wire x_w_w_w :
         |CHECK: wire x_w_w_w_w :
         |""".stripMargin
    )
  }

  property("Prefixing should work for verification ops") {
    ChiselStage.emitCHIRRTL {
      new Module {
        val foo, bar = IO(Input(UInt(8.W)))

        {
          val x5 = {
            val x2 = cover(foo =/= bar)
            val x3 = chisel3.assume(foo =/= 123.U)
            val x4 = printf("foo = %d\n", foo)
            x2
          }
        }
      }
    }.fileCheck()(
      """|CHECK: cover{{.*}}: x5
         |CHECK: assume{{.*}}: x5_x3
         |CHECK: printf{{.*}}: x5_x4
         |""".stripMargin
    )
  }

  property("Leading '_' in val names should be ignored in prefixes") {
    class Test extends Module {
      {
        val a = {
          val _b = {
            val c = Wire(UInt(3.W))
            4.U // literal because there is no name
          }
          _b
        }
      }
    }
    ChiselStage.emitCHIRRTL(new Test) should include("wire a_b_c :")
  }

  // This checks that we don't just blanket ignore leading _ in prefixes
  property("User-specified prefixes with '_' should be respected") {
    class Test extends Module {
      {
        val a = {
          val _b = prefix("_b") {
            val c = Wire(UInt(3.W))
          }
          4.U
        }
      }
    }
    ChiselStage.emitCHIRRTL(new Test) should include("wire a__b_c :")
  }

  property("Leading '_' in signal names should be ignored in prefixes from connections") {
    class Test extends Module {
      {
        val a = {
          val b = {
            val _c = IO(Output(UInt(3.W))) // port so not selected as wire
            _c := {
              val d = Wire(UInt(3.W))
              d
            }
            4.U // literal so there is no name
          }
          b
        }
      }
    }
    ChiselStage.emitCHIRRTL(new Test) should include("wire a_b_c_d :")
  }

  property("Prefixing of AffectsChiselPrefix objects should work") {
    ChiselStage.emitCHIRRTL {
      class NotAData extends AffectsChiselPrefix {
        val value = Wire(UInt(3.W))
      }
      class NotADataUnprefixed {
        val value = Wire(UInt(3.W))
      }
      new Module {
        {
          val nonData = new NotAData
          // Instance name of nonData.value should be nonData_value
          nonData.value := RegNext(3.U)

          val nonData2 = new NotADataUnprefixed
          // Instance name of nonData2.value should be value
          nonData2.value := RegNext(3.U)
        }
      }
    }.fileCheck()(
      """|CHECK: wire nonData_value :
         |CHECK: wire value :
         |""".stripMargin
    )
  }

  property("Prefixing should not be affected by repeated calls of suggestName") {
    class Test extends Module {
      val in = IO(Input(UInt(3.W)))
      val prefixed = {
        val wire = Wire(UInt(3.W)).suggestName("wire") // "prefixed_wire"
        wire := in

        val thisShouldNotBeHere = {
          // Second suggestName doesn't modify the instanceName since it was
          // already suggested, but also should not modify the prefix either

          // Incorrect behavior would rename the wire to
          // "prefixed_thisShouldNotBeHere_wire"
          wire.suggestName("wire")

          val out = IO(Output(UInt(3.W)))
          out := wire
          out
        }
        thisShouldNotBeHere
      }
    }
    ChiselStage.emitCHIRRTL(new Test) should include("wire prefixed_wire :")
  }

  property("Empty prefixes should be a no-op") {
    class MyModule extends Module {
      prefix("") {
        val w = Wire(UInt(3.W))
      }
      val x = prefix("") {
        val y = Wire(UInt(3.W))
        val z = Wire(UInt(3.W))
        y
      }
    }
    println(ChiselStage.emitCHIRRTL(new MyModule))
    ChiselStage
      .emitCHIRRTL(new MyModule)
      .fileCheck()(
        """|CHECK: wire w :
           |CHECK: wire x :
           |CHECK: wire x_z :
           |""".stripMargin
      )
  }

  property("skipPrefix should apply to empty prefixes") {
    class MyModule extends Module {
      // skipPrefix should remove the empty prefix, not x_
      val x = prefix("") {
        skipPrefix {
          val y = Wire(UInt(3.W))
          val z = Wire(UInt(3.W))
          y
        }
      }
    }
    ChiselStage
      .emitCHIRRTL(new MyModule)
      .fileCheck()(
        """|CHECK: wire x :
           |CHECK: wire x_z :
           |""".stripMargin
      )
  }
}
