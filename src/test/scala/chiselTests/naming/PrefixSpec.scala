// SPDX-License-Identifier: Apache-2.0

package chiselTests.naming

import chisel3._
import chisel3.aop.Select
import chisel3.experimental.{noPrefix, prefix, skipPrefix, AffectsChiselPrefix}
import chiselTests.{ChiselPropSpec, Utils}
import circt.stage.ChiselStage

class PrefixSpec extends ChiselPropSpec with Utils {
  implicit val minimumMajorVersion: Int = 12
  property("Scala plugin should interact with prefixing so last plugin name wins?") {
    class Test extends Module {
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
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("x1_first_wire1", "x1", "x2_second_wire1", "x2"))
    }
  }

  property("Nested prefixes should work") {
    class Test extends Module {
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
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(
        List(
          "x1_wire1",
          "x1_wire2",
          "x1_foo_wire1",
          "x1",
          "x2_wire1",
          "x2_wire2",
          "x2_foo_wire1",
          "x2"
        )
      )
    }
  }

  property("Skipping a prefix should work") {
    class Test extends Module {
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
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(
        List(
          "x1_wire1",
          "x1",
          "wire1",
          "x2",
          "wire1_1",
          "wire2"
        )
      )
    }
  }

  property("Prefixing seeded with signal") {
    class Test extends Module {
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
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("x1", "x1_wire", "x2", "x2_wire"))
    }
  }

  property("Automatic prefixing should work") {

    class Test extends Module {
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
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("ADAM_a", "ADAM", "JACOB_a", "JACOB"))
    }
  }

  property("No prefixing annotation on defs should work") {

    class Test extends Module {
      def builder(): UInt = noPrefix {
        val a = Wire(UInt(3.W))
        val b = Wire(UInt(3.W))
        b
      }

      { val noprefix = builder() }
    }
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("a", "noprefix"))
    }
  }

  property("Prefixing on temps should work") {

    class Test extends Module {
      def builder(): UInt = {
        val a = Wire(UInt(3.W))
        val b = Wire(UInt(3.W))
        a +& (b * a)
      }

      { val blah = builder() }
    }
    aspectTest(() => new Test) { top: Test =>
      Select.ops(top).map(x => (x._1, x._2.instanceName)) should be(
        List(
          ("mul", "_blah_T"),
          ("add", "blah")
        )
      )
    }
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
    aspectTest(() => new Test) { top: Test =>
      Select.wires(Select.instances(top).head).map(_.instanceName) should be(List("wire"))
    }
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
    aspectTest(() => new Test) { top: Test =>
      Select.wires(Select.instances(top).head).map(_.instanceName) should be(List("wire"))
    }
  }

  property("Instance names should not be added to prefix") {
    class Child(tpe: UInt) extends Module {
      {
        val io = IO(Input(tpe))
      }
    }

    class Test extends Module {
      {
        lazy val module = {
          val x = UInt(3.W)
          new Child(x)
        }
        val child = Module(module)
      }
    }
    aspectTest(() => new Test) { top: Test =>
      Select.ios(Select.instances(top).head).map(_.instanceName) should be(List("clock", "reset", "io"))
    }
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
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("wire"))
    }
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
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("iia_wire"))
    }
  }

  property("Prefixing should NOT be influenced by suggestName") {
    class Test extends Module {
      {
        val wire = {
          val x = Wire(UInt(3.W)) // wire_x
          Wire(UInt(3.W)).suggestName("foo")
        }
      }
    }
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("wire_x", "foo"))
    }
  }

  property("Prefixing should be influenced by the \"current name\" of the signal") {
    class Test extends Module {
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
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("foo", "wire_x", "bar", "wire2_x", "fizz", "fizz_x"))
    }
  }

  property("Prefixing have intuitive behavior") {
    class Test extends Module {
      {
        val wire = {
          val x = Wire(UInt(3.W)).suggestName("mywire")
          val y = Wire(UInt(3.W)).suggestName("mywire2")
          y := x
          y
        }
      }
    }
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("wire_mywire", "mywire2"))
    }
  }

  property("Prefixing on connection to subfields work") {
    class Test extends Module {
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
    aspectTest(() => new Test) { top: Test =>
      Select.registers(top).map(_.instanceName) should be(
        List(
          "wire_x_REG",
          "wire_y_REG",
          "wire_vec_0_REG",
          "wire_vec_REG",
          "wire_vec_1_REG"
        )
      )
    }
  }

  property("Prefixing on connection to IOs should work") {
    class Child extends Module {
      val in = IO(Input(UInt(3.W)))
      val out = IO(Output(UInt(3.W)))
      out := RegNext(in)
    }
    class Test extends Module {
      {
        val child = Module(new Child)
        child.in := RegNext(3.U)
      }
    }
    aspectTest(() => new Test) { top: Test =>
      Select.registers(top).map(_.instanceName) should be(
        List(
          "child_in_REG"
        )
      )
      Select.registers(Select.instances(top).head).map(_.instanceName) should be(
        List(
          "out_REG"
        )
      )
    }
  }

  property("Prefixing on bulk connects should work") {
    class Child extends Module {
      val in = IO(Input(UInt(3.W)))
      val out = IO(Output(UInt(3.W)))
      out := RegNext(in)
    }
    class Test extends Module {
      {
        val child = Module(new Child)
        child.in <> RegNext(3.U)
      }
    }
    aspectTest(() => new Test) { top: Test =>
      Select.registers(top).map(_.instanceName) should be(
        List(
          "child_in_REG"
        )
      )
      Select.registers(Select.instances(top).head).map(_.instanceName) should be(
        List(
          "out_REG"
        )
      )
    }
  }

  property("Connections should use the non-prefixed name of the connected Data") {
    class Test extends Module {
      prefix("foo") {
        val x = Wire(UInt(8.W))
        x := {
          val w = Wire(UInt(8.W))
          w := 3.U
          w + 1.U
        }
      }
    }
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("foo_x", "foo_x_w"))
    }
  }

  property("Connections to aggregate fields should use the non-prefixed aggregate name") {
    class Test extends Module {
      prefix("foo") {
        val x = Wire(new Bundle { val bar = UInt(8.W) })
        x.bar := {
          val w = Wire(new Bundle { val fizz = UInt(8.W) })
          w.fizz := 3.U
          w.fizz + 1.U
        }
      }
    }
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("foo_x", "foo_x_bar_w"))
    }
  }

  property("Prefixing with wires in recursive functions should grow linearly") {
    class Test extends Module {
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
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("x", "x_w_w", "x_w_w_w", "x_w_w_w_w"))
    }
  }

  property("Prefixing should work for verification ops") {
    class Test extends Module {
      val foo, bar = IO(Input(UInt(8.W)))

      {
        val x5 = {
          val x1 = chisel3.assert(1.U === 1.U)
          val x2 = cover(foo =/= bar)
          val x3 = chisel3.assume(foo =/= 123.U)
          val x4 = printf("foo = %d\n", foo)
          x1
        }
      }
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new Test)
    (chirrtl should include).regex("assert.*: x5")
    (chirrtl should include).regex("cover.*: x5_x2")
    (chirrtl should include).regex("assume.*: x5_x3")
    (chirrtl should include).regex("printf.*: x5_x4")
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
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("a_b_c"))
    }
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
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("a__b_c"))
    }
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
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("a_b_c_d"))
    }
  }

  property("Prefixing of AffectsChiselPrefix objects should work") {
    class NotAData extends AffectsChiselPrefix {
      val value = Wire(UInt(3.W))
    }
    class NotADataUnprefixed {
      val value = Wire(UInt(3.W))
    }
    class Test extends Module {
      {
        val nonData = new NotAData
        // Instance name of nonData.value should be nonData_value
        nonData.value := RegNext(3.U)

        val nonData2 = new NotADataUnprefixed
        // Instance name of nonData2.value should be value
        nonData2.value := RegNext(3.U)
      }
    }
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("nonData_value", "value"))
    }
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
    aspectTest(() => new Test) { top: Test =>
      Select.wires(top).map(_.instanceName) should be(List("prefixed_wire"))
    }
  }
}
