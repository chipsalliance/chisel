// See LICENSE for license details.

package chiselTests.naming

import chisel3._
import chisel3.aop.Select
import chisel3.experimental.{dump, noPrefix, prefix, treedump}
import chiselTests.ChiselPropSpec

class PrefixSpec extends ChiselPropSpec {
  property("Scala plugin should interact with prefixing so last plugin name wins?") {
    class Test extends MultiIOModule {
      def builder(): UInt = {
        val wire1 = Wire(UInt(3.W))
        val wire2 = Wire(UInt(3.W))
        wire2
      }

      val x1 = prefix("first") {
        builder()
      }
      val x2 = prefix("second") {
        builder()
      }
    }
    aspectTest(() => new Test) {
      top: Test => Select.wires(top).map(_.instanceName) should be (List("x1_first_wire1", "x1", "x2_second_wire1", "x2"))
    }
  }

  property("Nested prefixes should work") {
    class Test extends MultiIOModule {
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
      val x1 = builder()
      val x2 = builder()
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (
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

  property("Prefixing seeded with signal") {
    class Test extends MultiIOModule {
      @treedump
      @dump
      def builder(): UInt = {
        val wire = Wire(UInt(3.W))
        wire := 3.U
        wire
      }
      val x1 = Wire(UInt(3.W))
      x1 := {
        builder()
      }
      val x2 = Wire(UInt(3.W))
      x2 := {
        builder()
      }
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("x1", "x1_wire", "x2", "x2_wire"))
    }
  }

  property("Automatic prefixing should work") {

    class Test extends MultiIOModule {
      def builder(): UInt = {
        val a = Wire(UInt(3.W))
        val b = Wire(UInt(3.W))
        b
      }

      val ADAM = builder()
      val JACOB = builder()
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("ADAM_a", "ADAM", "JACOB_a", "JACOB"))
    }
  }

  property("No prefixing annotation on defs should work") {

    class Test extends MultiIOModule {
      def builder(): UInt = noPrefix {
        val a = Wire(UInt(3.W))
        val b = Wire(UInt(3.W))
        b
      }

      val noprefix = builder()
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("a", "noprefix"))
    }
  }

  property("Prefixing on temps should work") {

    class Test extends MultiIOModule {
      def builder(): UInt = {
        val a = Wire(UInt(3.W))
        val b = Wire(UInt(3.W))
        a +& (b * a)
      }

      val blah = builder()
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.ops(top).map(x => (x._1, x._2.instanceName)) should be (List(
          ("mul", "_blah_T"),
          ("add", "blah")
        ))
    }
  }

  property("Prefixing should not leak into child modules") {
    class Child extends MultiIOModule {
      val wire = Wire(UInt())
    }

    class Test extends MultiIOModule {
      val child = prefix("InTest") {
        Module(new Child)
      }
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top.child).map(_.instanceName) should be (List("wire"))
    }
  }
}
