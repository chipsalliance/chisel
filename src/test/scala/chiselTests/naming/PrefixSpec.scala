// See LICENSE for license details.

package chiselTests.naming

import chisel3._
import chisel3.aop.Select
import chisel3.experimental.{ValName, prefix}
import chiselTests.ChiselPropSpec

class PrefixSpec extends ChiselPropSpec {
  property("Scala plugin should interact with prefixing so last plugin name wins?") {
    class Test extends MultiIOModule {
      def builder() = {
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
      top: Test => Select.wires(top).map(_.instanceName) should be (List("first_wire1", "x1", "second_wire1", "x2"))
    }
  }

  property("Using String as prefix") {
    class Test extends MultiIOModule {
      def builder() = {
        val wire1 = Wire(UInt(3.W))
        val wire2 = Wire(UInt(3.W))
        wire2
      }
      val x1 = prefix("x1") { builder() }
      val x2 = prefix("x2") { builder() }
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("x1_wire1", "x1", "x2_wire1", "x2"))
    }
  }

  property("ValName should enable using assigned val as prefix") {
    class Test extends MultiIOModule {
      def builder()(implicit valName: ValName) = {
        prefix(valName.name) {
          val wire1 = Wire(UInt(3.W))
          val wire2 = Wire(UInt(3.W))
          wire2
        }
      }
      val x1 = builder()
      val x2 = builder()
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("x1_wire1", "x1", "x2_wire1", "x2"))
    }
  }

  property("Nested prefixes should work") {
    class Test extends MultiIOModule {
      def builder2(): UInt = {
        val wire1 = Wire(UInt(3.W))
        val wire2 = Wire(UInt(3.W))
        wire2
      }
      def builder()(implicit valName: ValName) = {
        prefix(valName.name) {
          val wire1 = Wire(UInt(3.W))
          val wire2 = Wire(UInt(3.W))
          prefix("foo") {
            builder2()
          }
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
      def builder() = {
        val wire = Wire(UInt(3.W))
        wire
      }
      val x1 = Wire(UInt(3.W))
      x1 := internal.prefix(x1) {
        builder()
      }
      val x2 = Wire(UInt(3.W))
      x2 := internal.prefix(x2) {
        builder()
      }
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("x1", "x1_wire", "x2", "x2_wire"))
    }
  }
}
