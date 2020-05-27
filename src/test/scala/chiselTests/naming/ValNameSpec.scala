// See LICENSE for license details.

package chiselTests.naming

import chisel3._
import chisel3.aop.Select
import chisel3.internal.{ValName, prefix}
import chiselTests.ChiselPropSpec

class ValNameSpec extends ChiselPropSpec {
  property("Nested prefixes should work with plugin") {
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

  property("Valname should work with bit extract") {
    class Test extends MultiIOModule {
      def builder()(implicit valName: ValName) = {
        prefix(valName.name) {
          val wire1 = Wire(UInt(3.W))
          val wire2 = Wire(UInt(3.W))
          wire2
        }
      }
      val x1 = builder()
      val x2 = builder().apply(1)
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (
          List(
            "x1_wire1",
            "x1",
            "x2_wire1",
            "x2_wire2"
          )
        )
    }
  }
}
