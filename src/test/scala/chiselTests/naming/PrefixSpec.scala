// See LICENSE for license details.

package chiselTests.naming

import chisel3._
import chisel3.aop.Select
import chisel3.experimental.{dump, noPrefix, prefix, treedump}
import chiselTests.{ChiselPropSpec, Utils}

class PrefixSpec extends ChiselPropSpec with Utils {
  implicit val minimumMajorVersion: Int = 12
  property("Scala plugin should interact with prefixing so last plugin name wins?") {
    class Test extends MultiIOModule {
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
      { val x1 = builder() }
      { val x2 = builder() }
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

      {
        val ADAM = builder()
        val JACOB = builder()
      }
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

      { val noprefix = builder() }
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

      { val blah = builder() }
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
      {
        val wire = Wire(UInt())
      }
    }

    class Test extends MultiIOModule {
      {
        val child = prefix("InTest") {
          Module(new Child)
        }
      }
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(Select.instances(top).head).map(_.instanceName) should be (List("wire"))
    }
  }

  property("Prefixing should not leak into child modules, example 2") {
    class Child extends MultiIOModule {
      {
        val wire = Wire(UInt())
      }
    }

    class Test extends MultiIOModule {
      val x = IO(Input(UInt(3.W)))
      val y = {
        lazy val module = new Child
        val child = Module(module)
      }
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(Select.instances(top).head).map(_.instanceName) should be (List("wire"))
    }
  }

  property("Instance names should not be added to prefix") {
    class Child(tpe: UInt) extends MultiIOModule {
      {
        val io = IO(Input(tpe))
      }
    }

    class Test extends MultiIOModule {
      {
        lazy val module = {
          val x = UInt(3.W)
          new Child(x)
        }
        val child = Module(module)
      }
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.ios(Select.instances(top).head).map(_.instanceName) should be (List("clock", "reset", "io"))
    }
  }


  property("Prefixing should not be caused by nested Iterable[Iterable[Any]]") {
    class Test extends MultiIOModule {
      {
        val iia = {
          val wire = Wire(UInt(3.W))
          List(List("Blah"))
        }
      }
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("wire"))
    }
  }

  property("Prefixing should be caused by nested Iterable[Iterable[Data]]") {
    class Test extends MultiIOModule {
      {
        val iia = {
          val wire = Wire(UInt(3.W))
          List(List(3.U))
        }
      }
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("iia_wire"))
    }
  }

  property("Prefixing should be the prefix during the last call to autoName/suggestName") {
    class Test extends MultiIOModule {
      {
        val wire = {
          val x = Wire(UInt(3.W)).suggestName("mywire")
          x
        }
      }
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("mywire"))
        Select.wires(top).map(_.instanceName) shouldNot be (List("wire_mywire"))
    }
  }

  property("Prefixing have intuitive behavior") {
    class Test extends MultiIOModule {
      {
        val wire = {
          val x = Wire(UInt(3.W)).suggestName("mywire")
          val y = Wire(UInt(3.W)).suggestName("mywire2")
          y := x
          y
        }
      }
    }
    aspectTest(() => new Test) {
      top: Test => Select.wires(top).map(_.instanceName) should be (List("wire_mywire", "mywire2"))
    }
  }

  property("Prefixing on connection to subfields work") {
    class Test extends MultiIOModule {
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
      }
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.registers(top).map(_.instanceName) should be (List(
          "wire_x_REG",
          "wire_y_REG",
          "wire_vec_0_REG",
          "wire_vec_REG"
        ))
    }
  }

  property("Prefixing on connection to IOs should work") {
    class Child extends MultiIOModule {
      val in = IO(Input(UInt(3.W)))
      val out = IO(Output(UInt(3.W)))
      out := RegNext(in)
    }
    class Test extends MultiIOModule {
      {
        val child = Module(new Child)
        child.in := RegNext(3.U)
      }
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.registers(top).map(_.instanceName) should be (List(
          "child_in_REG"
        ))
        Select.registers(Select.instances(top).head).map(_.instanceName) should be (List(
          "out_REG"
        ))
    }
  }

  property("Prefixing on bulk connects should work") {
    class Child extends MultiIOModule {
      val in = IO(Input(UInt(3.W)))
      val out = IO(Output(UInt(3.W)))
      out := RegNext(in)
    }
    class Test extends MultiIOModule {
      {
        val child = Module(new Child)
        child.in <> RegNext(3.U)
      }
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.registers(top).map(_.instanceName) should be (List(
          "child_in_REG"
        ))
        Select.registers(Select.instances(top).head).map(_.instanceName) should be (List(
          "out_REG"
        ))
    }
  }
}
