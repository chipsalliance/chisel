// SPDX-License-Identifier: Apache-2.0

package chiselTests.naming

import chisel3._
import chisel3.aop.Select
import chisel3.experimental.{prefix, treedump}
import chiselTests.{ChiselFlatSpec, Utils}

class NamePluginSpec extends ChiselFlatSpec with Utils {
  implicit val minimumScalaVersion: Int = 12

  "Scala plugin" should "name internally scoped components" in {
    class Test extends MultiIOModule {
      { val mywire = Wire(UInt(3.W))}
    }
    aspectTest(() => new Test) {
      top: Test => Select.wires(top).head.toTarget.ref should be("mywire")
    }
  }

  "Scala plugin" should "name internally scoped instances" in {
    class Inner extends MultiIOModule { }
    class Test extends MultiIOModule {
      { val myinstance = Module(new Inner) }
    }
    aspectTest(() => new Test) {
      top: Test => Select.instances(top).head.instanceName should be("myinstance")
    }
  }

  "Scala plugin" should "interact with prefixing" in {
    class Test extends MultiIOModule {
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
    aspectTest(() => new Test) {
      top: Test => Select.wires(top).map(_.instanceName) should be (List("first_wire", "second_wire"))
    }
  }

  "Scala plugin" should "interact with prefixing so last val name wins" in {
    class Test extends MultiIOModule {
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
    aspectTest(() => new Test) {
      top: Test => Select.wires(top).map(_.instanceName) should be (List("x1_first_wire1", "x1", "x2_second_wire1", "x2"))
    }
  }

  "Naming on option" should "work" in {

    class Test extends MultiIOModule {
      def builder(): Option[UInt] = {
        val a = Wire(UInt(3.W))
        Some(a)
      }

      { val blah = builder() }
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("blah"))
    }
  }


  "Naming on iterables" should "work" in {

    class Test extends MultiIOModule {
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
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("blah_0", "blah_1"))
    }
  }

  "Naming on nested iterables" should "work" in {

    class Test extends MultiIOModule {
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
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (
          List(
            "blah_0_0",
            "blah_0_1",
            "blah_1_0",
            "blah_1_1"
          ))
    }
  }

  "Naming on custom case classes" should "not work" in {
    case class Container(a: UInt, b: UInt)

    class Test extends MultiIOModule {
      def builder(): Container = {
        val a = Wire(UInt(3.W))
        val b = Wire(UInt(3.W))
        Container(a, b)
      }

      { val blah = builder() }
    }
    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("a", "b"))
    }
  }

  "Multiple names on an IO within a module" should "get the first name" in {
    class Test extends RawModule {
      {
        val a = IO(Output(UInt(3.W)))
        val b = a
      }
    }

    aspectTest(() => new Test) {
      top: Test =>
        Select.ios(top).map(_.instanceName) should be (List("a"))
    }
  }

  "Multiple names on a non-IO" should "get the first name" in {
    class Test extends MultiIOModule {
      {
        val a = Wire(UInt(3.W))
        val b = a
      }
    }

    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("a"))
    }
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

    aspectTest(() => new Test) {
      top: Test =>
        Select.ios(top).map(_.instanceName) should be (List("out"))
    }
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

    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("fizz"))
    }
  }

  "autoSeed" should "NOT override automatic naming for IO" in {
    class Test extends RawModule {
      {
        val a = IO(Output(UInt(3.W)))
        a.autoSeed("b")
      }
    }

    aspectTest(() => new Test) {
      top: Test =>
        Select.ios(top).map(_.instanceName) should be (List("a"))
    }
  }


  "autoSeed" should "override automatic naming for non-IO" in {
    class Test extends MultiIOModule {
      {
        val a = Wire(UInt(3.W))
        a.autoSeed("b")
      }
    }

    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("b"))
    }
  }

  "Unapply assignments" should "still be named" in {
    class Test extends MultiIOModule {
      {
        val (a, b) = (Wire(UInt(3.W)), Wire(UInt(3.W)))
      }
    }

    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("a", "b"))
    }
  }

  "Unapply assignments" should "not override already named things" in {
    class Test extends MultiIOModule {
      {
        val x = Wire(UInt(3.W))
        val (a, b) = (x, Wire(UInt(3.W)))
      }
    }

    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("x", "b"))
    }
  }

  "Case class unapply assignments" should "be named" in {
    case class Foo(x: UInt, y: UInt)
    class Test extends MultiIOModule {
      {
        def func() = Foo(Wire(UInt(3.W)), Wire(UInt(3.W)))
        val Foo(a, b) = func()
      }
    }

    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("a", "b"))
    }
  }

  "Complex unapply assignments" should "be named" in {
    case class Foo(x: UInt, y: UInt)
    class Test extends MultiIOModule {
      {
        val w = Wire(UInt(3.W))
        def func() = {
          val x = Foo(Wire(UInt(3.W)), Wire(UInt(3.W)))
          (x, w) :: Nil
        }
        val ((Foo(a, _), c) :: Nil) = func()
      }
    }

    aspectTest(() => new Test) {
      top: Test =>
        Select.wires(top).map(_.instanceName) should be (List("w", "a", "_WIRE"))
    }
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
    class Test extends MultiIOModule {
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

    aspectTest(() => new Test) {
      top: Test => Select.wires(top).map(_.instanceName) should be (List("a_b_c", "a_b", "a"))
    }
  }
}

