// See LICENSE for license details.

package chiselTests.naming

import chisel3. _
import chisel3.aop.Select
import chisel3.experimental.prefix
import chiselTests.ChiselPropSpec

class NamePluginSpec extends ChiselPropSpec {

  property("Scala plugin should name internally scoped components") {
    class Test extends MultiIOModule {
      { val mywire = Wire(UInt(3.W))}
    }
    aspectTest(() => new Test) {
      top: Test => Select.wires(top).head.toTarget.ref should be("mywire")
    }
  }

  property("Scala plugin should name internally scoped instances") {
    class Inner extends MultiIOModule { }
    class Test extends MultiIOModule {
      { val myinstance = Module(new Inner) }
    }
    aspectTest(() => new Test) {
      top: Test => Select.instances(top).head.instanceName should be("myinstance")
    }
  }

  property("Scala plugin interact with prefixing") {
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

  property("Scala plugin should interact with prefixing so last val name wins") {
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
}

