// SPDX-License-Identifier: Apache-2.0

package chiselTests.naming

import chisel3._
import chisel3.aop.Select
import chisel3.experimental.prefix
import chiselTests.{ChiselFunSpec, Utils}
import circt.stage.ChiselStage

import scala.collection.immutable.SeqMap

object StaticObject {
  class MyStaticRecord extends Record { val elements = SeqMap.empty }
  class MyStaticBundle extends Bundle
  class MyStaticModule extends Module
}

class ClassNameSpec extends ChiselFunSpec with Utils {
  object DynamicObject {
    class MyDynamicRecord extends Record { val elements = SeqMap.empty }
    class MyDynamicBundle extends Bundle
    class MyDynamicModule extends Module
  }

  describe("(0) Dynamic declarations") {
    it("(0.a): name of an dynamic class of Module") {
      ChiselStage.elaborate {
        val x = new DynamicObject.MyDynamicModule
        assert(x.chiselClassName == "MyDynamicModule")
        x
      }
    }
    it("(0.b): name of an dynamic class of Record") {
      ChiselStage.elaborate(
        new Module {
          val x = new DynamicObject.MyDynamicRecord()
          assert(x.chiselClassName == "MyDynamicRecord")
        }
      )
    }
    it("(0.c): name of an dynamic class of Bundle") {
      ChiselStage.elaborate(
        new Module {
          val x = new DynamicObject.MyDynamicBundle()
          assert(x.chiselClassName == "MyDynamicBundle")
        }
      )
    }
    it("(0.d): name of an dynamic class of Bundle defined in this it thingy") {
      // I don't know why this adds the $1 at the end, but :shrug: its current behavior
      class MyBundle extends Bundle
      ChiselStage.elaborate(
        new Module {
          import chisel3.experimental.BundleLiterals._
          val x = (new MyBundle())
          assert(x.chiselClassName == "MyBundle$1")
        }
      )
    }
  }

  describe("(1) Static declarations") {
    it("(1.a): name of an dynamic class of Module") {
      ChiselStage.elaborate {
        val x = new StaticObject.MyStaticModule
        assert(x.chiselClassName == "MyStaticModule")
        x
      }
    }
    it("(1.b): name of an dynamic class of Record") {
      ChiselStage.elaborate(
        new Module {
          val x = new StaticObject.MyStaticRecord()
          assert(x.chiselClassName == "MyStaticRecord")
        }
      )
    }
    it("(1.c): name of an dynamic class of Bundle") {
      ChiselStage.elaborate(
        new Module {
          val x = new StaticObject.MyStaticBundle()
          assert(x.chiselClassName == "MyStaticBundle")
        }
      )
    }
  }

  describe("(2) Static declarations") {
    it("(2.a): name of an anonymous class of Module is anonymous") {
      ChiselStage.elaborate {
        val x = new StaticObject.MyStaticModule {}
        assert(x.chiselClassName == "ClassNameSpec_Anon")
        x
      }
    }
    it("(2.b): name of an dynamic class of Record") {
      ChiselStage.elaborate(
        new Module {
          val x = new StaticObject.MyStaticRecord {}
          assert(x.chiselClassName == "")
        }
      )
    }
    it("(2.c): name of an dynamic class of Bundle") {
      ChiselStage.elaborate(
        new Module {
          val x = new StaticObject.MyStaticBundle {}
          assert(x.chiselClassName == "AnonymousBundle")
        }
      )
    }
  }
}
