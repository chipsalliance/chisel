// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage.emitCHIRRTL
import chisel3._
import chisel3.experimental.{BaseModule, ExtModule, SourceInfo, SourceLine}
import chisel3.experimental.hierarchy.Definition
import firrtl.ir.FileInfo
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

object SourceLocatorSpec {
  val thisFile = "src/test/scala-2/chiselTests/SourceLocatorSpec.scala"

  class RawModuleChild extends RawModule
  class ModuleChild extends Module
  class InheritanceModule extends ModuleChild
  class BlackBoxChild extends BlackBox {
    val io = IO(new Bundle {})
  }
  class ExtModuleChild extends ExtModule
  class WrapperTop[T <: BaseModule](gen: => T) extends RawModule {
    val child = Module(gen)
  }
  class ClassChild extends properties.Class
  class ClassTop extends RawModule {
    Definition(new ClassChild)
  }
  class Outer extends RawModule {
    class Inner extends RawModule
    val c = Module(new Inner)
    val c2 = Module(new RawModule {
      override def desiredName = "AnonymousModule"
    })
  }
  class DefinitionWrapper extends RawModule {
    Definition(new RawModuleChild)
  }
}

class SourceLocatorSpec extends AnyFunSpec with Matchers {
  import SourceLocatorSpec._

  describe("(0) Relative source paths") {
    it("(0.a): are emitted by default relative to `user-dir`") {
      class Top extends Module {
        val w = WireInit(UInt(1.W), 0.U)
      }
      val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("@[src/test/scala-2/chiselTests/SourceLocatorSpec.scala")
    }
  }

  describe("(1) Source locators with special characters") {
    val filename = "I need escaping\n\\\t].scala"
    val escaped = "I need escaping\\n\\\\\\t\\].scala"
    val info = SourceLine(filename, 123, 456)
    it("(1.a): are properly escaped when converting to FIRRTL") {
      val firrtl = FileInfo.fromUnescaped(filename)
      firrtl should equal(FileInfo(escaped))
    }
    it("(1.b): are properly escaped to FIRRTL through Chisel elaboration") {
      implicit val sl = info

      val chirrtl = emitCHIRRTL(new RawModule {
        val in = IO(Input(UInt(8.W)))
        val out = IO(Output(UInt(8.W)))
        out := in
      })
      chirrtl should include(escaped)
    }
    it("(1.c): can be properly unescaped") {
      val escapedInfo = FileInfo(escaped)
      escapedInfo.unescaped should equal(filename)
    }
  }

  describe("(2) Module source locators") {
    it("(2.a): modules extending RawModule should have a source locator") {
      val chirrtl = emitCHIRRTL(new RawModuleChild)
      chirrtl should include(s"module RawModuleChild : @[$thisFile 16:9]")
    }
    it("(2.b): modules extending Module should have a source locator") {
      val chirrtl = emitCHIRRTL(new ModuleChild)
      chirrtl should include(s"module ModuleChild : @[$thisFile 17:9]")
    }
    it("(2.c): modules extending other user modules should have a source locator") {
      val chirrtl = emitCHIRRTL(new InheritanceModule)
      chirrtl should include(s"module InheritanceModule : @[$thisFile 18:9]")
    }
    it("(2.d): modules extending BlackBox should have a source locator") {
      val chirrtl = emitCHIRRTL(new WrapperTop(new BlackBoxChild))
      chirrtl should include(s"extmodule BlackBoxChild : @[$thisFile 19:9]")
    }
    it("(2.e): modules extending ExtModule should have a source locator") {
      val chirrtl = emitCHIRRTL(new WrapperTop(new ExtModuleChild))
      chirrtl should include(s"extmodule ExtModuleChild : @[$thisFile 22:9]")
    }
    it("(2.f): user-defined Classes should have a source locator") {
      val chirrtl = emitCHIRRTL(new ClassTop)
      chirrtl should include(s"class ClassChild : @[$thisFile 26:9]")
    }
    it("(2.g): Inner and anonymous modules should have a source locators") {
      val chirrtl = emitCHIRRTL(new Outer)
      chirrtl should include(s"module Inner : @[$thisFile 31:11]")
      chirrtl should include(s"module AnonymousModule : @[$thisFile 33:25]")
    }
    it("(2.h): Definitions should have a source locator") {
      val chirrtl = emitCHIRRTL(new RawModuleChild)
      chirrtl should include(s"module RawModuleChild : @[$thisFile 16:9]")
    }
  }

  describe("(3) SourceLocator.makeMessage()") {
    it("(3.a) Should have click-to-source functionality") {
      val locator = SourceInfo.materialize
      // This click-to-source works in VSCode terminal, uncomment to manually test
      // println(s"Try clicking to this source locator! ${locator.makeMessage()}")
      locator.makeMessage() should include(s"$thisFile:117:32")
    }
  }
}
