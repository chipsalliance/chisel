// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage.emitCHIRRTL
import chisel3._
import chisel3.experimental.{BaseModule, SourceInfo, SourceLine}
import chisel3.experimental.hierarchy.Definition
import firrtl.ir.FileInfo
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import scala.annotation.nowarn

object SourceLocatorSpec {
  val thisFile = "src/test/scala/chiselTests/SourceLocatorSpec.scala"

  class RawModuleChild extends RawModule
  class ModuleChild extends Module
  class InheritanceModule extends ModuleChild
  @nowarn("cat=deprecation")
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

  def isScala2 = chisel3.BuildInfo.scalaVersion.startsWith("2.")

  describe("(0) Relative source paths") {
    it("(0.a): are emitted by default relative to `user-dir`") {
      class Top extends Module {
        val w = WireInit(UInt(1.W), 0.U)
      }
      val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include(s"@[$thisFile")
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
      if (isScala2) {
        chirrtl should include(s"module RawModuleChild : @[$thisFile 17:9]")
      } else {
        chirrtl should include(s"module RawModuleChild :\n") // no source locator yet
      }
    }
    it("(2.b): modules extending Module should have a source locator") {
      val chirrtl = emitCHIRRTL(new ModuleChild)
      if (isScala2) {
        chirrtl should include(s"module ModuleChild : @[$thisFile 18:9]")
      } else {
        chirrtl should include(s"module ModuleChild :\n") // no source locator yet
      }

    }
    it("(2.c): modules extending other user modules should have a source locator") {
      val chirrtl = emitCHIRRTL(new InheritanceModule)
      if (isScala2) {
        chirrtl should include(s"module InheritanceModule : @[$thisFile 19:9]")
      } else {
        chirrtl should include(s"module InheritanceModule :\n") // no source locator yet
      }
    }
    it("(2.d): modules extending BlackBox should have a source locator") {
      val chirrtl = emitCHIRRTL(new WrapperTop(new BlackBoxChild))
      if (isScala2) {
        chirrtl should include(s"extmodule BlackBoxChild : @[$thisFile 21:9]")
      } else {
        chirrtl should include(s"extmodule BlackBoxChild :\n") // no source locator yet
      }
    }
    it("(2.e): modules extending ExtModule should have a source locator") {
      val chirrtl = emitCHIRRTL(new WrapperTop(new ExtModuleChild))
      if (isScala2) {
        chirrtl should include(s"extmodule ExtModuleChild : @[$thisFile 24:9]")
      } else {
        chirrtl should include(s"extmodule ExtModuleChild :\n") // no source locator yet
      }
    }
    it("(2.f): user-defined Classes should have a source locator") {
      val chirrtl = emitCHIRRTL(new ClassTop)
      if (isScala2) {
        chirrtl should include(s"class ClassChild : @[$thisFile 28:9]")
      } else {
        chirrtl should include(s"class ClassChild :\n") // no source locator yet
      }
    }
    it("(2.g): Inner and anonymous modules should have a source locators") {
      val chirrtl = emitCHIRRTL(new Outer)
      if (isScala2) {
        chirrtl should include(s"module Inner : @[$thisFile 33:11]")
        chirrtl should include(s"module AnonymousModule : @[$thisFile 35:25]")
      } else {
        chirrtl should include(s"module Inner :\n") // no source locator yet
        chirrtl should include(s"module AnonymousModule :\n") // no source locator yet
      }
    }
    it("(2.h): Definitions should have a source locator") {
      val chirrtl = emitCHIRRTL(new RawModuleChild)
      if (isScala2) {
        chirrtl should include(s"module RawModuleChild : @[$thisFile 17:9]")
      } else {
        chirrtl should include(s"module RawModuleChild :\n") // no source locator yet
      }
    }
  }

  describe("(3) SourceLocator.makeMessage()") {
    it("(3.a) Should have click-to-source functionality") {
      val locator = SourceInfo.materialize
      // This click-to-source works in VSCode terminal, uncomment to manually test
      // println(s"Try clicking to this source locator! ${locator.makeMessage()}")
      if (isScala2) {
        locator.makeMessage() should include(s"$thisFile:155:32")
      } else {
        locator.makeMessage() should include(s"$thisFile:155:20")
      }
    }
  }
}
