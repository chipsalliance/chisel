// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage.emitCHIRRTL
import chisel3._
import chisel3.util.RegEnable
import chisel3.experimental.{BaseModule, SourceInfo, SourceLine}
import chisel3.experimental.hierarchy.Definition
import firrtl.ir.FileInfo
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import scala.annotation.nowarn

object SourceLocatorSpec {
  val thisFile = "src/test/scala/chiselTests/SourceLocatorSpec.scala"

  val locator = SourceInfo.materialize

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
  class SimpleDefinitions extends Module {
    val wire = Wire(UInt(8.W))
    val reg = Reg(UInt(8.W))
    val regInit = RegInit(0.U(8.W))
    val regNext = RegNext(0.U(8.W))
    val regEnable = RegEnable(0.U(8.W), true.B)
    val port = IO(Input(Bool()))
    val inst = Module(new RawModuleChild)
    val mem = Mem(1024, UInt(8.W))
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
        chirrtl should include(s"module RawModuleChild : @[$thisFile 20:9]")
      } else {
        chirrtl should include(s"module RawModuleChild :\n") // no source locator yet
      }
    }
    it("(2.b): modules extending Module should have a source locator") {
      val chirrtl = emitCHIRRTL(new ModuleChild)
      if (isScala2) {
        chirrtl should include(s"module ModuleChild : @[$thisFile 21:9]")
      } else {
        chirrtl should include(s"module ModuleChild :\n") // no source locator yet
      }

    }
    it("(2.c): modules extending other user modules should have a source locator") {
      val chirrtl = emitCHIRRTL(new InheritanceModule)
      if (isScala2) {
        chirrtl should include(s"module InheritanceModule : @[$thisFile 22:9]")
      } else {
        chirrtl should include(s"module InheritanceModule :\n") // no source locator yet
      }
    }
    it("(2.d): modules extending BlackBox should have a source locator") {
      val chirrtl = emitCHIRRTL(new WrapperTop(new BlackBoxChild))
      if (isScala2) {
        chirrtl should include(s"extmodule BlackBoxChild : @[$thisFile 24:9]")
      } else {
        chirrtl should include(s"extmodule BlackBoxChild :\n") // no source locator yet
      }
    }
    it("(2.e): modules extending ExtModule should have a source locator") {
      val chirrtl = emitCHIRRTL(new WrapperTop(new ExtModuleChild))
      if (isScala2) {
        chirrtl should include(s"extmodule ExtModuleChild : @[$thisFile 27:9]")
      } else {
        chirrtl should include(s"extmodule ExtModuleChild :\n") // no source locator yet
      }
    }
    it("(2.f): user-defined Classes should have a source locator") {
      val chirrtl = emitCHIRRTL(new ClassTop)
      if (isScala2) {
        chirrtl should include(s"class ClassChild : @[$thisFile 31:9]")
      } else {
        chirrtl should include(s"class ClassChild :\n") // no source locator yet
      }
    }
    it("(2.g): Inner and anonymous modules should have a source locators") {
      val chirrtl = emitCHIRRTL(new Outer)
      if (isScala2) {
        chirrtl should include(s"module Inner : @[$thisFile 36:11]")
        chirrtl should include(s"module AnonymousModule : @[$thisFile 38:25]")
      } else {
        chirrtl should include(s"module Inner :\n") // no source locator yet
        chirrtl should include(s"module AnonymousModule :\n") // no source locator yet
      }
    }
    it("(2.h): Definitions should have a source locator") {
      val chirrtl = emitCHIRRTL(new RawModuleChild)
      if (isScala2) {
        chirrtl should include(s"module RawModuleChild : @[$thisFile 20:9]")
      } else {
        chirrtl should include(s"module RawModuleChild :\n") // no source locator yet
      }
    }
  }

  describe("(3) SourceLocator.makeMessage()") {
    it("(3.a) Should have click-to-source functionality") {
      // This click-to-source works in VSCode terminal, uncomment to manually test
      // println(s"Try clicking to this source locator! ${locator.makeMessage()}")
      if (isScala2) {
        locator.makeMessage() should include(s"$thisFile:18:28")
      } else {
        locator.makeMessage() should include(s"$thisFile:18:16")
      }
    }
  }

  describe("(4) SourceLocator simple definitions") {
    it("(4.a): Simple definitions should have a source locator") {
      val chirrtl = emitCHIRRTL(new SimpleDefinitions)
      if (isScala2) {
        chirrtl should include(s"wire wire : UInt<8> @[$thisFile 46:20]")
        chirrtl should include(s"reg reg : UInt<8>, clock @[$thisFile 47:18]")
        chirrtl should include(s"regreset regInit : UInt<8>, clock, reset, UInt<8>(0h0) @[$thisFile 48:26]")
        chirrtl should include(s"reg regNext : UInt, clock @[$thisFile 49:26]")
        chirrtl should include(s"reg regEnable : UInt<8>, clock @[$thisFile 50:30]")
        chirrtl should include(s"input port : UInt<1> @[$thisFile 51:18]")
        chirrtl should include(s"inst inst of RawModuleChild @[$thisFile 52:22]")
        chirrtl should include(s"cmem mem : UInt<8>[1024] @[$thisFile 53:18]")
      } else {
        chirrtl should include(s"wire wire : UInt<8> @[$thisFile 46:30]")
        chirrtl should include(s"reg reg : UInt<8>, clock @[$thisFile 47:28]")
        chirrtl should include(s"regreset regInit : UInt<8>, clock, reset, UInt<8>(0h0) @[$thisFile 48:35]")
        chirrtl should include(s"reg regNext : UInt, clock @[$thisFile 49:35]")
        chirrtl should include(s"reg regEnable : UInt<8>, clock @[$thisFile 50:47]")
        chirrtl should include(s"input port : UInt<1> @[$thisFile 51:32]")
        chirrtl should include(s"inst inst of RawModuleChild @[core/src/main/scala-3/chisel3/ModuleIntf.scala 18:58]")
        chirrtl should include(s"cmem mem : UInt<8>[1024] @[$thisFile 53:34]")
      }
    }
  }
}
