// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._
import chisel3.internal.{Builder, NamedComponent}
import chisel3.internal.firrtl.{Formal, Verification}
import chisel3.internal.sourceinfo.SourceInfo

package object verification {

  /** Base simulation-only component. */
  abstract class BaseSim extends NamedComponent

  case class Assert(predicate: Bool, msg: String = "") extends BaseSim
  case class Assume(predicate: Bool, msg: String = "") extends BaseSim
  case class Cover(predicate: Bool, msg: String = "") extends BaseSim

  object assert {
    def apply(predicate: Bool, msg: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Assert = {
      val a = Assert(predicate, msg)
      when (!Module.reset.asBool) {
        val clock = Module.clock
        Builder.pushCommand(Verification(a, Formal.Assert, sourceInfo, clock.ref, predicate.ref, msg))
      }
      a
    }
  }

  object assume {
    def apply(predicate: Bool, msg: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Assume = {
      val a = Assume(predicate, msg)
      when (!Module.reset.asBool) {
        val clock = Module.clock
        Builder.pushCommand(Verification(a, Formal.Assume, sourceInfo, clock.ref, predicate.ref, msg))
      }
      a
    }
  }

  object cover {
    def apply(predicate: Bool, msg: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Cover = {
      val clock = Module.clock
      val c = Cover(predicate, msg)
      when (!Module.reset.asBool) {
        Builder.pushCommand(Verification(c, Formal.Cover, sourceInfo, clock.ref, predicate.ref, msg))
      }
      c
    }
  }
}
