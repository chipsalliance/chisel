// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._
import chisel3.internal.Builder
import chisel3.internal.firrtl.{Formal, Verification}
import chisel3.internal.sourceinfo.SourceInfo

package object verification {

  /** Named class for assertions. */
  final class Assert(predicate: Bool, msg: String = "") extends BaseSim

  /** Named class for assumes. */
  final class Assume(predicate: Bool, msg: String = "") extends BaseSim

  /** Named class for covers. */
  final class Cover(predicate: Bool, msg: String = "") extends BaseSim

  object assert {
    def apply(predicate: Bool, msg: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Assert = {
      val a = new Assert(predicate, msg)
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
      val a = new Assume(predicate, msg)
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
      val c = new Cover(predicate, msg)
      when (!Module.reset.asBool) {
        Builder.pushCommand(Verification(c, Formal.Cover, sourceInfo, clock.ref, predicate.ref, msg))
      }
      c
    }
  }
}
