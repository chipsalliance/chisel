// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._
import chisel3.internal.Builder
import chisel3.internal.firrtl.{Formal, Verification}
import chisel3.internal.sourceinfo.SourceInfo

package object verification {

  object assert {
    /** Named class for assertions. */
    final class Assert(private[chisel3] val predicate: Bool) extends BaseSim


    def apply(predicate: Bool, msg: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Assert = {
      val a = new Assert(predicate)
      when (!Module.reset.asBool) {
        val clock = Module.clock
        Builder.pushCommand(Verification(a, Formal.Assert, sourceInfo, clock.ref, predicate.ref, msg))
      }
      a
    }
  }

  object assume {
    /** Named class for assumes. */
    final class Assume(private[chisel3] val predicate: Bool) extends BaseSim

    def apply(predicate: Bool, msg: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Assume = {
      val a = new Assume(predicate)
      when (!Module.reset.asBool) {
        val clock = Module.clock
        Builder.pushCommand(Verification(a, Formal.Assume, sourceInfo, clock.ref, predicate.ref, msg))
      }
      a
    }
  }

  object cover {
    /** Named class for covers. */
    final class Cover(private[chisel3] val predicate: Bool) extends BaseSim

    def apply(predicate: Bool, msg: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Cover = {
      val clock = Module.clock
      val c = new Cover(predicate)
      when (!Module.reset.asBool) {
        Builder.pushCommand(Verification(c, Formal.Cover, sourceInfo, clock.ref, predicate.ref, msg))
      }
      c
    }
  }
}
