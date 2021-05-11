// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._
import chisel3.internal.Builder
import chisel3.internal.firrtl.{Formal, Verification}
import chisel3.internal.sourceinfo.SourceInfo

package object verification {
  object assert {
    def apply(predicate: Bool, msg: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      when (!Module.reset.asBool) {
        val clock = Module.clock
        Builder.pushCommand(Verification(Formal.Assert, sourceInfo, clock.ref, predicate.ref, msg))
      }
    }
  }

  object assume {
    def apply(predicate: Bool, msg: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      when (!Module.reset.asBool) {
        val clock = Module.clock
        Builder.pushCommand(Verification(Formal.Assume, sourceInfo, clock.ref, predicate.ref, msg))
      }
    }
  }

  object cover {
    def apply(predicate: Bool, msg: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Module.clock
      when (!Module.reset.asBool) {
        Builder.pushCommand(Verification(Formal.Cover, sourceInfo, clock.ref, predicate.ref, msg))
      }
    }
  }
}
