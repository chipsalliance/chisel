// See LICENSE for license details.

package chisel3.experimental

import chisel3.{Bool, CompileOptions}
import chisel3.internal.Builder
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.{Formal, Verification}
import chisel3.internal.sourceinfo.SourceInfo

package object verification {
  object assert {
    def apply(predicate: Bool, msg: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Builder.forcedClock
      pushCommand(Verification(Formal.Assert, sourceInfo, clock.ref,
        predicate.ref, msg))
    }
  }

  object assume {
    def apply(predicate: Bool, msg: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Builder.forcedClock
      pushCommand(Verification(Formal.Assume, sourceInfo, clock.ref,
        predicate.ref, msg))
    }
  }

  object cover {
    def apply(predicate: Bool, msg: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Builder.forcedClock
      pushCommand(Verification(Formal.Cover, sourceInfo, clock.ref,
        predicate.ref, msg))
    }
  }
}
