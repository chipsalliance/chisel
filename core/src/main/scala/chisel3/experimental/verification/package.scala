// SPDX-License-Identifier: Apache-2.0

package chisel3
package experimental

import chisel3.{Bool, CompileOptions}
import chisel3.internal.Builder
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.{Formal, Verification}
import chisel3.internal.sourceinfo.SourceInfo

package object verification {
  implicit class Delay[T <: Data](data: T) {
    def in(delay : Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
      
      val regs = for (i <- 0 until delay)
        yield Reg(data.cloneTypeFull)
      
      regs.foldRight(data) { (current, last) =>
        current := last
        current
      }
    }
  }
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

  object require {
    def apply(predicate: Bool, msg: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Builder.forcedClock
      //push an assume to the parent
      
      pushCommand(Verification(Formal.Require, sourceInfo, clock.ref,
        predicate.ref, msg))
    }
  }

  object ensure {
    def apply(predicate: Bool, msg: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Builder.forcedClock
      pushCommand(Verification(Formal.Ensure, sourceInfo, clock.ref,
        predicate.ref, msg))
    }
  }
}
