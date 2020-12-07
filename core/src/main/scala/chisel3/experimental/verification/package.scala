// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3.{Bool, CompileOptions}
import chisel3.internal.Builder
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.{Formal, Verification}
import chisel3.internal.sourceinfo.SourceInfo


// methodology
object Methodology extends Enumeration {
  val NonInstantiated = Value("nonInstantiated")
  val Combinatorial = Value("combinatorial")
  val MemoryInduction = Value("memoryInduction")
}

package object verification {
  object assert {
    def apply(predicate: Bool, msg: String = "", mtd: Methodology.Value = Methodology.NonInstantiated)(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Builder.forcedClock
      pushCommand(Verification(Formal.Assert, sourceInfo, clock.ref,
        predicate.ref, msg, mtd))
    }
  }

  object assume {
    def apply(predicate: Bool, msg: String = "", mtd: Methodology.Value = Methodology.NonInstantiated)(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Builder.forcedClock
      pushCommand(Verification(Formal.Assume, sourceInfo, clock.ref,
        predicate.ref, msg, mtd))
    }
  }

  object cover {
    def apply(predicate: Bool, msg: String = "", mtd: Methodology.Value = Methodology.NonInstantiated)(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Builder.forcedClock
      pushCommand(Verification(Formal.Cover, sourceInfo, clock.ref,
        predicate.ref, msg, mtd))
    }
  }

  object require {
    def apply(predicate: Bool, msg: String = "", mtd: Methodology.Value = Methodology.NonInstantiated)(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Builder.forcedClock
      //push an assume to the parent
      
      pushCommand(Verification(Formal.Require, sourceInfo, clock.ref,
        predicate.ref, msg, mtd))
    }
  }

  object ensure {
    def apply(predicate: Bool, msg: String = "", mtd: Methodology.Value = Methodology.NonInstantiated)(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Builder.forcedClock
      pushCommand(Verification(Formal.Ensure, sourceInfo, clock.ref,
        predicate.ref, msg, mtd))
    }
  }
}

