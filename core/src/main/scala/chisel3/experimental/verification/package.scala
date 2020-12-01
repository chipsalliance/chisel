// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3.{Bool, CompileOptions}
import chisel3.internal.Builder
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.{Formal, Methodology, Verification}
import chisel3.internal.sourceinfo.SourceInfo

package object verification {
  object assert {
    def apply(predicate: Bool, msg: String = "", mtd: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Builder.forcedClock
      val method = mtd match {
        case "" => Methodology.Trivial
        case "memoryInduction" => Methodology.MemoryInduction
      }
      pushCommand(Verification(Formal.Assert, sourceInfo, clock.ref,
        predicate.ref, msg, method))
    }
  }

  object assume {
    def apply(predicate: Bool, msg: String = "", mtd: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Builder.forcedClock
      val method = mtd match {
        case "" => Methodology.Trivial
        case "MemoryInduction" => Methodology.MemoryInduction
      }
      pushCommand(Verification(Formal.Assume, sourceInfo, clock.ref,
        predicate.ref, msg, method))
    }
  }

  object cover {
    def apply(predicate: Bool, msg: String = "", mtd: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Builder.forcedClock
      val method = mtd match {
        case "" => Methodology.Trivial
        case "MemoryInduction" => Methodology.MemoryInduction
      }
      pushCommand(Verification(Formal.Cover, sourceInfo, clock.ref,
        predicate.ref, msg, method))
    }
  }

  object require {
    def apply(predicate: Bool, msg: String = "", mtd: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Builder.forcedClock
      val method = mtd match {
        case "" => Methodology.Trivial
        case "MemoryInduction" => Methodology.MemoryInduction
      }
      //push an assume to the parent
      
      pushCommand(Verification(Formal.Require, sourceInfo, clock.ref,
        predicate.ref, msg, method))
    }
  }

  object ensure {
    def apply(predicate: Bool, msg: String = "", mtd: String = "")(
      implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
      val clock = Builder.forcedClock
      val method = mtd match {
        case "" => Methodology.Trivial
        case "MemoryInduction" => Methodology.MemoryInduction
      }
      pushCommand(Verification(Formal.Ensure, sourceInfo, clock.ref,
        predicate.ref, msg, method))
    }
  }
}
