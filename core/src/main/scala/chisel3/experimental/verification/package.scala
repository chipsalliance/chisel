// SPDX-License-Identifier: Apache-2.0

package chisel3
package experimental

import chisel3.{ Bool, CompileOptions }
import chisel3.internal.Builder
import chisel3.internal.Builder.{ pushCommand, pushOp }
import chisel3.internal.firrtl.{ ILit, PrimOp, DefPrim, Formal, Verification, Methodology }
import chisel3.internal.sourceinfo.SourceInfo

package object verification {
  implicit class Delay[T <: Data](data: T) {
    def in(delay: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
      val clock = Builder.forcedClock
      scala.Predef.require(delay > 0, "A signal can only be mesured in a positive amount of clock cycles")
      requireIsHardware(data, "")
      pushOp(DefPrim(sourceInfo, data.cloneTypeFull, PrimOp.InOp, data.ref, ILit(delay)))
    }
  }
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
