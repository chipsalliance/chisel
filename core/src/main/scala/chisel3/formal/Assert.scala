
package chisel3.formal


import chisel3.internal.Builder
import chisel3.{Bool, CompileOptions}
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.internal.Builder.pushCommand

object assert {
  def apply(predicate: Bool, msg: String = "")(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit = {
    val clock = Builder.forcedClock
    pushCommand(Assert(sourceInfo, clock.ref, predicate.ref, msg))
  }
}
