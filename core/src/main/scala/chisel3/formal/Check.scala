package chisel3.formal


import chisel3.{Bool, CompileOptions}
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.internal.Builder.pushCommand

object check {
  def apply(expr: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit = {
    pushCommand(Check(sourceInfo, expr.ref))
  }
}
