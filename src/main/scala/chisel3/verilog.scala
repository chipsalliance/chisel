package chisel3

import chisel3.stage.ChiselStage
import firrtl.AnnotationSeq

object getVerilogString {
  def apply(gen: => RawModule): String = ChiselStage.emitVerilog(gen)
}

object emitVerilog {
  def apply(gen: => RawModule, args: Array[String] = Array.empty, annotations: AnnotationSeq = Seq.empty): Unit = {
    (new ChiselStage).emitVerilog(gen, args, annotations)
  }
}
