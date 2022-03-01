package chisel3

import chisel3.stage.ChiselStage
import firrtl.AnnotationSeq

object getVerilogString {

  /**
    * Returns a string containing the Verilog for the module specified by
    * the target.
    *
    * @param gen the module to be converted to Verilog
    * @return a string containing the Verilog for the module specified by
    *         the target
    */
  def apply(gen: => RawModule): String = ChiselStage.emitVerilog(gen)

  /**
    * Returns a string containing the Verilog for the module specified by
    * the target accepting arguments and annotations
    *
    * @param gen the module to be converted to Verilog
    * @param args arguments to be passed to the compiler
    * @param annotations annotations to be passed to the compiler
    * @return a string containing the Verilog for the module specified by
    *         the target
    */
  def apply(gen: => RawModule, args: Array[String] = Array.empty, annotations: AnnotationSeq = Seq.empty): String =
    (new ChiselStage).emitVerilog(gen, args, annotations)
}

object emitVerilog {
  def apply(gen: => RawModule, args: Array[String] = Array.empty, annotations: AnnotationSeq = Seq.empty): Unit = {
    (new ChiselStage).emitVerilog(gen, args, annotations)
  }
}
