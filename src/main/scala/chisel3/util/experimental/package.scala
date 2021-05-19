package chisel3.util

import chisel3.Module
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import firrtl.AnnotationSeq

package object experimental {

  /** Helper functions to emit module with implicit class
    *
    * @example
    * {{{
    *   import chisel3.util.experimental.ImplicitDriver
    *   object Main extends App {
    *     println((new Hello()).emitVerilog)
    *   }
    * }}}
    */
  implicit class ImplicitDriver(module: => Module) {
    def toVerilogString = ChiselStage.emitVerilog(module)

    def toSystemVerilogString = ChiselStage.emitSystemVerilog(module)

    def toFirrtlString = ChiselStage.emitFirrtl(module)

    def toChirrtlString = ChiselStage.emitChirrtl(module)

    def execute(args: String*)(annos: AnnotationSeq = Nil): AnnotationSeq =
      (new ChiselStage)
        .execute(
          args.toArray,
          annos ++ Seq(new ChiselGeneratorAnnotation(() => module))
        )

    def compile(args: String*): AnnotationSeq = execute(args: _*)(Nil)
  }
}
