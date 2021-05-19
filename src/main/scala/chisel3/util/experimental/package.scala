package chisel3.util

import chisel3.Module
import chisel3.stage.ChiselStage

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
  }
}
