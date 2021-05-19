package chisel3.util

import chisel3.Module

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
    def emitVerilog = {
      chisel3.stage.ChiselStage.emitVerilog(module)
    }

    def emitSystemVerilog = {
      chisel3.stage.ChiselStage.emitSystemVerilog(module)
    }

    def emitFirrtl = {
      chisel3.stage.ChiselStage.emitFirrtl(module)
    }

    def emitChirrtl = {
      chisel3.stage.ChiselStage.emitChirrtl(module)
    }
  }
}
