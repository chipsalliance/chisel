// RUN: scala-cli %s

import chisel3._
import circt.stage.ChiselStage
class EmptyModule extends Module
println(ChiselStage.emitCHIRRTL(new EmptyModule))
