// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import chisel3._
import chisel3.experimental._
import chisel3.reflect.DataMirror
import chisel3.testers.{BasicTester, TesterDriver}
import chisel3.util._


class IntModuleTest extends IntModule {
  val foo = IO(new Bundle() {
    val in = Input(Bool())
    val out = Output(Bool())
  })
}

class IntModuleStringParam(str: String) extends IntModule(Map("STRING" -> str)) {
  val io = IO(new Bundle {
    val out = UInt(32.W)
  })
}

class IntModuleRealParam(dbl: Double) extends IntModule(Map("REAL" -> dbl)) {
  val io = IO(new Bundle {
    val out = UInt(64.W)
  })
}

class IntModuleTypeParam(w: Int, raw: String) extends IntModule(Map("T" -> RawParam(raw))) {
  val io = IO(new Bundle {
    val out = UInt(w.W)
  })
}

class IntModuleNoIO extends IntModule {
  // Whoops! typo
  val ioo = IO(new Bundle {
    val out = Output(UInt(8.W))
  })
}
