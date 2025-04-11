// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.testers.BasicTester
import chisel3.util.circt.{SRAM, SRAMPort, SRAMPortInfo}
import circt.stage.ChiselStage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

private class SRAMTop extends Module {
  val singlePortSRAM = Module(
    new SRAM(
      8,
      64,
      Seq(
        new SRAMPortInfo(true, true, false)
      )
    )
  )
  val dualPortSRAM = Module(
    new SRAM(
      8,
      64,
      Seq(
        new SRAMPortInfo(false, true, false),
        new SRAMPortInfo(true, false, false)
      )
    )
  )

}

class SRAMSpec extends AnyFlatSpec with Matchers {
  it should "Should work for types" in {
    val fir = ChiselStage.emitCHIRRTL(new SRAMTop)
    print(fir)
    (fir.split('\n').map(_.trim) should contain)
      .allOf(
        "intmodule SRAMw8x64 :",
        "output readwrite : { flip clock : Clock, flip address : UInt<6>, flip chipEnable : UInt<1>, flip writeEnable : UInt<1>, flip writeData : UInt<8>, readData : UInt<8>}",
        "output write : { flip clock : Clock, flip address : UInt<6>, flip chipEnable : UInt<1>, flip writeEnable : UInt<1>, flip writeData : UInt<8>}",
        "output read : { flip clock : Clock, flip address : UInt<6>, flip chipEnable : UInt<1>, readData : UInt<8>}"
      )
  }
}
