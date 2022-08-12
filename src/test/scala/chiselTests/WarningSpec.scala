// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.util._
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.experimental.ChiselEnum
import chiselTests.ChiselFlatSpec

class WarningSpec extends ChiselFlatSpec with Utils {
  behavior.of("Warnings")

  "Warnings" should "be de-duplicated" in {
    object MyEnum extends ChiselEnum {
      val e0, e1, e2 = Value
    }

    class MyModule extends Module {
      val in = IO(Input(UInt(2.W)))
      val out1 = IO(Output(MyEnum()))
      val out2 = IO(Output(MyEnum()))
      out1 := MyEnum(in); out2 := MyEnum(in)
    }
    val (log, _) = grabLog(ChiselStage.elaborate(new MyModule))
    def countSubstring(s: String, sub: String) =
      s.sliding(sub.length).count(_ == sub)
    countSubstring(log, "Casting non-literal UInt") should be(1)
  }
}
