// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import chisel3.util._

class WarningSpec extends ChiselFlatSpec with Utils {
  behavior.of("Warnings")

  object MyEnum extends ChiselEnum {
    val e0, e1, e2 = Value
  }

  class MyModule extends Module {
    val in = IO(Input(UInt(2.W)))
    val out1 = IO(Output(MyEnum()))
    val out2 = IO(Output(MyEnum()))
    def func(out: EnumType): Unit = {
      out := MyEnum(in)
    }
    func(out1)
    func(out2)
  }

  "Warnings" should "be de-duplicated" in {
    val (log, _) = grabLog(ChiselStage.emitCHIRRTL(new MyModule))
    def countSubstring(s: String, sub: String) =
      s.sliding(sub.length).count(_ == sub)
    countSubstring(log, "Casting non-literal UInt") should be(1)
  }

  "Warnings" should "be treated as errors with warningsAsErrors" in {
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      val args = Array("--warnings-as-errors")
      ChiselStage.emitCHIRRTL(new MyModule, args)
    }
  }
}
