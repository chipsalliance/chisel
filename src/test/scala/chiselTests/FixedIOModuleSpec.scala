// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage

class FixedIOModuleSpec extends ChiselFlatSpec with Utils with MatchesAndOmits {

  "FixedIOModule" should "create a module with flattened IO" in {

    class Foo(width: Int) extends FixedIORawModule[UInt](UInt(width.W)) {
      io :<>= DontCare
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo(8)))("output io : UInt<8>")()
  }

  "PolymoprhicIOModule error behavior" should "disallow further IO creation" in {
    class Foo extends FixedIORawModule[Bool](Bool()) {
      val a = IO(Bool())
    }
    val exception = intercept[IllegalArgumentException] {
      ChiselStage.emitCHIRRTL(new Foo)
    }
    exception.getMessage should include("This module cannot have user-created IO")
  }

  "FixedIOBlackBox" should "create a module with flattend IO" in {

    class Bar extends FixedIOExtModule(UInt(1.W))

    class BazIO extends Bundle {
      val a = UInt(2.W)
    }

    class Baz extends FixedIOExtModule(new BazIO)

    class Foo extends RawModule {
      val bar = Module(new Bar)
      val baz = Module(new Baz)
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))(
      "output io : UInt<1>",
      "output a : UInt<2>"
    )()
  }

}
