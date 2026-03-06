// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class FirrtlKeywordEscapingSpec extends AnyFlatSpec with Matchers with FileCheck {

  behavior.of("FIRRTL keyword escaping")

  it should "escape FIRRTL type keywords used as port names" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val UInt = IO(chisel3.UInt(2.W))
        UInt :<= 3.U
      }
    }.fileCheck() {
      """|CHECK: output `UInt` : UInt<2>
         |CHECK: connect `UInt`, UInt<2>(0h3)
         |""".stripMargin
    }
  }

  it should "compile the user's example to SystemVerilog without errors" in {
    class Foo extends RawModule {
      val UInt = IO(chisel3.UInt(2.W))
      UInt :<= 3.U
    }

    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck() {
        """|CHECK: output `UInt` : UInt<2>
           |CHECK: connect `UInt`, UInt<2>(0h3)
           |""".stripMargin
      }

    // This should not throw an error
    val sv = ChiselStage.emitSystemVerilog(
      gen = new Foo,
      args = Array("--throw-on-first-error")
    )
    sv should not be empty
  }

  it should "escape FIRRTL keywords used as wire names" in {
    ChiselStage
      .emitCHIRRTL(new Module {
        val io = IO(new Bundle {})
        val wire = Wire(chisel3.UInt(8.W))
        val reg = Wire(chisel3.UInt(8.W))
        val node = Wire(chisel3.UInt(8.W))
        val Clock = Wire(chisel3.UInt(8.W))
        val Reset = Wire(chisel3.UInt(8.W))
        val mux = Wire(chisel3.UInt(8.W))

        wire := 1.U
        reg := 2.U
        node := 3.U
        Clock := 4.U
        Reset := 5.U
        mux := 6.U
      })
      .fileCheck() {
        """|CHECK: wire `wire` : UInt<8>
           |CHECK: wire `reg` : UInt<8>
           |CHECK: wire `node` : UInt<8>
           |CHECK: wire `Clock` : UInt<8>
           |CHECK: wire `Reset` : UInt<8>
           |CHECK: wire `mux` : UInt<8>
           |""".stripMargin
      }
  }

  it should "escape FIRRTL statement keywords" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val when = IO(Output(chisel3.UInt(8.W)))
        val skip = IO(Output(chisel3.UInt(8.W)))
        val inst = IO(Output(chisel3.UInt(8.W)))

        when :<= 1.U
        skip :<= 2.U
        inst :<= 3.U
      })
      .fileCheck() {
        """|CHECK: output `when` : UInt<8>
           |CHECK: output `skip` : UInt<8>
           |CHECK: output `inst` : UInt<8>
           |""".stripMargin
      }
  }

  it should "escape FIRRTL primop keywords" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val add = IO(Output(chisel3.UInt(8.W)))
        val sub = IO(Output(chisel3.UInt(8.W)))
        val mul = IO(Output(chisel3.UInt(8.W)))
        val and = IO(Output(chisel3.UInt(8.W)))
        val or = IO(Output(chisel3.UInt(8.W)))
        val xor = IO(Output(chisel3.UInt(8.W)))

        add :<= 1.U
        sub :<= 2.U
        mul :<= 3.U
        and :<= 4.U
        or :<= 5.U
        xor :<= 6.U
      })
      .fileCheck() {
        """|CHECK: output `add` : UInt<8>
           |CHECK: output `sub` : UInt<8>
           |CHECK: output `mul` : UInt<8>
           |CHECK: output `and` : UInt<8>
           |CHECK: output `or` : UInt<8>
           |CHECK: output `xor` : UInt<8>
           |""".stripMargin
      }
  }

  it should "escape module-level keywords" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val module = IO(Output(chisel3.UInt(8.W)))
        val circuit = IO(Output(chisel3.UInt(8.W)))
        val input = IO(Output(chisel3.UInt(8.W)))
        val output = IO(Output(chisel3.UInt(8.W)))

        module :<= 1.U
        circuit :<= 2.U
        input :<= 3.U
        output :<= 4.U
      })
      .fileCheck() {
        """|CHECK: output `module` : UInt<8>
           |CHECK: output `circuit` : UInt<8>
           |CHECK: output `input` : UInt<8>
           |CHECK: output `output` : UInt<8>
           |""".stripMargin
      }
  }

  it should "escape connect-like keywords" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val connect = IO(Output(chisel3.UInt(8.W)))
        val invalidate = IO(Output(chisel3.UInt(8.W)))
        val attach = IO(Output(chisel3.UInt(8.W)))

        connect :<= 1.U
        invalidate :<= 2.U
        attach :<= 3.U
      })
      .fileCheck() {
        """|CHECK: output `connect` : UInt<8>
           |CHECK: output `invalidate` : UInt<8>
           |CHECK: output `attach` : UInt<8>
           |""".stripMargin
      }
  }

  it should "escape command keywords" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val printf = IO(Output(chisel3.UInt(8.W)))
        val assert = IO(Output(chisel3.UInt(8.W)))
        val assume = IO(Output(chisel3.UInt(8.W)))
        val cover = IO(Output(chisel3.UInt(8.W)))

        printf :<= 1.U
        assert :<= 2.U
        assume :<= 3.U
        cover :<= 4.U
      })
      .fileCheck() {
        """|CHECK: output `printf` : UInt<8>
           |CHECK: output `assert` : UInt<8>
           |CHECK: output `assume` : UInt<8>
           |CHECK: output `cover` : UInt<8>
           |""".stripMargin
      }
  }

  it should "escape CIRCT-specific keywords like Unknown" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val Unknown = IO(Output(chisel3.UInt(8.W)))
        val Bool = IO(Output(chisel3.UInt(1.W)))
        val reset = IO(Output(chisel3.UInt(1.W)))

        Unknown :<= 42.U
        Bool :<= 1.U
        reset :<= 0.U
      })
      .fileCheck() {
        """|CHECK: output `Unknown` : UInt<8>
           |CHECK: output `Bool` : UInt<1>
           |CHECK: output `reset` : UInt<1>
           |""".stripMargin
      }
  }
}
