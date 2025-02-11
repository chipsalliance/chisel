// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental

import chisel3._
import chisel3.util.Valid
import circt.stage.ChiselStage.emitCHIRRTL
import chisel3.experimental.Analog
import chiselTests.{ChiselFlatSpec, FileCheck}
import chisel3.reflect.DataMirror
import scala.collection.immutable.SeqMap
import circt.stage.ChiselStage

class FlatIOSpec extends ChiselFlatSpec with FileCheck {
  behavior.of("FlatIO")

  it should "create ports without a prefix" in {
    class MyModule extends RawModule {
      val io = FlatIO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      io.out := io.in
    }
    generateFirrtlAndFileCheck(new MyModule)(
      """|CHECK:      input in : UInt<8>
         |CHECK-NEXT: output out : UInt<8>
         |CHECK:      connect out, in
         |"""".stripMargin
    )
  }

  it should "support bulk connections between FlatIOs and regular IOs" in {
    class MyModule extends RawModule {
      val in = FlatIO(Input(Valid(UInt(8.W))))
      val out = IO(Output(Valid(UInt(8.W))))
      out := in
    }
    generateFirrtlAndFileCheck(new MyModule)(
      """|CHECK:      connect out.bits, bits
         |CHECK-NEXT: connect out.valid, valid
         |"""".stripMargin
    )
  }

  it should "support dynamically indexing Vecs inside of FlatIOs" in {
    class MyModule extends RawModule {
      val io = FlatIO(new Bundle {
        val addr = Input(UInt(2.W))
        val in = Input(Vec(4, UInt(8.W)))
        val out = Output(Vec(4, UInt(8.W)))
      })
      io.out(io.addr) := io.in(io.addr)
    }
    val chirrtl = emitCHIRRTL(new MyModule)
    chirrtl should include("connect out[addr], in[addr]")
  }

  it should "support Analog members" in {
    class MyBundle extends Bundle {
      val foo = Output(UInt(8.W))
      val bar = Analog(8.W)
    }
    class MyModule extends RawModule {
      val io = FlatIO(new Bundle {
        val in = Flipped(new MyBundle)
        val out = new MyBundle
      })
      io.out <> io.in
    }
    generateFirrtlAndFileCheck(new MyModule)(
      """|CHECK:      attach (out.bar, in.bar)
         |CHECK-NEXT: connect out.foo, in.foo
         |"""".stripMargin
    )
  }

  it should "be an `IO` for elements and vectors" in {

    class Foo extends RawModule {
      val a = FlatIO(UInt(1.W))
      val b = FlatIO(Vec(2, UInt(2.W)))
    }
    generateFirrtlAndFileCheck(new Foo)(
      """|CHECK:      output a : UInt<1>
         |CHECK-NEXT: output b : UInt<2>[2]
         |"""".stripMargin
    )
  }

  it should "maintain port order for Bundles" in {
    class MyBundle extends Bundle {
      val foo = Bool()
      val bar = Bool()
    }
    class MyModule extends Module {
      val io = IO(Input(new MyBundle))
    }
    class MyFlatIOModule extends Module {
      val io = FlatIO(Input(new MyBundle))
    }

    generateSystemVerilogAndFileCheck(new MyModule)(
      """|CHECK:      io_foo
         |CHECK-NEXT: io_bar
         |"""".stripMargin
    )
    generateSystemVerilogAndFileCheck(new MyFlatIOModule)(
      """|CHECK:      foo
         |CHECK-NEXT: bar
         |"""".stripMargin
    )
  }

  it should "maintain port order for Records" in {
    class MyRecord extends Record {
      val elements = SeqMap("foo" -> Bool(), "bar" -> Bool())
    }
    class MyModule extends Module {
      val io = IO(Input(new MyRecord))
    }
    class MyFlatIOModule extends Module {
      val io = FlatIO(Input(new MyRecord))
    }
    generateSystemVerilogAndFileCheck(new MyModule)(
      """|CHECK:      io_bar
         |CHECK-NEXT: io_foo
         |"""".stripMargin
    )
    generateSystemVerilogAndFileCheck(new MyFlatIOModule)(
      """|CHECK:      bar
         |CHECK-NEXT: foo
         |"""".stripMargin
    )
  }

}
