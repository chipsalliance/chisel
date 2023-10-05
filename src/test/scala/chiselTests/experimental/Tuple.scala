// See LICENSE for license details.

package chiselTests.experimental

import chisel3._
import chisel3.experimental.conversions._
import chiselTests.ChiselFlatSpec
import circt.stage.ChiselStage

class TupleSpec extends ChiselFlatSpec {

  behavior.of("Tuple")

  it should "enable using Tuple2 like Data" in {
    class MyModule extends Module {
      val a, b, c, d = IO(Input(UInt(8.W)))
      val sel = IO(Input(Bool()))
      val y, z = IO(Output(UInt(8.W)))
      (y, z) := Mux(sel, (a, b), (c, d))
    }
    // Verilog instead of CHIRRTL because the optimizations make it much prettier
    val verilog = ChiselStage.emitSystemVerilog(new MyModule)
    verilog should include("assign y = sel ? a : c;")
    verilog should include("assign z = sel ? b : d;")
  }

  it should "support nesting of tuples" in {
    class MyModule extends Module {
      val a, b, c, d = IO(Input(UInt(8.W)))
      val w, x, y, z = IO(Output(UInt(8.W)))
      ((w, x), (y, z)) := ((a, b), (c, d))
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect w, a")
    chirrtl should include("connect x, b")
    chirrtl should include("connect y, c")
    chirrtl should include("connect z, d")
  }

  it should "enable using Tuple3 like Data" in {
    class MyModule extends Module {
      val a, b, c = IO(Input(UInt(8.W)))
      val f, g, h = IO(Input(UInt(8.W)))
      val sel = IO(Input(Bool()))
      val v, w, x = IO(Output(UInt(8.W)))
      (v, w, x) := Mux(sel, (a, b, c), (f, g, h))
    }
    // Verilog instead of CHIRRTL because the optimizations make it much prettier
    val verilog = ChiselStage.emitSystemVerilog(new MyModule)
    verilog should include("assign v = sel ? a : f;")
    verilog should include("assign w = sel ? b : g;")
    verilog should include("assign x = sel ? c : h;")
  }

  it should "enable using Tuple4 like Data" in {
    class MyModule extends Module {
      val a, b, c, d = IO(Input(UInt(8.W)))
      val f, g, h, i = IO(Input(UInt(8.W)))
      val sel = IO(Input(Bool()))
      val v, w, x, y = IO(Output(UInt(8.W)))
      (v, w, x, y) := Mux(sel, (a, b, c, d), (f, g, h, i))
    }
    // Verilog instead of CHIRRTL because the optimizations make it much prettier
    val verilog = ChiselStage.emitSystemVerilog(new MyModule)
    verilog should include("assign v = sel ? a : f;")
    verilog should include("assign w = sel ? b : g;")
    verilog should include("assign x = sel ? c : h;")
    verilog should include("assign y = sel ? d : i;")
  }

  it should "enable using Tuple5 like Data" in {
    class MyModule extends Module {
      val a0, a1, a2, a3, a4 = IO(Input(UInt(8.W)))
      val b0, b1, b2, b3, b4 = IO(Input(UInt(8.W)))
      val sel = IO(Input(Bool()))
      val z0, z1, z2, z3, z4 = IO(Output(UInt(8.W)))
      (z0, z1, z2, z3, z4) := Mux(sel, (a0, a1, a2, a3, a4), (b0, b1, b2, b3, b4))
    }
    // Verilog instead of CHIRRTL because the optimizations make it much prettier
    val verilog = ChiselStage.emitSystemVerilog(new MyModule)
    for (i <- 0 until 5) {
      verilog should include(s"assign z$i = sel ? a$i : b$i;")
    }
  }

  it should "enable using Tuple6 like Data" in {
    class MyModule extends Module {
      val a0, a1, a2, a3, a4, a5 = IO(Input(UInt(8.W)))
      val b0, b1, b2, b3, b4, b5 = IO(Input(UInt(8.W)))
      val sel = IO(Input(Bool()))
      val z0, z1, z2, z3, z4, z5 = IO(Output(UInt(8.W)))
      (z0, z1, z2, z3, z4, z5) := Mux(sel, (a0, a1, a2, a3, a4, a5), (b0, b1, b2, b3, b4, b5))
    }
    // Verilog instead of CHIRRTL because the optimizations make it much prettier
    val verilog = ChiselStage.emitSystemVerilog(new MyModule)
    for (i <- 0 until 6) {
      verilog should include(s"assign z$i = sel ? a$i : b$i;")
    }
  }

  it should "enable using Tuple7 like Data" in {
    class MyModule extends Module {
      val a0, a1, a2, a3, a4, a5, a6 = IO(Input(UInt(8.W)))
      val b0, b1, b2, b3, b4, b5, b6 = IO(Input(UInt(8.W)))
      val sel = IO(Input(Bool()))
      val z0, z1, z2, z3, z4, z5, z6 = IO(Output(UInt(8.W)))
      (z0, z1, z2, z3, z4, z5, z6) := Mux(sel, (a0, a1, a2, a3, a4, a5, a6), (b0, b1, b2, b3, b4, b5, b6))
    }
    // Verilog instead of CHIRRTL because the optimizations make it much prettier
    val verilog = ChiselStage.emitSystemVerilog(new MyModule)
    for (i <- 0 until 7) {
      verilog should include(s"assign z$i = sel ? a$i : b$i;")
    }
  }

  it should "enable using Tuple8 like Data" in {
    class MyModule extends Module {
      val a0, a1, a2, a3, a4, a5, a6, a7 = IO(Input(UInt(8.W)))
      val b0, b1, b2, b3, b4, b5, b6, b7 = IO(Input(UInt(8.W)))
      val sel = IO(Input(Bool()))
      val z0, z1, z2, z3, z4, z5, z6, z7 = IO(Output(UInt(8.W)))
      (z0, z1, z2, z3, z4, z5, z6, z7) := Mux(sel, (a0, a1, a2, a3, a4, a5, a6, a7), (b0, b1, b2, b3, b4, b5, b6, b7))
    }
    // Verilog instead of CHIRRTL because the optimizations make it much prettier
    val verilog = ChiselStage.emitSystemVerilog(new MyModule)
    for (i <- 0 until 8) {
      verilog should include(s"assign z$i = sel ? a$i : b$i;")
    }
  }

  it should "enable using Tuple9 like Data" in {
    class MyModule extends Module {
      val a0, a1, a2, a3, a4, a5, a6, a7, a8 = IO(Input(UInt(8.W)))
      val b0, b1, b2, b3, b4, b5, b6, b7, b8 = IO(Input(UInt(8.W)))
      val sel = IO(Input(Bool()))
      val z0, z1, z2, z3, z4, z5, z6, z7, z8 = IO(Output(UInt(8.W)))
      (z0, z1, z2, z3, z4, z5, z6, z7, z8) :=
        Mux(sel, (a0, a1, a2, a3, a4, a5, a6, a7, a8), (b0, b1, b2, b3, b4, b5, b6, b7, b8))
    }
    // Verilog instead of CHIRRTL because the optimizations make it much prettier
    val verilog = ChiselStage.emitSystemVerilog(new MyModule)
    for (i <- 0 until 9) {
      verilog should include(s"assign z$i = sel ? a$i : b$i;")
    }
  }

  it should "enable using Tuple10 like Data" in {
    class MyModule extends Module {
      val a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = IO(Input(UInt(8.W)))
      val b0, b1, b2, b3, b4, b5, b6, b7, b8, b9 = IO(Input(UInt(8.W)))
      val sel = IO(Input(Bool()))
      val z0, z1, z2, z3, z4, z5, z6, z7, z8, z9 = IO(Output(UInt(8.W)))
      (z0, z1, z2, z3, z4, z5, z6, z7, z8, z9) :=
        Mux(sel, (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9), (b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    }
    // Verilog instead of CHIRRTL because the optimizations make it much prettier
    val verilog = ChiselStage.emitSystemVerilog(new MyModule)
    for (i <- 0 until 10) {
      verilog should include(s"assign z$i = sel ? a$i : b$i;")
    }
  }

}
