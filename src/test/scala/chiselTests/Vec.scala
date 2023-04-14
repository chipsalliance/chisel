// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import circt.stage.ChiselStage.emitCHIRRTL
import chisel3._
import chisel3.util._

class IOTesterModFill(vecSize: Int) extends Module {
  // This should generate a BindingException when we attempt to wire up the Vec.fill elements
  //  since they're pure types and hence unsynthesizeable.
  val io = IO(new Bundle {
    val in = Input(VecInit(Seq.fill(vecSize) { UInt() }))
    val out = Output(VecInit(Seq.fill(vecSize) { UInt() }))
  })
  io.out := io.in
}

class HugeVecTester(n: Int) extends RawModule {
  require(n > 0)
  val myVec = Wire(Vec(n, UInt()))
  myVec.foreach { x =>
    x := 123.U
  }
}

class ReduceTreeTester extends Module {
  class FooIO[T <: Data](n: Int, private val gen: T) extends Bundle {
    val in = Flipped(Vec(n, new DecoupledIO(gen)))
    val out = new DecoupledIO(gen)
  }

  class Foo[T <: Data](n: Int, private val gen: T) extends Module {
    val io = IO(new FooIO(n, gen))

    def foo(a: DecoupledIO[T], b: DecoupledIO[T]) = {
      a.ready := true.B
      b.ready := true.B
      val out = Wire(new DecoupledIO(gen))

      out.valid := true.B

      val regSel = RegInit(false.B)
      out.bits := Mux(regSel, a.bits, b.bits)
      out.ready := a.ready
      out
    }

    io.out <> io.in.reduceTree(foo)
  }

  val dut = Module(new Foo(5, UInt(5.W)))
  dut.io := DontCare
}

class VecSpec extends ChiselPropSpec with Utils {

  private def uint(value: BigInt): String = uint(value, value.bitLength.max(1))
  private def uint(value: BigInt, width: Int): String =
    s"""UInt<$width>(0h${value.toString(16)})"""

  property("Vecs should be assignable") {
    val values = (0 until 10).toList
    val width = 4
    val chirrtl = emitCHIRRTL(new RawModule {
      val v = VecInit(values.map(_.U(width.W)))
    })
    for (v <- values) {
      chirrtl should include(s"connect v[$v], ${uint(v, width)}")
    }
  }

  property("Vecs should be passed through vec IO") {
    val values = (0 until 10).toList
    val width = 4

    val chirrtl = emitCHIRRTL(new RawModule {
      val out = IO(Output(Vec(values.size, UInt(width.W))))
      val v = VecInit(values.map(_.U(width.W)))
      out := v
    })
    chirrtl should include("output out : UInt<4>[10]")
    chirrtl should include("connect out, v")
  }

  property("Vec.fill with a pure type should generate an exception") {
    a[BindingException] should be thrownBy extractCause[BindingException] {
      ChiselStage.emitCHIRRTL(new IOTesterModFill(8))
    }
  }

  property("VecInit should tabulate correctly") {
    val n = 4
    val w = 3
    val chirrtl = emitCHIRRTL(new RawModule {
      val x = VecInit(Seq.tabulate(n) { i => (i * 2).asUInt })
      val u = VecInit.tabulate(n)(i => (i * 2).asUInt)
    })
    chirrtl should include(s"wire x : UInt<$w>[$n]")
    chirrtl should include(s"wire u : UInt<$w>[$n]")
    for (i <- 0 until n) {
      chirrtl should include(s"connect x[$i], ${uint(i * 2)}")
      chirrtl should include(s"connect u[$i], ${uint(i * 2)}")
    }
  }

  property("VecInit should tabulate 2D vec correctly") {
    val n = 2
    val m = 3
    val w = 2
    val chirrtl = emitCHIRRTL(new RawModule {
      val v = VecInit.tabulate(n, m) { case (i, j) => (i + j).asUInt }
    })
    chirrtl should include(s"wire v : UInt<$w>[$m][$n]")
    for (i <- 0 until n) {
      for (j <- 0 until m) {
        chirrtl should include(s"connect v[$i][$j], ${uint(i + j)}")
      }
    }
  }

  property("VecInit should tabulate 3D vec correctly") {
    val n = 2
    val m = 3
    val o = 2
    val w = 3
    val chirrtl = emitCHIRRTL(new RawModule {
      val v = VecInit.tabulate(n, m, o) { case (i, j, k) => (i + j + k).asUInt }
    })
    chirrtl should include(s"wire v : UInt<$w>[$o][$m][$n]")
    for (i <- 0 until n) {
      for (j <- 0 until m) {
        for (k <- 0 until o) {
          chirrtl should include(s"connect v[$i][$j][$k], ${uint(i + j + k)}")
        }
      }
    }
  }

  property("VecInit should fill correctly") {
    val n = 4
    val value = 13
    val w = 4
    val chirrtl = emitCHIRRTL(new RawModule {
      val x = VecInit(Seq.fill(n)(value.U))
      val u = VecInit.fill(n)(value.U)
    })
    chirrtl should include(s"wire x : UInt<$w>[$n]")
    chirrtl should include(s"wire u : UInt<$w>[$n]")
    val valueFir = uint(value)
    for (i <- 0 until n) {
      chirrtl should include(s"connect x[$i], $valueFir")
      chirrtl should include(s"connect u[$i], $valueFir")
    }
  }

  property("VecInit.fill should support size 0 Vecs") {
    val chirrtl = emitCHIRRTL(new RawModule {
      val out = IO(Output(Vec(0, UInt(8.W))))
      val u = VecInit.fill(0)(8.U)
      out := u
    })
    chirrtl should include("wire u : UInt<4>[0]")
  }

  property("VecInit should fill 2D vec correctly") {
    val n = 2
    val m = 3
    val value = 7
    val w = 3
    val chirrtl = emitCHIRRTL(new RawModule {
      val v = VecInit.fill(n, m)(value.asUInt)
    })
    chirrtl should include(s"wire v : UInt<$w>[$m][$n]")
    val valueFir = uint(value)
    for (i <- 0 until n) {
      for (j <- 0 until m) {
        chirrtl should include(s"connect v[$i][$j], $valueFir")
      }
    }
  }

  property("VecInit should fill 3D vec correctly") {
    val n = 2
    val m = 3
    val o = 2
    val value = 11
    val w = 4
    val chirrtl = emitCHIRRTL(new RawModule {
      val v = VecInit.fill(n, m, o)(value.asUInt)
    })
    chirrtl should include(s"wire v : UInt<$w>[$o][$m][$n]")
    val valueFir = uint(value)
    for (i <- 0 until n) {
      for (j <- 0 until m) {
        for (k <- 0 until o) {
          chirrtl should include(s"connect v[$i][$j][$k], $valueFir")
        }
      }
    }
  }

  property("VecInit should support 2D fill bidirectional wire connection") {
    val n = 2
    val m = 3
    val chirrtl = emitCHIRRTL(new Module {
      val vec2D = VecInit.fill(n, m) {
        val mod = Module(new PassthroughModule)
        mod.io
      }
    })
    var idx = 0
    for (i <- 0 until n) {
      for (j <- 0 until m) {
        val suffix = if (idx > 0) s"_$idx" else ""
        chirrtl should include(s"connect vec2D[$i][$j].out, vec2D_mod$suffix.io.out")
        chirrtl should include(s"connect vec2D_mod$suffix.io.in, vec2D[$i][$j].in")
        idx += 1
      }
    }
  }

  property("VecInit should support 3D fill bidirectional wire connection") {
    val n = 2
    val m = 3
    val o = 2
    val chirrtl = emitCHIRRTL(new Module {
      val vec3D = VecInit.fill(n, m, o) {
        val mod = Module(new PassthroughModule)
        mod.io
      }
    })
    var idx = 0
    for (i <- 0 until n) {
      for (j <- 0 until m) {
        for (k <- 0 until o) {
          val suffix = if (idx > 0) s"_$idx" else ""
          chirrtl should include(s"connect vec3D[$i][$j][$k].out, vec3D_mod$suffix.io.out")
          chirrtl should include(s"connect vec3D_mod$suffix.io.in, vec3D[$i][$j][$k].in")
          idx += 1
        }
      }
    }
  }

  property("VecInit should support 2D tabulate bidirectional wire connection") {
    val n = 2
    val m = 3
    val chirrtl = emitCHIRRTL(new Module {
      val mods = Vector.fill(n, m)(Module(new PassthroughModule))
      val vec2D = VecInit.tabulate(n, m) { (i, j) =>
        // Swizzle a bit for fun and profit
        mods((i + 1) % n)((j + 2) % m).io
      }
    })
    for (i <- 0 until n) {
      for (j <- 0 until m) {
        val suffix = s"_${(i + 1) % n}_${(j + 2) % m}"
        chirrtl should include(s"connect vec2D[$i][$j].out, mods$suffix.io.out")
        chirrtl should include(s"connect mods$suffix.io.in, vec2D[$i][$j].in")
      }
    }
  }

  property("VecInit should support 3D tabulate bidirectional wire connection") {
    val n = 2
    val m = 3
    val o = 2
    val chirrtl = emitCHIRRTL(new Module {
      val mods = Vector.fill(n, m, o)(Module(new PassthroughModule))
      val vec2D = VecInit.tabulate(n, m, o) { (i, j, k) =>
        // Swizzle a bit for fun and profit
        mods((i + 1) % n)((j + 2) % m)(k).io
      }
    })
    for (i <- 0 until n) {
      for (j <- 0 until m) {
        for (k <- 0 until o) {
          val suffix = s"_${(i + 1) % n}_${(j + 2) % m}_$k"
          chirrtl should include(s"connect vec2D[$i][$j][$k].out, mods$suffix.io.out")
          chirrtl should include(s"connect mods$suffix.io.in, vec2D[$i][$j][$k].in")
        }
      }
    }
  }

  property("Infering widths on huge Vecs should not cause a stack overflow") {
    ChiselStage.emitSystemVerilog(new HugeVecTester(10000))
  }

  property("A Reg of a Vec of a single 1 bit element should compile and work") {
    val chirrtl = emitCHIRRTL(new Module {
      val oneBitUnitRegVec = Reg(Vec(1, UInt(1.W)))
      oneBitUnitRegVec(0) := 1.U(1.W)
    })
    chirrtl should include("reg oneBitUnitRegVec : UInt<1>[1], clock")
    chirrtl should include("connect oneBitUnitRegVec[0], UInt<1>(0h1)")
  }

  property("A Vec with zero entries should compile and have zero width") {

    val chirrtl = emitCHIRRTL(new Module {
      require(Vec(0, Bool()).getWidth == 0)

      val bundleWithZeroEntryVec = new Bundle {
        val foo = Bool()
        val bar = Vec(0, Bool())
      }
      require(bundleWithZeroEntryVec.getWidth == 1)

      val m = Module(new Module {
        val io = IO(Output(bundleWithZeroEntryVec))
        val zero = 0.U.asTypeOf(bundleWithZeroEntryVec)
        require(zero.getWidth == 1)
        io := zero
      })
      val w = WireDefault(m.io.bar)

    })
    chirrtl should include("output io : { foo : UInt<1>, bar : UInt<1>[0]}")
    chirrtl should include("wire zero : { foo : UInt<1>, bar : UInt<1>[0]}")
    chirrtl should include("connect zero.foo, UInt<1>(0h0)")
    chirrtl should include("connect io, zero")
    chirrtl should include("wire w : UInt<1>[0]")
    chirrtl should include("connect w, m.io.bar")
  }

  property("It should be possible to bulk connect a Vec and a Seq") {
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val out = Output(Vec(4, UInt(8.W)))
      })
      val seq = Seq.fill(4)(0.U)
      io.out <> seq
    })
  }

  property("Bulk connecting a Vec and Seq of different sizes should report a ChiselException") {
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL(new Module {
        val io = IO(new Bundle {
          val out = Output(Vec(4, UInt(8.W)))
        })
        val seq = Seq.fill(5)(0.U)
        io.out <> seq
      })
    }
  }

  property("It should be possible to initialize a Vec with DontCare") {
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val out = Output(Vec(4, UInt(8.W)))
      })
      io.out := VecInit(Seq(4.U, 5.U, DontCare, 2.U))
    })
  }

  property("Indexing a Chisel type Vec by a hardware type should give a sane error message") {
    a[ExpectedHardwareException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL {
        new Module {
          val io = IO(new Bundle {})
          val foo = Vec(2, Bool())
          foo(0.U) := false.B
        }
      }
    }
  }

  property("reduceTree should preserve input/output type") {
    ChiselStage.emitCHIRRTL(new ReduceTreeTester)
  }

  property("Vecs of empty Bundles and empty Records should work") {
    class MyModule(gen: Record) extends Module {
      val idx = IO(Input(UInt(2.W)))
      val in = IO(Input(gen))
      val out = IO(Output(gen))

      val reg = RegInit(0.U.asTypeOf(Vec(4, gen)))
      reg(idx) := in
      out := reg(idx)
    }
    class EmptyBundle extends Bundle
    class EmptyRecord extends Record {
      val elements = collection.immutable.ListMap.empty
    }
    for (gen <- List(new EmptyBundle, new EmptyRecord)) {
      val chirrtl = ChiselStage.emitCHIRRTL(new MyModule(gen))
      chirrtl should include("input in : { }")
      chirrtl should include("regreset reg : { }[4]")
    }
  }

  property("Vecs should emit static indices when indexing with a literal UInt and not warn based on width") {
    val (log, chirrtl) = grabLog(emitCHIRRTL(new RawModule {
      val vec = IO(Input(Vec(4, UInt(8.W))))
      val out = IO(Output(UInt(8.W)))
      out := vec(1.U)
    }))
    chirrtl should include("connect out, vec[1]")
    log should be("")
  }

  property("Vecs should warn on out-of-bounds literal indices") {
    val (log, chirrtl) = grabLog(emitCHIRRTL(new RawModule {
      val vec = IO(Input(Vec(4, UInt(8.W))))
      val out = IO(Output(UInt(8.W)))
      out := vec(10.U)
    }))
    chirrtl should include("""connect out, vec[UInt<2>(0h2)]""")
    log should include("Dynamic index with width 4 is too wide for Vec of size 4 (expected index width 2)")
  }

  property("Vecs should warn on too large dynamic indices") {
    val (log, _) = grabLog(emitCHIRRTL(new RawModule {
      val vec = IO(Input(Vec(7, UInt(8.W))))
      val idx = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      out := vec(idx)
    }))
    log should include("Dynamic index with width 8 is too wide for Vec of size 7 (expected index width 3)")
  }

  property("Vecs should warn on too small dynamic indices") {
    val (log, _) = grabLog(emitCHIRRTL(new RawModule {
      val vec = IO(Input(Vec(7, UInt(8.W))))
      val idx = IO(Input(UInt(2.W)))
      val out = IO(Output(UInt(8.W)))
      out := vec(idx)
    }))
    log should include("Dynamic index with width 2 is too narrow for Vec of size 7 (expected index width 3)")
  }

  // This will have to be checked in firtool
  property("Vecs should not warn on inferred with dynamic indices") {
    val (log, _) = grabLog(emitCHIRRTL(new RawModule {
      val vec = IO(Input(Vec(7, UInt(8.W))))
      val idx = IO(Input(UInt(2.W)))
      val out = IO(Output(UInt(8.W)))
      val jdx = WireInit(UInt(), idx)
      out := vec(jdx)
    }))
    log should be("")
  }
}
