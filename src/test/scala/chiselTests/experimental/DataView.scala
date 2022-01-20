// See LICENSE for license details.

package chiselTests.experimental

import chiselTests.ChiselFlatSpec
import chisel3._
import chisel3.experimental.dataview._
import chisel3.experimental.conversions._
import chisel3.experimental.DataMirror.internal.chiselTypeClone
import chisel3.experimental.HWTuple2
import chisel3.stage.ChiselStage
import chisel3.util.{Decoupled, DecoupledIO}

object SimpleBundleDataView {
  class BundleA(val w: Int) extends Bundle {
    val foo = UInt(w.W)
  }
  class BundleB(val w: Int) extends Bundle {
    val bar = UInt(w.W)
  }
  implicit val v1 = DataView[BundleA, BundleB](a => new BundleB(a.w), _.foo -> _.bar)
  implicit val v2 = v1.invert(b => new BundleA(b.w))
}

object VecBundleDataView {
  class MyBundle extends Bundle {
    val foo = UInt(8.W)
    val bar = UInt(8.W)
  }
  implicit val v1: DataView[MyBundle, Vec[UInt]] = DataView(_ => Vec(2, UInt(8.W)), _.foo -> _(1), _.bar -> _(0))
  implicit val v2 = v1.invert(_ => new MyBundle)
}

object FlatDecoupledDataView {
  class FizzBuzz extends Bundle {
    val fizz = UInt(8.W)
    val buzz = UInt(8.W)
  }
  class FlatDecoupled extends Bundle {
    val valid = Output(Bool())
    val ready = Input(Bool())
    val fizz = Output(UInt(8.W))
    val buzz = Output(UInt(8.W))
  }
  implicit val view = DataView[FlatDecoupled, DecoupledIO[FizzBuzz]](
    _ => Decoupled(new FizzBuzz),
    _.valid -> _.valid,
    _.ready -> _.ready,
    _.fizz -> _.bits.fizz,
    _.buzz -> _.bits.buzz
  )
  implicit val view2 = view.invert(_ => new FlatDecoupled)
}

class DataViewSpec extends ChiselFlatSpec {

  behavior of "DataView"

  it should "support simple Bundle viewing" in {
    import SimpleBundleDataView._
    class MyModule extends Module {
      val in = IO(Input(new BundleA(8)))
      val out = IO(Output(new BundleB(8)))
      out := in.viewAs[BundleB]
    }
    val chirrtl = ChiselStage.emitChirrtl(new MyModule)
    chirrtl should include("out.bar <= in.foo")
  }

  it should "be a bidirectional mapping" in {
    import SimpleBundleDataView._
    class MyModule extends Module {
      val in = IO(Input(new BundleA(8)))
      val out = IO(Output(new BundleB(8)))
      out.viewAs[BundleA] := in
    }
    val chirrtl = ChiselStage.emitChirrtl(new MyModule)
    chirrtl should include("out.bar <= in.foo")
  }

  it should "handle viewing UInts as UInts" in {
    class MyModule extends Module {
      val in = IO(Input(UInt(8.W)))
      val foo = IO(Output(UInt(8.W)))
      val bar = IO(Output(UInt(8.W)))
      foo := in.viewAs[UInt]
      bar.viewAs[UInt] := in
    }
    val chirrtl = ChiselStage.emitChirrtl(new MyModule)
    chirrtl should include("foo <= in")
    chirrtl should include("bar <= in")
  }

  it should "handle viewing Bundles as their same concrete type" in {
    class MyBundle extends Bundle {
      val foo = UInt(8.W)
    }
    class MyModule extends Module {
      val in = IO(Input(new MyBundle))
      val fizz = IO(Output(new MyBundle))
      val buzz = IO(Output(new MyBundle))
      fizz := in.viewAs[MyBundle]
      buzz.viewAs[MyBundle] := in
    }
    val chirrtl = ChiselStage.emitChirrtl(new MyModule)
    chirrtl should include("fizz.foo <= in.foo")
    chirrtl should include("buzz.foo <= in.foo")
  }

  it should "handle viewing Vecs as their same concrete type" in {
    class MyModule extends Module {
      val in = IO(Input(Vec(1, UInt(8.W))))
      val fizz = IO(Output(Vec(1, UInt(8.W))))
      val buzz = IO(Output(Vec(1, UInt(8.W))))
      fizz := in.viewAs[Vec[UInt]]
      buzz.viewAs[Vec[UInt]] := in
    }
    val chirrtl = ChiselStage.emitChirrtl(new MyModule)
    chirrtl should include("fizz[0] <= in[0]")
    chirrtl should include("buzz[0] <= in[0]")
  }

  it should "handle viewing Vecs as Bundles and vice versa" in {
    import VecBundleDataView._
    class MyModule extends Module {
      val in = IO(Input(new MyBundle))
      val out = IO(Output(Vec(2, UInt(8.W))))
      val out2 = IO(Output(Vec(2, UInt(8.W))))
      out := in.viewAs[Vec[UInt]]
      out2.viewAs[MyBundle] := in
    }
    val chirrtl = ChiselStage.emitChirrtl(new MyModule)
    chirrtl should include("out[0] <= in.bar")
    chirrtl should include("out[1] <= in.foo")
    chirrtl should include("out2[0] <= in.bar")
    chirrtl should include("out2[1] <= in.foo")
  }

  it should "work with bidirectional connections for nested types" in {
    import FlatDecoupledDataView._
    class MyModule extends Module {
      val enq = IO(Flipped(Decoupled(new FizzBuzz)))
      val deq = IO(new FlatDecoupled)
      val deq2 = IO(new FlatDecoupled)
      deq <> enq.viewAs[FlatDecoupled]
      deq2.viewAs[DecoupledIO[FizzBuzz]] <> enq
    }
    val chirrtl = ChiselStage.emitChirrtl(new MyModule)
    chirrtl should include("deq.valid <= enq.valid")
    chirrtl should include("enq.ready <= deq.ready")
    chirrtl should include("deq.fizz <= enq.bits.fizz")
    chirrtl should include("deq.buzz <= enq.bits.buzz")
    chirrtl should include("deq2.valid <= enq.valid")
    chirrtl should include("enq.ready <= deq2.ready")
    chirrtl should include("deq2.fizz <= enq.bits.fizz")
    chirrtl should include("deq2.buzz <= enq.bits.buzz")
  }

  it should "support viewing a Bundle as a Parent Bundle type" in {
    class Foo extends Bundle {
      val foo = UInt(8.W)
    }
    class Bar extends Foo {
      val bar = UInt(8.W)
    }
    class MyModule extends Module {
      val fooIn = IO(Input(new Foo))
      val barOut = IO(Output(new Bar))
      barOut.viewAsSupertype(new Foo) := fooIn

      val barIn = IO(Input(new Bar))
      val fooOut = IO(Output(new Foo))
      fooOut := barIn.viewAsSupertype(new Foo)
    }
    val chirrtl = ChiselStage.emitChirrtl(new MyModule)
    chirrtl should include("barOut.foo <= fooIn.foo")
    chirrtl should include("fooOut.foo <= barIn.foo")
  }

  it should "error if viewing a parent Bundle as a child Bundle type" in {
    assertTypeError("""
      class Foo extends Bundle {
        val foo = UInt(8.W)
      }
      class Bar extends Foo {
        val bar = UInt(8.W)
      }
      class MyModule extends Module {
        val barIn = IO(Input(new Bar))
        val fooOut = IO(Output(new Foo))
        fooOut.viewAs(new Bar) := barIn
      }
    """)
  }

  it should "work in UInt operations" in {
    class MyBundle extends Bundle {
      val value = UInt(8.W)
    }
    class MyModule extends Module {
      val a = IO(Input(UInt(8.W)))
      val b = IO(Input(new MyBundle))
      val cond = IO(Input(Bool()))
      val and, mux, bitsCat = IO(Output(UInt(8.W)))
      // Chisel unconditionally emits a node, so name it at least
      val x = a.viewAs[UInt] & b.viewAs[MyBundle].value
      and.viewAs[UInt] := x

      val y = Mux(cond.viewAs[Bool], a.viewAs[UInt], b.value.viewAs[UInt])
      mux.viewAs[UInt] := y

      // TODO should we have a macro so that we don't need .apply?
      val aBits = a.viewAs[UInt].apply(3, 0)
      val bBits = b.viewAs[MyBundle].value(3, 0)
      val abCat = aBits.viewAs[UInt] ## bBits.viewAs[UInt]
      bitsCat := abCat
    }
    val chirrtl = ChiselStage.emitChirrtl(new MyModule)
    val expected = List(
      "node x = and(a, b.value)",
      "and <= x",
      "node y = mux(cond, a, b.value)",
      "mux <= y",
      "node aBits = bits(a, 3, 0)",
      "node bBits = bits(b.value, 3, 0)",
      "node abCat = cat(aBits, bBits)",
      "bitsCat <= abCat"
    )
    for (line <- expected) {
      chirrtl should include(line)
    }
  }

  it should "support .asUInt of Views" in {
    import VecBundleDataView._
    class MyModule extends Module {
      val barIn = IO(Input(new MyBundle))
      val fooOut = IO(Output(UInt()))
      val cat = barIn.viewAs[Vec[UInt]].asUInt
      fooOut := cat
    }
    val chirrtl = ChiselStage.emitChirrtl(new MyModule)
    chirrtl should include ("node cat = cat(barIn.foo, barIn.bar)")
    chirrtl should include ("fooOut <= cat")
  }

  it should "be composable" in {
    // Given DataView[A, B] and DataView[B, C], derive DataView[A, C]
    class Foo(val foo: UInt) extends Bundle
    class Bar(val bar: UInt) extends Bundle
    class Fizz(val fizz: UInt) extends Bundle

    implicit val foo2bar = DataView[Foo, Bar](f => new Bar(chiselTypeClone(f.foo)), _.foo -> _.bar)
    implicit val bar2fizz = DataView[Bar, Fizz](b => new Fizz(chiselTypeClone(b.bar)), _.bar -> _.fizz)

    implicit val foo2fizz: DataView[Foo, Fizz] = foo2bar.andThen(bar2fizz)

    class MyModule extends Module {
      val a, b = IO(Input(new Foo(UInt(8.W))))
      val y, z = IO(Output(new Fizz(UInt(8.W))))
      y := a.viewAs[Fizz]
      z := b.viewAs[Bar].viewAs[Fizz]
    }
    val chirrtl = ChiselStage.emitChirrtl(new MyModule)
    chirrtl should include ("y.fizz <= a.foo")
    chirrtl should include ("z.fizz <= b.foo")
  }

  it should "enable using Seq like Data" in {
    class MyModule extends Module {
      val a, b, c, d = IO(Input(UInt(8.W)))
      val sel = IO(Input(Bool()))
      val y, z = IO(Output(UInt(8.W)))
      // Unclear why the implicit conversion does not work in this case for Seq
      // That being said, it's easy enough to cast via `.viewAs` with or without
      Seq(y, z) := Mux(sel, Seq(a, b).viewAs, Seq(c, d).viewAs[Vec[UInt]])
    }
    // Verilog instead of CHIRRTL because the optimizations make it much prettier
    val verilog = ChiselStage.emitVerilog(new MyModule)
    verilog should include ("assign y = sel ? a : c;")
    verilog should include ("assign z = sel ? b : d;")
  }

  // This example should be turned into a built-in feature
  it should "enable viewing Seqs as Vecs" in {

    class MyModule extends Module {
      val a, b, c = IO(Input(UInt(8.W)))
      val x, y, z = IO(Output(UInt(8.W)))
      Seq(x, y, z) := VecInit(a, b, c)
    }
    // Verilog instead of CHIRRTL because the optimizations make it much prettier
    val verilog = ChiselStage.emitVerilog(new MyModule)
    verilog should include ("assign x = a;")
    verilog should include ("assign y = b;")
    verilog should include ("assign z = c;")
  }

  it should "support recursive composition of views" in {
    class MyModule extends Module {
      val a, b, c, d = IO(Input(UInt(8.W)))
      val w, x, y, z = IO(Output(UInt(8.W)))

      // A little annoying that we need the type annotation on VecInit to get the implicit conversion to work
      // Note that one can just use the Seq on the RHS so there is an alternative (may lack discoverability)
      // We could also overload `VecInit` instead of relying on the implicit conversion
      Seq((w, x), (y, z)) := VecInit[HWTuple2[UInt, UInt]]((a, b), (c, d))
    }
    val verilog = ChiselStage.emitVerilog(new MyModule)
    verilog should include ("assign w = a;")
    verilog should include ("assign x = b;")
    verilog should include ("assign y = c;")
    verilog should include ("assign z = d;")
  }

  it should "support dynamic indexing for Vec identity views" in {
    class MyModule extends Module {
      val dataIn = IO(Input(UInt(8.W)))
      val addr = IO(Input(UInt(2.W)))
      val dataOut = IO(Output(UInt(8.W)))

      val vec = RegInit(0.U.asTypeOf(Vec(4, UInt(8.W))))
      val view = vec.viewAs[Vec[UInt]]
      // Dynamic indexing is more of a "generator" in Chisel3 than an individual node
      // This style is not recommended, this is just testing the behavior
      val selected = view(addr)
      selected := dataIn
      dataOut := selected
    }
    val chirrtl = ChiselStage.emitChirrtl(new MyModule)
    chirrtl should include ("vec[addr] <= dataIn")
    chirrtl should include ("dataOut <= vec[addr]")
  }

  it should "error if you try to dynamically index a Vec view that does not correspond to a Vec target" in {
    class MyModule extends Module {
      val inA, inB = IO(Input(UInt(8.W)))
      val outA, outB = IO(Output(UInt(8.W)))
      val idx = IO(Input(UInt(1.W)))

      val a, b, c, d = RegInit(0.U)

      // Dynamic indexing is more of a "generator" in Chisel3 than an individual node
      // This style is not recommended, this is just testing the behavior
      val selected = Seq((a, b), (c, d)).apply(idx)
      selected := (inA, inB)
      (outA, outB) := selected
    }
    (the [InvalidViewException] thrownBy {
      ChiselStage.emitChirrtl(new MyModule)
    }).getMessage should include ("Dynamic indexing of Views is not yet supported")
  }

  it should "error if the mapping is non-total in the view" in {
    class MyBundle(val foo: UInt, val bar: UInt) extends Bundle
    implicit val dv = DataView[UInt, MyBundle](_ => new MyBundle(UInt(), UInt()), _ -> _.bar)
    class MyModule extends Module {
      val tpe = new MyBundle(UInt(8.W), UInt(8.W))
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(tpe))
      out := in.viewAs[MyBundle]
    }
    val err = the [InvalidViewException] thrownBy (ChiselStage.emitVerilog(new MyModule))
    err.toString should include ("View field '_.foo' is missing")
  }

  it should "error if the mapping is non-total in the target" in {
    implicit val dv = DataView[(UInt, UInt), UInt](_ => UInt(), _._1 -> _)
    class MyModule extends Module {
      val a, b = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      out := (a, b).viewAs[UInt]
    }
    val err = the [InvalidViewException] thrownBy (ChiselStage.emitVerilog(new MyModule))
    err.toString should include ("Target field '_._2' is missing")
  }

  it should "error if the mapping contains Data that are not part of the Target" in {
    class BundleA extends Bundle {
      val foo = UInt(8.W)
    }
    class BundleB extends Bundle {
      val fizz = UInt(8.W)
      val buzz = UInt(8.W)
    }
    implicit val dv = DataView[BundleA, BundleB](_ => new BundleB, _.foo -> _.fizz, (_, b) => (3.U, b.buzz))
    class MyModule extends Module {
      val in = IO(Input(new BundleA))
      val out = IO(Output(new BundleB))
      out := in.viewAs[BundleB]
    }
    val err = the [InvalidViewException] thrownBy (ChiselStage.emitVerilog(new MyModule))
    err.toString should include ("View mapping must only contain Elements within the Target")
  }

  it should "error if the mapping contains Data that are not part of the View" in {
    class BundleA extends Bundle {
      val foo = UInt(8.W)
    }
    class BundleB extends Bundle {
      val fizz = UInt(8.W)
      val buzz = UInt(8.W)
    }
    implicit val dv = DataView[BundleA, BundleB](_ => new BundleB, _.foo -> _.fizz, (_, b) => (3.U, b.buzz))
    implicit val dv2 = dv.invert(_ => new BundleA)
    class MyModule extends Module {
      val in = IO(Input(new BundleA))
      val out = IO(Output(new BundleB))
      out.viewAs[BundleA] := in
    }
    val err = the [InvalidViewException] thrownBy (ChiselStage.emitVerilog(new MyModule))
    err.toString should include ("View mapping must only contain Elements within the View")
  }

  it should "error if a view has a width that does not match the target" in {
    class BundleA extends Bundle {
      val foo = UInt(8.W)
    }
    class BundleB extends Bundle {
      val bar = UInt(4.W)
    }
    implicit val dv = DataView[BundleA, BundleB](_ => new BundleB, _.foo -> _.bar)
    class MyModule extends Module {
      val in = IO(Input(new BundleA))
      val out = IO(Output(new BundleB))
      out := in.viewAs[BundleB]
    }
    val err = the [InvalidViewException] thrownBy ChiselStage.emitChirrtl(new MyModule)
    val expected = """View field _\.bar UInt<4> has width <4> that is incompatible with target value .+'s width <8>""".r
    err.getMessage should fullyMatch regex expected
  }

  it should "error if a view has a known width when the target width is unknown" in {
    class BundleA extends Bundle {
      val foo = UInt()
    }
    class BundleB extends Bundle {
      val bar = UInt(4.W)
    }
    implicit val dv = DataView[BundleA, BundleB](_ => new BundleB, _.foo -> _.bar)
    class MyModule extends Module {
      val in = IO(Input(new BundleA))
      val out = IO(Output(new BundleB))
      out := in.viewAs[BundleB]
    }
    val err = the [InvalidViewException] thrownBy ChiselStage.emitChirrtl(new MyModule)
    val expected = """View field _\.bar UInt<4> has width <4> that is incompatible with target value .+'s width <unknown>""".r
    err.getMessage should fullyMatch regex expected
  }

  behavior of "PartialDataView"

  it should "still error if the mapping is non-total in the view" in {
    class MyBundle(val foo: UInt, val bar: UInt) extends Bundle
    implicit val dv = PartialDataView[UInt, MyBundle](_ => new MyBundle(UInt(), UInt()), _ -> _.bar)
    class MyModule extends Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(new MyBundle(UInt(8.W), UInt(8.W))))
      out := in.viewAs[MyBundle]
    }
    val err = the [InvalidViewException] thrownBy (ChiselStage.emitVerilog(new MyModule))
    err.toString should include ("View field '_.foo' is missing")
  }

  it should "NOT error if the mapping is non-total in the target" in {
    implicit val dv = PartialDataView[(UInt, UInt), UInt](_ => UInt(), _._2 -> _)
    class MyModule extends Module {
      val a, b = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      out := (a, b).viewAs[UInt]
    }
    val verilog = ChiselStage.emitVerilog(new MyModule)
    verilog should include ("assign out = b;")
  }
}
