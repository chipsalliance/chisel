// See LICENSE for license details.

package chiselTests.experimental

import chisel3._
import chisel3.experimental.conversions._
import chisel3.experimental.dataview._
import chisel3.experimental.{Analog, HWTuple2}
import chisel3.probe._
import chisel3.reflect.DataMirror.internal.chiselTypeClone
import chisel3.util.{Decoupled, DecoupledIO, Valid, ValidIO}
import chiselTests.ChiselFlatSpec
import circt.stage.ChiselStage
import scala.collection.immutable.SeqMap

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

object ValidExtensions {
  implicit def view[T <: Data] = DataView[T, Valid[T]](
    x => Valid(x.cloneType), // Valid will strip direction with `Output(...)` anyway
    _ -> _.bits,
    (_, v) => true.B -> v.valid
  )
}

object DontCareDataView {
  class Foo extends Bundle {
    val a = UInt(8.W)
    val b = UInt(8.W)
  }
  implicit def view = DataView[UInt, Foo](
    _ => new Foo,
    _ -> _.a,
    (_, f) => DontCare -> f.b
  )
}

class DataViewSpec extends ChiselFlatSpec {

  behavior.of("DataView")

  it should "support simple Bundle viewing" in {
    import SimpleBundleDataView._
    class MyModule extends Module {
      val in = IO(Input(new BundleA(8)))
      val out = IO(Output(new BundleB(8)))
      out := in.viewAs[BundleB]
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect out.bar, in.foo")
  }

  it should "be a bidirectional mapping" in {
    import SimpleBundleDataView._
    class MyModule extends Module {
      val in = IO(Input(new BundleA(8)))
      val out = IO(Output(new BundleB(8)))
      out.viewAs[BundleA] := in
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect out.bar, in.foo")
  }

  it should "handle viewing UInts as UInts" in {
    class MyModule extends Module {
      val in = IO(Input(UInt(8.W)))
      val foo = IO(Output(UInt(8.W)))
      val bar = IO(Output(UInt(8.W)))
      foo := in.viewAs[UInt]
      bar.viewAs[UInt] := in
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect foo, in")
    chirrtl should include("connect bar, in")
  }

  it should "handle viewing Analogs as Analogs" in {
    class MyModule extends Module {
      val foo = IO(Analog(8.W))
      val bar = IO(Analog(8.W))
      foo <> bar.viewAs[Analog]
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("attach (foo, bar)")
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
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect fizz, in")
    chirrtl should include("connect buzz, in")
  }

  it should "handle viewing Vecs as their same concrete type" in {
    class MyModule extends Module {
      val in = IO(Input(Vec(1, UInt(8.W))))
      val fizz = IO(Output(Vec(1, UInt(8.W))))
      val buzz = IO(Output(Vec(1, UInt(8.W))))
      fizz := in.viewAs[Vec[UInt]]
      buzz.viewAs[Vec[UInt]] := in
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect fizz, in")
    chirrtl should include("connect buzz, in")
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
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect out[0], in.bar")
    chirrtl should include("connect out[1], in.foo")
    chirrtl should include("connect out2[0], in.bar")
    chirrtl should include("connect out2[1], in.foo")
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
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect deq.valid, enq.valid")
    chirrtl should include("connect enq.ready, deq.ready")
    chirrtl should include("connect deq.fizz, enq.bits.fizz")
    chirrtl should include("connect deq.buzz, enq.bits.buzz")
    chirrtl should include("connect deq2.valid, enq.valid")
    chirrtl should include("connect enq.ready, deq2.ready")
    chirrtl should include("connect deq2.fizz, enq.bits.fizz")
    chirrtl should include("connect deq2.buzz, enq.bits.buzz")
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
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect barOut.foo, fooIn.foo")
    chirrtl should include("connect fooOut.foo, barIn.foo")
  }

  it should "support viewing a Record as a Parent Record type" in {
    object Base {
      def elements = SeqMap("foo" -> UInt(8.W))
    }
    class Foo extends Record {
      override val elements = Base.elements
    }
    class Bar extends Foo {
      override val elements = Base.elements ++ SeqMap("bar" -> UInt(8.W))
    }

    class MyModule extends Module {
      val fooIn = IO(Input(new Foo))
      val barOut = IO(Output(new Bar))
      barOut.viewAsSupertype(new Foo) := fooIn

      val barIn = IO(Input(new Bar))
      val fooOut = IO(Output(new Foo))
      fooOut := barIn.viewAsSupertype(new Foo)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect barOut.foo, fooIn.foo")
    chirrtl should include("connect fooOut.foo, barIn.foo")
  }

  it should "fail if you try viewing a Record as a poorly inherited Parent Record type" in {
    class Foo extends Record {
      override val elements = SeqMap("foo" -> UInt(8.W), "quz" -> UInt(8.W))
    }
    class Bar extends Foo {
      // No actual relationship in the elements...
      override val elements = SeqMap("bar" -> UInt(8.W))
    }

    class MyModule extends Module {
      val fooIn = IO(Input(new Foo))
      val barOut = IO(Output(new Bar))
      barOut.viewAsSupertype(new Foo) := fooIn

      val barIn = IO(Input(new Bar))
      val fooOut = IO(Output(new Foo))
      fooOut := barIn.viewAsSupertype(new Foo)
    }
    (the[InvalidViewException] thrownBy {
      ChiselStage.emitCHIRRTL(new MyModule)
    }).getMessage should include("View fields '_.quz, _.foo' are missing.")
  }

  it should "support viewing a Record of the same Scala type as supertype for the purposes of mismatched directions" in {
    class Foo extends Record {
      override val elements = SeqMap("foo" -> Decoupled(UInt(8.W)))
    }

    class MyModule extends Module {
      val ifc = IO(Flipped(new Foo))
      val ifcMon = IO(Input(new Foo))

      ifc :<= DontCare
      ifcMon.viewAsSupertype(chiselTypeOf(ifc)) :>= ifc
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("invalidate ifc.foo.bits")
    chirrtl should include("connect ifc.foo.ready, ifcMon.foo.ready")
  }

  it should "be easy to make a PartialDataView viewing a Bundle as a Parent Bundle type" in {
    class Foo(x: Int) extends Bundle {
      val foo = UInt(x.W)
    }
    class Bar(val x: Int) extends Foo(x) {
      val bar = UInt(x.W)
    }
    implicit val view = PartialDataView.supertype[Bar, Foo](b => new Foo(b.x))
    class MyModule extends Module {
      val fooIn = IO(Input(new Foo(8)))
      val barOut = IO(Output(new Bar(8)))
      barOut.viewAs[Foo] := fooIn

      val barIn = IO(Input(new Bar(8)))
      val fooOut = IO(Output(new Foo(8)))
      fooOut := barIn.viewAs[Foo]
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect barOut.foo, fooIn.foo")
    chirrtl should include("connect fooOut.foo, barIn.foo")
  }

  it should "support viewing structural supertypes" in {
    class A extends Bundle {
      val x = UInt(3.W)
      val y = UInt(3.W)
      val z = UInt(3.W)
    }
    class B extends Bundle {
      val x = UInt(3.W)
      val y = UInt(3.W)
    }
    class C extends Bundle {
      val y = UInt(3.W)
      val z = UInt(3.W)
    }
    class MyModule extends Module {
      val io = IO(new Bundle {
        val a = Input(new A)
        val b = Output(new B)
        val c = Output(new C)
        val inc = Input(new C)
        val outa = Output(new A)
      })
      io.b <> io.a.viewAsSupertype(new B)
      io.c <> io.a.viewAsSupertype(new C)
      io.outa.viewAsSupertype(new C) <> io.inc
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect io.b.x, io.a.x")
    chirrtl should include("connect io.c.z, io.a.z")
    chirrtl should include("connect io.outa.z, io.inc.z")
    chirrtl should include("connect io.outa.y, io.inc.y")
  }

  it should "support viewing structural supertypes with bundles" in {
    class Foo extends Bundle {
      val y = UInt(3.W)
    }
    class A extends Bundle {
      val foo = new Foo
      val x = UInt(3.W)
      val z = UInt(3.W)
    }
    class B extends Bundle {
      val x = UInt(3.W)
      val foo = new Foo
    }
    class C extends Bundle {
      val foo = new Foo
      val z = UInt(3.W)
    }
    class MyModule extends Module {
      val io = IO(new Bundle {
        val a = Input(new A)
        val b = Output(new B)
        val c = Output(new C)
        val inc = Input(new C)
        val outa = Output(new A)
      })
      io.b <> io.a.viewAsSupertype(new B)
      io.c <> io.a.viewAsSupertype(new C)
      io.outa.viewAsSupertype(new C) <> io.inc
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect io.b.x, io.a.x")
    chirrtl should include("connect io.c.z, io.a.z")
    chirrtl should include("connect io.outa.foo, io.inc.foo")
    chirrtl should include("connect io.b.foo, io.a.foo")
    chirrtl should include("connect io.c.foo, io.a.foo")
  }

  it should "error during elaboration for sub-type errors that cannot be found at compile-time" in {
    class A extends Bundle {
      val x = UInt(3.W)
      val y = UInt(4.W)
      val z = UInt(3.W)
    }
    class B extends Bundle {
      val x = UInt(3.W)
      val y = UInt(3.W)
    }
    class MyModule extends Module {
      val io = IO(new Bundle {
        val a = Input(new A)
        val b = Output(new B)
      })
      io.b <> io.a.viewAsSupertype(new B)
    }
    a[ChiselException] should be thrownBy (ChiselStage.emitSystemVerilog(new MyModule))
  }

  it should "support viewing structural supertypes with generated types" in {
    class Foo extends Bundle {
      val foo = UInt(8.W)
    }
    class Bar extends Foo {
      val bar = UInt(8.W)
    }
    class MyInterface[T <: Data](gen: () => T) extends Bundle {
      val foo = gen()
      val fizz = UInt(8.W)
      val buzz = UInt(8.W)
    }
    class MyModule extends Module {
      val fooIf = IO(Input(new MyInterface(() => UInt(8.W))))
      val fooOut = IO(Output(new Foo))
      fooOut <> fooIf.viewAsSupertype(new Foo)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
  }

  it should "throw a type error for structural non-supertypes with different members" in {
    assertTypeError("""
      class A extends Bundle {
        val x = UInt(3.W)
        val y = UInt(3.W)
        val z = UInt(3.W)
      }
      class B extends Bundle {
        val x = UInt(3.W)
        val y = UInt(3.W)
      }
      class MyModule extends Module {
        val io = IO(new Bundle{
          val a = Input(new A)
          val b = Output(new B)
        })
        io.a <> io.b.viewAsSupertype(new A)
      }
      """)
  }

  it should "throw a type error for structural non-supertypes with different types" in {
    assertTypeError("""
      class A extends Bundle {
        val x = SInt(3.W)
        val y = UInt(3.W)
        val z = UInt(3.W)
      }
      class B extends Bundle {
        val x = UInt(3.W)
        val y = UInt(3.W)
      }
      class MyModule extends Module {
        val io = IO(new Bundle{
          val a = Input(new A)
          val b = Output(new B)
        })
        io.b <> io.a.viewAsSupertype(new B)
      }
      """)
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
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val expected = List(
      "node x = and(a, b.value)",
      "connect and, x",
      "node y = mux(cond, a, b.value)",
      "connect mux, y",
      "node aBits = bits(a, 3, 0)",
      "node bBits = bits(b.value, 3, 0)",
      "node abCat = cat(aBits, bBits)",
      "connect bitsCat, abCat"
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
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("node cat = cat(barIn.foo, barIn.bar)")
    chirrtl should include("connect fooOut, cat")
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
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect y.fizz, a.foo")
    chirrtl should include("connect z.fizz, b.foo")
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
    val verilog = ChiselStage.emitSystemVerilog(new MyModule)
    verilog should include("assign y = sel ? a : c;")
    verilog should include("assign z = sel ? b : d;")
  }

  // This example should be turned into a built-in feature
  it should "enable viewing Seqs as Vecs" in {

    class MyModule extends Module {
      val a, b, c = IO(Input(UInt(8.W)))
      val x, y, z = IO(Output(UInt(8.W)))
      Seq(x, y, z) := VecInit(a, b, c)
    }
    // Verilog instead of CHIRRTL because the optimizations make it much prettier
    val verilog = ChiselStage.emitSystemVerilog(new MyModule)
    verilog should include("assign x = a;")
    verilog should include("assign y = b;")
    verilog should include("assign z = c;")
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
    val verilog = ChiselStage.emitSystemVerilog(new MyModule)
    verilog should include("assign w = a;")
    verilog should include("assign x = b;")
    verilog should include("assign y = c;")
    verilog should include("assign z = d;")
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
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect vec[addr], dataIn")
    chirrtl should include("connect dataOut, vec[addr]")
  }

  it should "support dynamic indexing for Vecs that correspond 1:1 in a view" in {
    class MyBundle extends Bundle {
      val foo = Vec(4, UInt(8.W))
      val bar = UInt(2.W)
    }
    implicit val myView = DataView[(Vec[UInt], UInt), MyBundle](
      _ => new MyBundle,
      _._1 -> _.foo,
      _._2 -> _.bar
    )
    class MyModule extends Module {
      val dataIn = IO(Input(UInt(8.W)))
      val addr = IO(Input(UInt(2.W)))
      val dataOut = IO(Output(UInt(8.W)))

      val vec = RegInit(0.U.asTypeOf(Vec(4, UInt(8.W))))
      val addrReg = Reg(UInt(2.W))
      val view = (vec, addrReg).viewAs[MyBundle]
      // Dynamic indexing is more of a "generator" in Chisel3 than an individual node
      // This style is not recommended, this is just testing the behavior
      val selected = view.foo(view.bar)
      view.bar := addr
      selected := dataIn
      dataOut := selected
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("connect vec[addrReg], dataIn")
    chirrtl should include("connect dataOut, vec[addrReg]")
  }

  it should "allow views between reset types" in {
    class A extends Bundle {
      val bool = Bool()
      val asyncreset = AsyncReset()
    }

    class B extends Bundle {
      val reset_0 = Reset()
      val reset_1 = Reset()
    }

    class Foo extends RawModule {
      val a = Wire(new A)
      val b = Wire(new B)

      implicit val view = DataView[A, B](
        _ => new B,
        _.bool -> _.reset_0,
        _.asyncreset -> _.reset_1
      )

      a.viewAs[B] := b
    }

    (ChiselStage
      .emitCHIRRTL(new Foo, Array("--full-stacktrace"))
      .split('\n')
      .map(_.takeWhile(_ != '@'))
      .map(_.trim) should contain).allOf(
      "connect a.bool, b.reset_0",
      "connect a.asyncreset, b.reset_1"
    )
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
    (the[InvalidViewException] thrownBy {
      ChiselStage.emitCHIRRTL(new MyModule)
    }).getMessage should include("Dynamic indexing of Views is not yet supported")
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
    val err = the[InvalidViewException] thrownBy (ChiselStage.emitSystemVerilog(new MyModule))
    err.toString should include("View field '_.foo' is missing")
  }

  it should "error if the mapping is non-total in the target" in {
    implicit val dv = DataView[(UInt, UInt), UInt](_ => UInt(), _._1 -> _)
    class MyModule extends Module {
      val a, b = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      out := (a, b).viewAs[UInt]
    }
    val err = the[InvalidViewException] thrownBy (ChiselStage.emitSystemVerilog(new MyModule))
    err.toString should include("Target field '_._2' is missing")
  }

  it should "error if the mapping contains Data that are not part of the Target" in {
    class BundleA extends Bundle {
      val foo = UInt(8.W)
    }
    class BundleB extends Bundle {
      val fizz = UInt(8.W)
      val buzz = UInt(8.W)
    }
    class MyModule extends Module {
      val in = IO(Input(new BundleA))
      val out = IO(Output(new BundleB))
      val foo = Wire(UInt(8.W))
      implicit val dv = DataView[BundleA, BundleB](_ => new BundleB, _.foo -> _.fizz, (_, b) => (foo, b.buzz))
      out := in.viewAs[BundleB]
    }
    val err = the[InvalidViewException] thrownBy (ChiselStage.emitSystemVerilog(new MyModule))
    err.toString should include("View mapping must only contain Elements within the Target")
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
    val err = the[InvalidViewException] thrownBy (ChiselStage.emitSystemVerilog(new MyModule))
    err.toString should include("View mapping must only contain Elements within the View")
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
    val err = the[InvalidViewException] thrownBy ChiselStage.emitCHIRRTL(new MyModule)
    val expected = """View field _\.bar UInt<4> has width <4> that is incompatible with target value .+'s width <8>""".r
    (err.getMessage should fullyMatch).regex(expected)
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
    val err = the[InvalidViewException] thrownBy ChiselStage.emitCHIRRTL(new MyModule)
    val expected =
      """View field _\.bar UInt<4> has width <4> that is incompatible with target value .+'s width <unknown>""".r
    (err.getMessage should fullyMatch).regex(expected)
  }

  it should "support invalidation" in {
    class MyModule extends Module {
      val a, b, c, d, e, f = IO(Output(UInt(8.W)))
      val foo = (a, b).viewAs
      val bar = (c, d).viewAs
      val fizz = (e, f).viewAs
      foo := DontCare
      bar <> DontCare
      fizz._1 := DontCare
      fizz._2 <> DontCare
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val expected = ('a' to 'f').map(c => s"invalidate $c")
    for (line <- expected) {
      chirrtl should include(line)
    }
  }

  it should "error when calling .viewAs on non-hardware Data" in {
    class MyBundle(val foo: UInt, val bar: UInt) extends Bundle
    implicit val view =
      DataView[(UInt, UInt), MyBundle](x => new MyBundle(x._1.cloneType, x._2.cloneType), _._1 -> _.foo, _._2 -> _.bar)
    class MyModule extends Module {
      (UInt(8.W), UInt(8.W)).viewAs[MyBundle]
    }
    a[ChiselException] shouldBe thrownBy {
      ChiselStage.emitCHIRRTL(new MyModule, Array("--throw-on-first-error"))
    }
  }

  it should "handle viewing Probes as their referenced type" in {
    class MyModule extends Module {
      val a = IO(Output(Probe(Bool())))
      val w = WireInit(Bool(), false.B)
      val w_probe = ProbeValue(w)
      val w_probe_view = w_probe.viewAs[Bool]
      define(a, w_probe_view)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("define a = probe(w)")
  }

  it should "handle viewing RWProbes as their referenced type" in {
    class MyModule extends Module {
      val a = IO(Output(RWProbe(Bool())))
      val w = WireInit(Bool(), false.B)
      val w_probe = RWProbeValue(w)
      val w_probe_view = w_probe.viewAs[Bool]
      define(a, w_probe_view)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("define a = rwprobe(w)")
  }

  it should "error if attempting to define a viewed a Probe as a RWProbe" in {
    class MyModule extends Module {
      val a = IO(Output(RWProbe(Bool())))
      val w = WireInit(Bool(), false.B)
      val w_probe = ProbeValue(w)
      val w_probe_view = w_probe.viewAs[Bool]
      define(a, w_probe_view)
    }
    val err = the[ChiselException] thrownBy (ChiselStage.emitCHIRRTL(new MyModule, Array("--throw-on-first-error")))
    err.toString should include("Cannot use a non-writable probe expression to define a writable probe")
  }

  it should "error if attempting to connect a viewed a Probe to a connectable" in {
    class MyModule extends Module {
      val a = IO(Output(Bool()))
      val w = WireInit(Bool(), false.B)
      val w_probe = ProbeValue(w)
      val w_probe_view = w_probe.viewAs[Bool]
      a := w_probe_view
    }
    val err = the[ChiselException] thrownBy (ChiselStage.emitCHIRRTL(new MyModule, Array("--throw-on-first-error")))
    err.toString should include("Probed type cannot participate in a mono connection")
  }

  it should "support literals as part of the target" in {
    import ValidExtensions._
    class MyModule extends Module {
      val in0, in1, in2, in3, in4 = IO(Input(UInt(8.W)))
      val out0, out1, out2, out3, out4 = IO(Output(Valid(UInt(8.W))))
      out0 := in0.viewAs[Valid[UInt]]
      out1 <> in1.viewAs[Valid[UInt]]
      out2 :<>= in2.viewAs[Valid[UInt]]
      out3 :<= in3.viewAs[Valid[UInt]]
      out4 :#= in4.viewAs[Valid[UInt]]
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    for (i <- 0 until 5) {
      chirrtl should include(s"connect out$i.bits, in$i")
      chirrtl should include(s"connect out$i.valid, UInt<1>(0h1)")
    }
  }

  it should "error if a literal in a target would be driven" in {
    import ValidExtensions._
    class MyModule(op: (Valid[UInt], Valid[UInt]) => Unit) extends Module {
      val in = IO(Input(Valid(UInt(8.W))))
      val out = IO(Output(UInt(8.W)))
      op(out.viewAs[Valid[UInt]], in)
    }

    val ops = Seq[(Valid[UInt], Valid[UInt]) => Unit](
      _ := _,
      _ <> _,
      _ :<>= _,
      _ :<= _,
      _ :#= _
    )

    for (op <- ops) {
      a[ChiselException] shouldBe thrownBy {
        ChiselStage.emitCHIRRTL(new MyModule(op), Array("--throw-on-first-error"))
      }
    }
  }

  it should "error if a literal is used as part of the view" in {
    implicit val dv =
      DataView.mapping[Valid[UInt], UInt](_.bits.cloneType, (v, x) => Seq(v.bits -> x, v.valid -> true.B))
    class MyModule extends Module {
      val in = IO(Input(Valid(UInt(8.W))))
      val out = IO(Output(UInt(8.W)))
      out := in.viewAs[UInt]
    }
    val e = the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(new MyModule, Array("--throw-on-first-error"))
    }
    e.getMessage should include("View mapping must only contain Elements within the View")
  }

  it should "support DontCare as part of the target" in {
    import DontCareDataView._
    class MyModule extends Module {
      val in0, in1, in2, in3, in4 = IO(Input(UInt(8.W)))
      val out0, out1, out2, out3, out4 = IO(Output(new Foo))
      out0 := in0.viewAs[Foo]
      out1 <> in1.viewAs[Foo]
      out2 :<>= in2.viewAs[Foo]
      out3 :<= in3.viewAs[Foo]
      out4 :#= in4.viewAs[Foo]
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule, Array("--full-stacktrace"))
    for (i <- 0 until 5) {
      chirrtl should include(s"connect out$i.a, in$i")
      chirrtl should include(s"invalidate out$i.b")
    }
  }

  it should "error if a DontCare in a target would be driven" in {
    import DontCareDataView._
    class MyModule(op: (Foo, Foo) => Unit) extends Module {
      val in = IO(Input(new Foo))
      val out = IO(Output(UInt(8.W)))
      op(out.viewAs[Foo], in)
    }

    val ops = Seq[(Foo, Foo) => Unit](
      _ := _,
      _ :<>= _,
      _ :<= _,
      _ :#= _
    )

    for (op <- ops) {
      a[ChiselException] shouldBe thrownBy {
        ChiselStage.emitCHIRRTL(new MyModule(op), Array("--throw-on-first-error"))
      }
    }
    // <> does magical things with DontCare, including invalidating inputs!
    // But the behavior matches DontCare <> in.b
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule(_ <> _))
    chirrtl should include("invalidate in.b")
    chirrtl should include("connect out, in.a")
  }

  it should "error if DontCare is used as part of the view" in {
    implicit val dv =
      DataView.mapping[Valid[UInt], UInt](_.bits.cloneType, (v, x) => Seq(v.bits -> x, v.valid -> DontCare))
    class MyModule extends Module {
      val in = IO(Input(Valid(UInt(8.W))))
      val out = IO(Output(UInt(8.W)))
      out := in.viewAs[UInt]
    }
    val e = the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(new MyModule, Array("--throw-on-first-error"))
    }
    e.getMessage should include("View mapping must only contain Elements within the View")
  }

  it should "use the target in error messages involving the view" in {
    val e1 = the[ChiselException] thrownBy (
      ChiselStage.emitCHIRRTL(
        new RawModule {
          override def desiredName = "Top"
          val in = IO(Input(UInt(8.W)))
          val out = IO(Output(SInt(8.W)))
          out.viewAs[SInt] := in
        },
        Array("--throw-on-first-error")
      )
    )
    e1.getMessage should include(
      "Connection between sink (Top.out: IO[SInt<8>]) and source (Top.in: IO[UInt<8>]) failed"
    )

    val e2 = the[ChiselException] thrownBy (
      ChiselStage.emitCHIRRTL(
        new RawModule {
          override def desiredName = "Top"
          val foo, bar = IO(Input(UInt(8.W)))
          val fizz, buzz = IO(Output(SInt(8.W)))
          // Unfortunately, := does a bad job when the view doesn't map to a single Data, but at least connectables do a good job
          ((fizz, buzz).viewAs: Data) :#= ((foo, bar).viewAs: Data)
        },
        Array("--throw-on-first-error")
      )
    )
    e2.getMessage should include(
      "Connection between sink (Top.buzz: IO[SInt<8>]) and source (Top.bar: IO[UInt<8>]) failed"
    )
  }

  behavior.of("PartialDataView")

  it should "still error if the mapping is non-total in the view" in {
    class MyBundle(val foo: UInt, val bar: UInt) extends Bundle
    implicit val dv = PartialDataView[UInt, MyBundle](_ => new MyBundle(UInt(), UInt()), _ -> _.bar)
    class MyModule extends Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(new MyBundle(UInt(8.W), UInt(8.W))))
      out := in.viewAs[MyBundle]
    }
    val err = the[InvalidViewException] thrownBy (ChiselStage.emitSystemVerilog(new MyModule))
    err.toString should include("View field '_.foo' is missing")
  }

  it should "NOT error if the mapping is non-total in the target" in {
    implicit val dv = PartialDataView[(UInt, UInt), UInt](_ => UInt(), _._2 -> _)
    class MyModule extends Module {
      val a, b = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      out := (a, b).viewAs[UInt]
    }
    val verilog = ChiselStage.emitSystemVerilog(new MyModule)
    verilog should include("assign out = b;")
  }
}
