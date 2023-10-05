// SPDX-License-Identifier: Apache-2.0

package chiselTests
package experimental.hierarchy

import chisel3._
import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance}
import chisel3.util.{DecoupledIO, Valid}

// TODO/Notes
// - In backport, clock/reset are not automatically assigned. I think this is fixed in 3.5
// - CircuitTarget for annotations on the definition are wrong - needs to be fixed.
class InstanceSpec extends ChiselFunSpec with Utils {
  import Annotations._
  import Examples._
  describe("(0) Instance instantiation") {
    it("(0.a): name of an instance should be correct") {
      class Top extends Module {
        val definition = Definition(new AddOne)
        val i0 = Instance(definition)
      }
      val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("inst i0 of AddOne")
    }
    it("(0.b): name of an instanceclone should not error") {
      class Top extends Module {
        val definition = Definition(new AddTwo)
        val i0 = Instance(definition)
        val i = i0.i0 // This should not error
      }
      val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("inst i0 of AddTwo")
    }
    it("(0.c): accessing internal fields through non-generated means is hard to do") {
      class Top extends Module {
        val definition = Definition(new AddOne)
        val i0 = Instance(definition)
        //i0.lookup(_.in) // Uncommenting this line will give the following error:
        //"You are trying to access a macro-only API. Please use the @public annotation instead."
        i0.in
      }
      val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("inst i0 of AddOne")
    }
    it("(0.d): BlackBoxes should be supported") {
      class Top extends Module {
        val in = IO(Input(UInt(32.W)))
        val out = IO(Output(UInt(32.W)))
        val io = IO(new Bundle {
          val in = Input(UInt(32.W))
          val out = Output(UInt(32.W))
        })
        val definition = Definition(new AddOneBlackBox)
        val i0 = Instance(definition)
        val i1 = Instance(definition)
        i0.io.in := in
        out := i0.io.out
        io <> i1.io
      }
      val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("inst i0 of AddOneBlackBox")
      chirrtl should include("inst i1 of AddOneBlackBox")
      chirrtl should include("connect i0.in, in")
      chirrtl should include("connect out, i0.out")
      chirrtl should include("connect i1.in, io.in")
      chirrtl should include("connect io.out, i1.out")
    }
    it("(0.e): Instances with Bundles with unsanitary names should be supported") {
      class Top extends Module {
        val definition = Definition(new HasUnsanitaryBundleField)
        val i0 = Instance(definition)
        i0.in := 123.U
      }
      val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("connect i0.realIn.aminusx, UInt<7>(0h7b)")
    }
  }
  describe("(1) Annotations on instances in same chisel compilation") {
    it("(1.a): should work on a single instance, annotating the instance") {
      class Top extends Module {
        val definition: Definition[AddOne] = Definition(new AddOne)
        val i0:         Instance[AddOne] = Instance(definition)
        mark(i0, "i0")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(MarkAnnotation("~Top|Top/i0:AddOne".it, "i0"))
    }
    it("(1.b): should work on a single instance, annotating an inner wire") {
      class Top extends Module {
        val definition: Definition[AddOne] = Definition(new AddOne)
        val i0:         Instance[AddOne] = Instance(definition)
        mark(i0.innerWire, "i0.innerWire")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i0:AddOne>innerWire".rt, "i0.innerWire")
      )
    }
    it("(1.c): should work on a two nested instances, annotating the instance") {
      class Top extends Module {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        val i0:         Instance[AddTwo] = Instance(definition)
        mark(i0.i0, "i0.i0")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i0:AddTwo/i0:AddOne".it, "i0.i0")
      )
    }
    it("(1.d): should work on a two nested instances, annotating the inner wire") {
      class Top extends Module {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        val i0:         Instance[AddTwo] = Instance(definition)
        mark(i0.i0.innerWire, "i0.i0.innerWire")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i0:AddTwo/i0:AddOne>innerWire".rt, "i0.i0.innerWire")
      )
    }
    it("(1.e): should work on a nested module in an instance, annotating the module") {
      class Top extends Module {
        val definition: Definition[AddTwoMixedModules] = Definition(new AddTwoMixedModules)
        val i0:         Instance[AddTwoMixedModules] = Instance(definition)
        mark(i0.i1, "i0.i1")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i0:AddTwoMixedModules/i1:AddOne_1".it, "i0.i1")
      )
    }
    it("(1.f): should work on an instantiable container, annotating a wire") {
      class Top extends Module {
        val definition: Definition[AddOneWithInstantiableWire] = Definition(new AddOneWithInstantiableWire)
        val i0:         Instance[AddOneWithInstantiableWire] = Instance(definition)
        mark(i0.wireContainer.innerWire, "i0.innerWire")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i0:AddOneWithInstantiableWire>innerWire".rt, "i0.innerWire")
      )
    }
    it("(1.g): should work on an instantiable container, annotating a module") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableModule)
        val i0 = Instance(definition)
        mark(i0.moduleContainer.i0, "i0.i0")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i0:AddOneWithInstantiableModule/i0:AddOne".it, "i0.i0")
      )
    }
    it("(1.h): should work on an instantiable container, annotating an instance") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableInstance)
        val i0 = Instance(definition)
        mark(i0.instanceContainer.i0, "i0.i0")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i0:AddOneWithInstantiableInstance/i0:AddOne".it, "i0.i0")
      )
    }
    it("(1.i): should work on an instantiable container, annotating an instantiable container's module") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableInstantiable)
        val i0 = Instance(definition)
        mark(i0.containerContainer.container.i0, "i0.i0")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i0:AddOneWithInstantiableInstantiable/i0:AddOne".it, "i0.i0")
      )
    }
    it("(1.j): should work on public member which references public member of another instance") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableInstantiable)
        val i0 = Instance(definition)
        mark(i0.containerContainer.container.i0, "i0.i0")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i0:AddOneWithInstantiableInstantiable/i0:AddOne".it, "i0.i0")
      )
    }
    it("(1.k): should work for targets on definition to have correct circuit name") {
      class Top extends Module {
        val definition = Definition(new AddOneWithAnnotation)
        val i0 = Instance(definition)
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|AddOneWithAnnotation>innerWire".rt, "innerWire")
      )
    }
    it("(1.l): should work on things with type parameters") {
      class Top extends Module {
        val definition = Definition(new HasTypeParams[UInt](UInt(3.W)))
        val i0 = Instance(definition)
        mark(i0.blah, "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i0:HasTypeParams>blah".rt, "blah")
      )
    }
  }
  describe("(2) Annotations on designs not in the same chisel compilation") {
    it("(2.a): should work on an innerWire, marked in a different compilation") {
      val first = elaborateAndGetModule(new AddTwo)
      class Top(x: AddTwo) extends Module {
        val parent = Instance(Definition(new ViewerParent(x, false, true)))
      }
      val (_, annos) = getFirrtlAndAnnos(new Top(first))
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~AddTwo|AddTwo/i0:AddOne>innerWire".rt, "first")
      )
    }
    it("(2.b): should work on an innerWire, marked in a different compilation, in instanced instantiable") {
      val first = elaborateAndGetModule(new AddTwo)
      class Top(x: AddTwo) extends Module {
        val parent = Instance(Definition(new ViewerParent(x, true, false)))
      }
      val (_, annos) = getFirrtlAndAnnos(new Top(first))
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~AddTwo|AddTwo/i0:AddOne>innerWire".rt, "second")
      )
    }
    it("(2.c): should work on an innerWire, marked in a different compilation, in instanced module") {
      val first = elaborateAndGetModule(new AddTwo)
      class Top(x: AddTwo) extends Module {
        val d = Definition(new ViewerParent(x, false, false))
        val parent = Instance(d)
        mark(parent.viewer.x.i0.innerWire, "third")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top(first))
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~AddTwo|AddTwo/i0:AddOne>innerWire".rt, "third")
      )
    }
  }
  describe("(3) @public") {
    it("(3.a): should work on multi-vals") {
      class Top() extends Module {
        val mv = Instance(Definition(new MultiVal()))
        mark(mv.x, "mv.x")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(MarkAnnotation("~Top|Top/mv:MultiVal>x".rt, "mv.x"))
    }
    it("(3.b): should work on lazy vals") {
      class Top() extends Module {
        val lv = Instance(Definition(new LazyVal()))
        mark(lv.x, lv.y)
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(MarkAnnotation("~Top|Top/lv:LazyVal>x".rt, "Hi"))
    }
    it("(3.c): should work on islookupables") {
      class Top() extends Module {
        val p = Parameters("hi", 0)
        val up = Instance(Definition(new UsesParameters(p)))
        mark(up.x, up.y.string + up.y.int)
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/up:UsesParameters>x".rt, "hi0")
      )
    }
    it("(3.d): should work on lists") {
      class Top() extends Module {
        val i = Instance(Definition(new HasList()))
        mark(i.x(1), i.y(1).toString)
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(MarkAnnotation("~Top|Top/i:HasList>x_1".rt, "2"))
    }
    it("(3.e): should work on seqs") {
      class Top() extends Module {
        val i = Instance(Definition(new HasSeq()))
        mark(i.x(1), i.y(1).toString)
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(MarkAnnotation("~Top|Top/i:HasSeq>x_1".rt, "2"))
    }
    it("(3.f): should work on options") {
      class Top() extends Module {
        val i = Instance(Definition(new HasOption()))
        i.x.map(x => mark(x, "x"))
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(MarkAnnotation("~Top|Top/i:HasOption>x".rt, "x"))
    }
    it("(3.g): should work on vecs") {
      class Top() extends Module {
        val i = Instance(Definition(new HasVec()))
        mark(i.x, "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(MarkAnnotation("~Top|Top/i:HasVec>x".rt, "blah"))
    }
    it("(3.h): should work on statically indexed vectors external to module") {
      class Top() extends Module {
        val i = Instance(Definition(new HasVec()))
        mark(i.x(1), "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(MarkAnnotation("~Top|Top/i:HasVec>x[1]".rt, "blah"))
    }
    it("(3.i): should work on statically indexed vectors internal to module") {
      class Top() extends Module {
        val i = Instance(Definition(new HasIndexedVec()))
        mark(i.y, "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i:HasIndexedVec>x[1]".rt, "blah")
      )
    }
    it("(3.j): should work on accessed subfields of aggregate ports") {
      class Top extends Module {
        val input = IO(Input(Valid(UInt(8.W))))
        val i = Instance(Definition(new HasSubFieldAccess))
        i.valid := input.valid
        i.bits := input.bits
        mark(i.valid, "valid")
        mark(i.bits, "bits")
      }
      val expected = List(
        "~Top|Top/i:HasSubFieldAccess>in.valid".rt -> "valid",
        "~Top|Top/i:HasSubFieldAccess>in.bits".rt -> "bits"
      )
      val lines = List(
        "connect i.in.valid, input.valid",
        "connect i.in.bits, input.bits"
      )
      val (chirrtl, annos) = getFirrtlAndAnnos(new Top)
      val text = chirrtl.serialize
      for (line <- lines) {
        text should include(line)
      }
      for (e <- expected.map(MarkAnnotation.tupled)) {
        annos should contain(e)
      }
    }
    ignore("(3.k): should work on vals in constructor arguments") {
      class Top() extends Module {
        val i = Instance(Definition(new HasPublicConstructorArgs(10)))
        //mark(i.x, i.int.toString)
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i:HasPublicConstructorArgs>x".rt, "10")
      )
    }
    it("(3.l): should work on eithers") {
      class Top() extends Module {
        val i = Instance(Definition(new HasEither()))
        i.x.map(x => mark(x, "xright")).left.map(x => mark(x, "xleft"))
        i.y.map(x => mark(x, "yright")).left.map(x => mark(x, "yleft"))
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i:HasEither>x".rt, "xright")
      )
      annos.collect { case c: MarkAnnotation => c } should contain(MarkAnnotation("~Top|Top/i:HasEither>y".rt, "yleft"))
    }
    it("(3.m): should work on tuple2") {
      class Top() extends Module {
        val i = Instance(Definition(new HasTuple2()))
        mark(i.xy._1, "x")
        mark(i.xy._2, "y")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(MarkAnnotation("~Top|Top/i:HasTuple2>x".rt, "x"))
      annos.collect { case c: MarkAnnotation => c } should contain(MarkAnnotation("~Top|Top/i:HasTuple2>y".rt, "y"))
    }

    it("(3.n): should properly support val modifiers") {
      class SupClass extends Module {
        val value = 10
        val overriddenVal = 10
      }
      trait SupTrait {
        def x: Int
        def y: Int
      }
      @instantiable class SubClass() extends SupClass with SupTrait {
        // This errors
        //@public private val privateVal = 10
        // This errors
        //@public protected val protectedVal = 10
        @public override val overriddenVal = 12
        @public final val finalVal = 12
        @public lazy val lazyValue = 12
        @public val value = value
        @public final override lazy val x: Int = 3
        @public override final lazy val y: Int = 4
      }
    }
    it("(3.o): should work with Mems/SyncReadMems") {
      class Top() extends Module {
        val i = Instance(Definition(new HasMems()))
        mark(i.mem, "Mem")
        mark(i.syncReadMem, "SyncReadMem")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(MarkAnnotation("~Top|Top/i:HasMems>mem".rt, "Mem"))
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i:HasMems>syncReadMem".rt, "SyncReadMem")
      )
    }
    it("(3.p): should make connectable IOs on nested IsInstantiables that have IO Datas in them") {
      val (chirrtl, _) = getFirrtlAndAnnos(new AddTwoNestedInstantiableData(4))
      exactly(3, chirrtl.serialize.split('\n')) should include("connect i1.in, i0.out")
    }
    it(
      "(3.q): should make connectable IOs on nested IsInstantiables's Data when the Instance and Definition do not have the same parent"
    ) {
      val (chirrtl, _) = getFirrtlAndAnnos(new AddTwoNestedInstantiableDataWrapper(4))
      exactly(3, chirrtl.serialize.split('\n')) should include("connect i1.in, i0.out")
    }
  }
  describe("(4) toInstance") {
    it("(4.a): should work on modules") {
      class Top() extends Module {
        val i = Module(new AddOne())
        f(i.toInstance)
      }
      def f(i: Instance[AddOne]): Unit = mark(i.innerWire, "blah")
      val (_, annos) = getFirrtlAndAnnos(new Top)
      //TODO: Should this be ~Top|Top/i:AddOne>innerWire ???
      annos.collect { case c: MarkAnnotation => c } should contain(MarkAnnotation("~Top|AddOne>innerWire".rt, "blah"))
    }
    it("(4.b): should work on IsInstantiable") {
      class Top() extends Module {
        val i = Module(new AddTwo())
        val v = new Viewer(i, false)
        mark(f(v.toInstance), "blah")
      }
      def f(i: Instance[Viewer]): Data = i.x.i0.innerWire
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|AddTwo/i0:AddOne>innerWire".rt, "blah")
      )
    }
    it("(4.c): should work on seqs of modules") {
      class Top() extends Module {
        val is = Seq(Module(new AddTwo()), Module(new AddTwo())).map(_.toInstance)
        mark(f(is), "blah")
      }
      def f(i: Seq[Instance[AddTwo]]): Data = i.head.i0.innerWire
      val (c, annos) = getFirrtlAndAnnos(new Top)
      println(c.serialize)
      //TODO: Should this be ~Top|Top... ??
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|AddTwo/i0:AddOne>innerWire".rt, "blah")
      )
    }
    it("(4.d): should work on seqs of IsInstantiable") {
      class Top() extends Module {
        val i = Module(new AddTwo())
        val vs = Seq(new Viewer(i, false), new Viewer(i, false)).map(_.toInstance)
        mark(f(vs), "blah")
      }
      def f(i: Seq[Instance[Viewer]]): Data = i.head.x.i0.innerWire
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|AddTwo/i0:AddOne>innerWire".rt, "blah")
      )
    }
    it("(4.e): should work on options of modules") {
      class Top() extends Module {
        val is: Option[Instance[AddTwo]] = Some(Module(new AddTwo())).map(_.toInstance)
        mark(f(is), "blah")
      }
      def f(i: Option[Instance[AddTwo]]): Data = i.get.i0.innerWire
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|AddTwo/i0:AddOne>innerWire".rt, "blah")
      )
    }
  }
  describe("(5) Absolute Targets should work as expected") {
    it("(5.a): toAbsoluteTarget on a port of an instance") {
      class Top() extends Module {
        val i = Instance(Definition(new AddTwo()))
        amark(i.in, "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(MarkAnnotation("~Top|Top/i:AddTwo>in".rt, "blah"))
    }
    it("(5.b): toAbsoluteTarget on a subinstance's data within an instance") {
      class Top() extends Module {
        val i = Instance(Definition(new AddTwo()))
        amark(i.i0.innerWire, "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i:AddTwo/i0:AddOne>innerWire".rt, "blah")
      )
    }
    it("(5.c): toAbsoluteTarget on a submodule's data within an instance") {
      class Top() extends Module {
        val i = Instance(Definition(new AddTwoMixedModules()))
        amark(i.i1.in, "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i:AddTwoMixedModules/i1:AddOne_1>in".rt, "blah")
      )
    }
    it("(5.d): toAbsoluteTarget on a submodule's data, in an aggregate, within an instance") {
      class Top() extends Module {
        val i = Instance(Definition(new InstantiatesHasVec()))
        amark(i.i1.x.head, "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i:InstantiatesHasVec/i1:HasVec_1>x[0]".rt, "blah")
      )
    }
    it("(5.e): toAbsoluteTarget on a submodule's data, in an aggregate, within an instance, ILit") {
      class MyBundle extends Bundle { val x = UInt(3.W) }
      @instantiable
      class HasVec() extends Module {
        @public val x = Wire(Vec(3, new MyBundle()))
      }
      @instantiable
      class InstantiatesHasVec() extends Module {
        @public val i0 = Instance(Definition(new HasVec()))
        @public val i1 = Module(new HasVec())
      }
      class Top() extends Module {
        val i = Instance(Definition(new InstantiatesHasVec()))
        amark(i.i1.x.head.x, "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i:InstantiatesHasVec/i1:HasVec_1>x[0].x".rt, "blah")
      )
    }
    it("(5.f): toAbsoluteTarget on a subinstance") {
      class Top() extends Module {
        val i = Instance(Definition(new AddTwo()))
        amark(i.i1, "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|Top/i:AddTwo/i1:AddOne".it, "blah")
      )
    }
    it("(5.g): should work for absolute targets on definition to have correct circuit name") {
      class Top extends Module {
        val definition = Definition(new AddOneWithAbsoluteAnnotation)
        val i0 = Instance(definition)
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos.collect { case c: MarkAnnotation => c } should contain(
        MarkAnnotation("~Top|AddOneWithAbsoluteAnnotation>innerWire".rt, "innerWire")
      )
    }
  }
  describe("(6) @instantiable traits should work as expected") {
    class MyBundle extends Bundle {
      val in = Input(UInt(8.W))
      val out = Output(UInt(8.W))
    }
    @instantiable
    trait ModuleIntf extends BaseModule {
      @public val io = IO(new MyBundle)
    }
    @instantiable
    class ModuleWithCommonIntf(suffix: String = "") extends Module with ModuleIntf {
      override def desiredName: String = super.desiredName + suffix
      @public val sum = io.in + 1.U

      io.out := sum
    }
    class BlackBoxWithCommonIntf extends BlackBox with ModuleIntf

    it("(6.a): A Module that implements an @instantiable trait should be instantiable as that trait") {
      class Top extends Module {
        val i: Instance[ModuleIntf] = Instance(Definition(new ModuleWithCommonIntf))
        mark(i.io.in, "gotcha")
        mark(i, "inst")
      }
      val expected = List(
        "~Top|Top/i:ModuleWithCommonIntf>io.in".rt -> "gotcha",
        "~Top|Top/i:ModuleWithCommonIntf".it -> "inst"
      )
      val (chirrtl, annos) = getFirrtlAndAnnos(new Top)
      for (e <- expected.map(MarkAnnotation.tupled)) {
        annos should contain(e)
      }
    }
    it(
      "(6.b): An @instantiable Module that implements an @instantiable trait should be able to use extension methods from both"
    ) {
      class Top extends Module {
        val i: Instance[ModuleWithCommonIntf] = Instance(Definition(new ModuleWithCommonIntf))
        mark(i.io.in, "gotcha")
        mark(i.sum, "also this")
        mark(i, "inst")
      }
      val expected = List(
        "~Top|Top/i:ModuleWithCommonIntf>io.in".rt -> "gotcha",
        "~Top|Top/i:ModuleWithCommonIntf>sum".rt -> "also this",
        "~Top|Top/i:ModuleWithCommonIntf".it -> "inst"
      )
      val (chirrtl, annos) = getFirrtlAndAnnos(new Top)
      for (e <- expected.map(MarkAnnotation.tupled)) {
        annos should contain(e)
      }
    }
    it("(6.c): A BlackBox that implements an @instantiable trait should be instantiable as that trait") {
      class Top extends Module {
        val i: Instance[ModuleIntf] = Module(new BlackBoxWithCommonIntf).toInstance
        mark(i.io.in, "gotcha")
        mark(i, "module")
      }
      val expected = List(
        "~Top|BlackBoxWithCommonIntf>in".rt -> "gotcha",
        "~Top|BlackBoxWithCommonIntf".mt -> "module"
      )
      val (chirrtl, annos) = getFirrtlAndAnnos(new Top)
      for (e <- expected.map(MarkAnnotation.tupled)) {
        annos should contain(e)
      }
    }
    it("(6.d): It should be possible to have Vectors of @instantiable traits mixing concrete subclasses") {
      class Top extends Module {
        val proto = Definition(new ModuleWithCommonIntf("X"))
        val insts: Seq[Instance[ModuleIntf]] = Vector(
          Module(new ModuleWithCommonIntf("Y")).toInstance,
          Module(new BlackBoxWithCommonIntf).toInstance,
          Instance(proto)
        )
        mark(insts(0).io.in, "foo")
        mark(insts(1).io.in, "bar")
        mark(insts(2).io.in, "fizz")
      }
      val expected = List(
        "~Top|ModuleWithCommonIntfY>io.in".rt -> "foo",
        "~Top|BlackBoxWithCommonIntf>in".rt -> "bar",
        "~Top|Top/insts_2:ModuleWithCommonIntfX>io.in".rt -> "fizz"
      )
      val (chirrtl, annos) = getFirrtlAndAnnos(new Top)
      for (e <- expected.map(MarkAnnotation.tupled)) {
        annos should contain(e)
      }
    }
  }
  // TODO don't forget to test this with heterogeneous Views (eg. viewing a tuple of a port and non-port as a single Bundle)
  describe("(7) @instantiable and @public should compose with DataView") {
    import chisel3.experimental.dataview._
    it("(7.a): should work on simple Views") {
      @instantiable
      class MyModule extends RawModule {
        val in = IO(Input(UInt(8.W)))
        @public val out = IO(Output(UInt(8.W)))
        val sum = in + 1.U
        out := sum + 1.U
        @public val foo = in.viewAs[UInt]
        @public val bar = sum.viewAs[UInt]
      }
      class Top extends RawModule {
        val foo = IO(Input(UInt(8.W)))
        val bar = IO(Output(UInt(8.W)))
        val i = Instance(Definition(new MyModule))
        i.foo := foo
        bar := i.out
        mark(i.out, "out")
        mark(i.foo, "foo")
        mark(i.bar, "bar")
      }
      val expectedAnnos = List(
        "~Top|Top/i:MyModule>out".rt -> "out",
        "~Top|Top/i:MyModule>in".rt -> "foo",
        "~Top|Top/i:MyModule>sum".rt -> "bar"
      )
      val expectedLines = List(
        "connect i.in, foo",
        "connect bar, i.out"
      )
      val (chirrtl, annos) = getFirrtlAndAnnos(new Top)
      val text = chirrtl.serialize
      for (line <- expectedLines) {
        text should include(line)
      }
      for (e <- expectedAnnos.map(MarkAnnotation.tupled)) {
        annos should contain(e)
      }
    }

    ignore("(7.b): should work on Aggregate Views") {
      import chiselTests.experimental.FlatDecoupledDataView._
      type RegDecoupled = DecoupledIO[FizzBuzz]
      @instantiable
      class MyModule extends RawModule {
        private val a = IO(Flipped(new FlatDecoupled))
        private val b = IO(new FlatDecoupled)
        @public val enq = a.viewAs[RegDecoupled]
        @public val deq = b.viewAs[RegDecoupled]
        @public val enq_valid = enq.valid // Also return a subset of the view
        deq <> enq
      }
      class Top extends RawModule {
        val foo = IO(Flipped(new RegDecoupled(new FizzBuzz)))
        val bar = IO(new RegDecoupled(new FizzBuzz))
        val i = Instance(Definition(new MyModule))
        i.enq <> foo
        i.enq_valid := foo.valid // Make sure connections also work for @public on elements of a larger Aggregate
        i.deq.ready := bar.ready
        bar.valid := i.deq.valid
        bar.bits := i.deq.bits
        mark(i.enq, "enq")
        mark(i.enq.bits, "enq.bits")
        mark(i.deq.bits.fizz, "deq.bits.fizz")
        mark(i.enq_valid, "enq_valid")
      }
      val expectedAnnos = List(
        "~Top|Top/i:MyModule>a".rt -> "enq", // Not split, checks 1:1
        "~Top|Top/i:MyModule>a.fizz".rt -> "enq.bits", // Split, checks non-1:1 inner Aggregate
        "~Top|Top/i:MyModule>a.buzz".rt -> "enq.bits",
        "~Top|Top/i:MyModule>b.fizz".rt -> "deq.bits.fizz", // Checks 1 inner Element
        "~Top|Top/i:MyModule>a.valid".rt -> "enq_valid"
      )
      val expectedLines = List(
        "connect i.a.valid, foo.valid",
        "connect foo.ready, i.a.ready",
        "connect i.a.fizz, foo.bits.fizz",
        "connect i.a.buzz, foo.bits.buzz",
        "connect bar.valid, i.b.valid",
        "connect i.b.ready, bar.ready",
        "connect bar.bits.fizz, i.b.fizz",
        "connect bar.bits.buzz, i.b.buzz"
      )
      val (chirrtl, annos) = getFirrtlAndAnnos(new Top)
      val text = chirrtl.serialize
      for (line <- expectedLines) {
        text should include(line)
      }
      for (e <- expectedAnnos.map(MarkAnnotation.tupled)) {
        annos should contain(e)
      }
    }

    it("(7.c): should work on views of views") {
      import chiselTests.experimental.SimpleBundleDataView._
      @instantiable
      class MyModule extends RawModule {
        private val a = IO(Input(UInt(8.W)))
        private val b = IO(Output(new BundleA(8)))
        @public val in = a.viewAs[UInt].viewAs[UInt]
        @public val out = b.viewAs[BundleB].viewAs[BundleA].viewAs[BundleB]
        out.bar := in
      }
      class Top extends RawModule {
        val foo = IO(Input(UInt(8.W)))
        val bar = IO(Output(new BundleB(8)))
        val i = Instance(Definition(new MyModule))
        i.in := foo
        bar := i.out
        bar.bar := i.out.bar
        mark(i.in, "in")
        mark(i.out.bar, "out_bar")
      }
      val expected = List(
        "~Top|Top/i:MyModule>a".rt -> "in",
        "~Top|Top/i:MyModule>b.foo".rt -> "out_bar"
      )
      val lines = List(
        "connect i.a, foo",
        "connect bar.bar, i.b.foo"
      )
      val (chirrtl, annos) = getFirrtlAndAnnos(new Top)
      val text = chirrtl.serialize
      for (line <- lines) {
        text should include(line)
      }
      for (e <- expected.map(MarkAnnotation.tupled)) {
        annos should contain(e)
      }
    }

    it("(7.d): should work with DataView + implicit conversion") {
      import chisel3.experimental.conversions._
      @instantiable
      class MyModule extends RawModule {
        private val a = IO(Input(UInt(8.W)))
        private val b = IO(Output(UInt(8.W)))
        @public val ports = Seq(a, b)
        b := a
      }
      class Top extends RawModule {
        val foo = IO(Input(UInt(8.W)))
        val bar = IO(Output(UInt(8.W)))
        val i = Instance(Definition(new MyModule))
        i.ports <> Seq(foo, bar)
        mark(i.ports, "i.ports")
      }
      val expected = List(
        // Not 1:1 so will get split out
        "~Top|Top/i:MyModule>a".rt -> "i.ports",
        "~Top|Top/i:MyModule>b".rt -> "i.ports"
      )
      val lines = List(
        "connect i.a, foo",
        "connect bar, i.b"
      )
      val (chirrtl, annos) = getFirrtlAndAnnos(new Top)
      val text = chirrtl.serialize
      for (line <- lines) {
        text should include(line)
      }
      for (e <- expected.map(MarkAnnotation.tupled)) {
        annos should contain(e)
      }
    }

    it("(7.e): should work on Views of BlackBoxes") {
      @instantiable
      class MyBlackBox extends BlackBox {
        @public val io = IO(new Bundle {
          val in = Input(UInt(8.W))
          val out = Output(UInt(8.W))
        })
        @public val innerView = io.viewAs
        @public val foo = io.in.viewAs[UInt]
        @public val bar = io.out.viewAs[UInt]
      }
      class Top extends RawModule {
        val foo = IO(Input(UInt(8.W)))
        val bar = IO(Output(UInt(8.W)))
        val i = Instance(Definition(new MyBlackBox))
        val outerView = i.io.viewAs
        i.foo := foo
        bar := i.bar
        mark(i.foo, "i.foo")
        mark(i.bar, "i.bar")
        mark(i.innerView.in, "i.innerView.in")
        mark(outerView.out, "outerView.out")
      }
      val inst = "~Top|Top/i:MyBlackBox"
      val expectedAnnos = List(
        s"$inst>in".rt -> "i.foo",
        s"$inst>out".rt -> "i.bar",
        s"$inst>in".rt -> "i.innerView.in",
        s"$inst>out".rt -> "outerView.out"
      )
      val expectedLines = List(
        "connect i.in, foo",
        "connect bar, i.out"
      )
      val (chirrtl, annos) = getFirrtlAndAnnos(new Top)
      val text = chirrtl.serialize
      for (line <- expectedLines) {
        text should include(line)
      }
      for (e <- expectedAnnos.map(MarkAnnotation.tupled)) {
        annos should contain(e)
      }
    }
  }

  describe("(8) @instantiable and @public should compose with CloneModuleAsRecord") {
    it("(8.a): it should support @public on a CMAR Record in Definitions") {
      @instantiable
      class HasCMAR extends Module {
        @public val in = IO(Input(UInt(8.W)))
        @public val out = IO(Output(UInt(8.W)))
        @public val m = Module(new AggregatePortModule)
        @public val c = experimental.CloneModuleAsRecord(m)
      }
      class Top extends Module {
        val d = Definition(new HasCMAR)
        mark(d.c("io"), "c.io")
        val bun = d.c("io").asInstanceOf[Record]
        mark(bun.elements("out"), "c.io.out")
      }
      val expected = List(
        "~Top|HasCMAR/c:AggregatePortModule>io".rt -> "c.io",
        "~Top|HasCMAR/c:AggregatePortModule>io.out".rt -> "c.io.out"
      )
      val (_, annos) = getFirrtlAndAnnos(new Top)
      for (e <- expected.map(MarkAnnotation.tupled)) {
        annos should contain(e)
      }
    }
    it("(8.b): it should support @public on a CMAR Record in Instances") {
      @instantiable
      class HasCMAR extends Module {
        @public val in = IO(Input(UInt(8.W)))
        @public val out = IO(Output(UInt(8.W)))
        @public val m = Module(new AggregatePortModule)
        @public val c = experimental.CloneModuleAsRecord(m)
      }
      class Top extends Module {
        val i = Instance(Definition(new HasCMAR))
        mark(i.c("io"), "i.c.io")
        val bun = i.c("io").asInstanceOf[Record]
        mark(bun.elements("out"), "i.c.io.out")
      }
      val expected = List(
        "~Top|Top/i:HasCMAR/c:AggregatePortModule>io".rt -> "i.c.io",
        "~Top|Top/i:HasCMAR/c:AggregatePortModule>io.out".rt -> "i.c.io.out"
      )
      val (_, annos) = getFirrtlAndAnnos(new Top)
      for (e <- expected.map(MarkAnnotation.tupled)) {
        annos should contain(e)
      }
    }
  }
  describe("(9) isA[..]") {
    it("(9.a): it should work on simple classes") {
      class Top extends Module {
        val d = Definition(new AddOne)
        require(d.isA[AddOne])
      }
      getFirrtlAndAnnos(new Top)
    }
    it("(9.b): it should not work on inner classes") {
      class InnerClass extends Module
      class Top extends Module {
        val d = Definition(new InnerClass)
        "require(d.isA[Module])" should compile // ensures that the test below is checking something useful
        "require(d.isA[InnerClass])" shouldNot compile
      }
      getFirrtlAndAnnos(new Top)
    }
    it("(9.c): it should work on super classes") {
      class InnerClass extends Module
      class Top extends Module {
        val d = Definition(new InnerClass)
        require(d.isA[Module])
      }
      getFirrtlAndAnnos(new Top)
    }
    it("(9.d): it should work after casts") {
      class Top extends Module {
        val d0: Definition[Module] = Definition(new AddOne)
        require(d0.isA[AddOne])
        val d1: Definition[Module] = Definition((new AddOne).asInstanceOf[Module])
        require(d1.isA[AddOne])
        val i0: Instance[Module] = Instance(d0)
        require(i0.isA[AddOne])
        val i1: Instance[Module] = Instance(d1)
        require(i1.isA[AddOne])
        val i2: Instance[Module] = Instance(Definition(new AddOne))
        require(i2.isA[AddOne])
      }
      getFirrtlAndAnnos(new Top)
    }
    it("(9.e): it should ignore type parameters (even though it would be nice if it didn't)") {
      class Top extends Module {
        val d0: Definition[Module] = Definition(new HasTypeParams(Bool()))
        require(d0.isA[HasTypeParams[Bool]])
        require(d0.isA[HasTypeParams[_]])
        require(d0.isA[HasTypeParams[UInt]])
        require(!d0.isA[HasBlah])
      }
      getFirrtlAndAnnos(new Top)
    }
  }
  describe("(10) Select APIs") {
    it("(10.a): instancesOf") {
      val aspect = aop.inspecting.InspectingAspect({ m: AddTwoMixedModules =>
        val targets = aop.Select.instancesOf[AddOne](m.toDefinition).map { i: Instance[AddOne] => i.toTarget }
        targets should be(
          Seq(
            "~AddTwoMixedModules|AddTwoMixedModules/i0:AddOne".it,
            "~AddTwoMixedModules|AddTwoMixedModules/i1:AddOne_1".it
          )
        )
      })
      getFirrtlAndAnnos(new AddTwoMixedModules, Seq(aspect))
    }
    it("(10.b): instancesIn") {
      val aspect = aop.inspecting.InspectingAspect({ m: AddTwoMixedModules =>
        val insts = aop.Select.instancesIn(m.toDefinition)
        val abs = insts.map { i: Instance[BaseModule] => i.toAbsoluteTarget }
        val rel = insts.map { i: Instance[BaseModule] => i.toTarget }
        abs should be(
          Seq(
            "~AddTwoMixedModules|AddTwoMixedModules/i0:AddOne".it,
            "~AddTwoMixedModules|AddTwoMixedModules/i1:AddOne_1".it
          )
        )
        rel should be(
          Seq(
            "~AddTwoMixedModules|AddTwoMixedModules/i0:AddOne".it,
            "~AddTwoMixedModules|AddTwoMixedModules/i1:AddOne_1".it
          )
        )
      })
      getFirrtlAndAnnos(new AddTwoMixedModules, Seq(aspect))
    }
    it("(10.c): allInstancesOf") {
      val aspect = aop.inspecting.InspectingAspect({ m: AddFour =>
        val insts = aop.Select.allInstancesOf[AddOne](m.toDefinition)
        val abs = insts.map { i: Instance[AddOne] => i.in.toAbsoluteTarget }
        val rel = insts.map { i: Instance[AddOne] => i.in.toTarget }
        rel should be(
          Seq(
            "~AddFour|AddFour/i0:AddTwoMixedModules/i0:AddOne>in".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>in".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i0:AddOne>in".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>in".rt
          )
        )
        abs should be(
          Seq(
            "~AddFour|AddFour/i0:AddTwoMixedModules/i0:AddOne>in".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>in".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i0:AddOne>in".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>in".rt
          )
        )
      })
      getFirrtlAndAnnos(new AddFour, Seq(aspect))
    }
    it("(10.d): definitionsOf") {
      val aspect = aop.inspecting.InspectingAspect({ m: AddTwoMixedModules =>
        val targets = aop.Select.definitionsOf[AddOne](m.toDefinition).map { i: Definition[AddOne] => i.in.toTarget }
        targets should be(
          Seq(
            "~AddTwoMixedModules|AddOne>in".rt,
            "~AddTwoMixedModules|AddOne_1>in".rt
          )
        )
      })
      getFirrtlAndAnnos(new AddTwoMixedModules, Seq(aspect))
    }
    it("(10.e): definitionsIn") {
      val aspect = aop.inspecting.InspectingAspect({ m: AddTwoMixedModules =>
        val targets = aop.Select.definitionsIn(m.toDefinition).map { i: Definition[BaseModule] => i.toTarget }
        targets should be(
          Seq(
            "~AddTwoMixedModules|AddOne".mt,
            "~AddTwoMixedModules|AddOne_1".mt
          )
        )
      })
      getFirrtlAndAnnos(new AddTwoMixedModules, Seq(aspect))
    }
    it("(10.f): allDefinitionsOf") {
      val aspect = aop.inspecting.InspectingAspect({ m: AddFour =>
        val targets = aop.Select.allDefinitionsOf[AddOne](m.toDefinition).map { i: Definition[AddOne] => i.in.toTarget }
        targets should be(
          Seq(
            "~AddFour|AddOne>in".rt,
            "~AddFour|AddOne_1>in".rt
          )
        )
      })
      getFirrtlAndAnnos(new AddFour, Seq(aspect))
    }
    it("(10.g): Select.collectDeep should fail when combined with hierarchy package") {
      val aspect = aop.inspecting.InspectingAspect({ m: AddFour =>
        aop.Select.collectDeep(m) { case m: AddOne => m.toTarget }
      })
      intercept[Exception] { getFirrtlAndAnnos(new AddFour, Seq(aspect)) }
    }
    it("(10.h): Select.getDeep should fail when combined with hierarchy package") {
      val aspect = aop.inspecting.InspectingAspect({ m: AddFour =>
        aop.Select.getDeep(m) { m: BaseModule => Nil }
      })
      intercept[Exception] { getFirrtlAndAnnos(new AddFour, Seq(aspect)) }
    }
    it("(10.i): Select.instances should fail when combined with hierarchy package") {
      val aspect = aop.inspecting.InspectingAspect({ m: AddFour =>
        aop.Select.instances(m)
      })
      intercept[Exception] { getFirrtlAndAnnos(new AddFour, Seq(aspect)) }
    }
    it("(10.j): allInstancesOf.ios") {
      val aspect = aop.inspecting.InspectingAspect({ m: AddFour =>
        val abs = aop.Select.allInstancesOf[AddOne](m.toDefinition).flatMap { i: Instance[AddOne] =>
          aop.Select.ios(i).map(_.toAbsoluteTarget)
        }
        val rel = aop.Select.allInstancesOf[AddOne](m.toDefinition).flatMap { i: Instance[AddOne] =>
          aop.Select.ios(i).map(_.toTarget)
        }
        abs should be(
          Seq(
            "~AddFour|AddFour/i0:AddTwoMixedModules/i0:AddOne>clock".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i0:AddOne>reset".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i0:AddOne>in".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i0:AddOne>out".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>clock".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>reset".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>in".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>out".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i0:AddOne>clock".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i0:AddOne>reset".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i0:AddOne>in".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i0:AddOne>out".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>clock".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>reset".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>in".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>out".rt
          )
        )

        rel should be(
          Seq(
            "~AddFour|AddFour/i0:AddTwoMixedModules/i0:AddOne>clock".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i0:AddOne>reset".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i0:AddOne>in".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i0:AddOne>out".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>clock".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>reset".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>in".rt,
            "~AddFour|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>out".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i0:AddOne>clock".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i0:AddOne>reset".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i0:AddOne>in".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i0:AddOne>out".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>clock".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>reset".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>in".rt,
            "~AddFour|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>out".rt
          )
        )
      })
      getFirrtlAndAnnos(new AddFour, Seq(aspect))
    }
    it("(10.k): allDefinitionsOf.ios") {
      val aspect = aop.inspecting.InspectingAspect({ m: AddFour =>
        val abs = aop.Select.allDefinitionsOf[AddOne](m.toDefinition).flatMap { i: Definition[AddOne] =>
          aop.Select.ios(i).map(_.toAbsoluteTarget)
        }
        val rel = aop.Select.allDefinitionsOf[AddOne](m.toDefinition).flatMap { i: Definition[AddOne] =>
          aop.Select.ios(i).map(_.toTarget)
        }
        abs should be(
          Seq(
            "~AddFour|AddOne>clock".rt,
            "~AddFour|AddOne>reset".rt,
            "~AddFour|AddOne>in".rt,
            "~AddFour|AddOne>out".rt,
            "~AddFour|AddOne_1>clock".rt,
            "~AddFour|AddOne_1>reset".rt,
            "~AddFour|AddOne_1>in".rt,
            "~AddFour|AddOne_1>out".rt
          )
        )

        rel should be(
          Seq(
            "~AddFour|AddOne>clock".rt,
            "~AddFour|AddOne>reset".rt,
            "~AddFour|AddOne>in".rt,
            "~AddFour|AddOne>out".rt,
            "~AddFour|AddOne_1>clock".rt,
            "~AddFour|AddOne_1>reset".rt,
            "~AddFour|AddOne_1>in".rt,
            "~AddFour|AddOne_1>out".rt
          )
        )

      })
      getFirrtlAndAnnos(new AddFour, Seq(aspect))
    }
    it("(10.l): Select.instancesIn for typed BaseModules") {
      val aspect = aop.inspecting.InspectingAspect({ m: HasMultipleTypeParamsInside =>
        val targets = aop.Select.instancesIn(m.toDefinition).map { i: Instance[BaseModule] => i.toTarget }
        targets should be(
          Seq(
            "~HasMultipleTypeParamsInside|HasMultipleTypeParamsInside/i00:HasTypeParams".it,
            "~HasMultipleTypeParamsInside|HasMultipleTypeParamsInside/i01:HasTypeParams".it,
            "~HasMultipleTypeParamsInside|HasMultipleTypeParamsInside/i10:HasTypeParams_1".it,
            "~HasMultipleTypeParamsInside|HasMultipleTypeParamsInside/i11:HasTypeParams_1".it
          )
        )
      })
      getFirrtlAndAnnos(new HasMultipleTypeParamsInside, Seq(aspect))
    }
    it("(10.m): Select.instancesOf for typed BaseModules if type is ignored") {
      val aspect = aop.inspecting.InspectingAspect({ m: HasMultipleTypeParamsInside =>
        val targets =
          aop.Select.instancesOf[HasTypeParams[_]](m.toDefinition).map { i: Instance[HasTypeParams[_]] => i.toTarget }
        targets should be(
          Seq(
            "~HasMultipleTypeParamsInside|HasMultipleTypeParamsInside/i00:HasTypeParams".it,
            "~HasMultipleTypeParamsInside|HasMultipleTypeParamsInside/i01:HasTypeParams".it,
            "~HasMultipleTypeParamsInside|HasMultipleTypeParamsInside/i10:HasTypeParams_1".it,
            "~HasMultipleTypeParamsInside|HasMultipleTypeParamsInside/i11:HasTypeParams_1".it
          )
        )
      })
      getFirrtlAndAnnos(new HasMultipleTypeParamsInside, Seq(aspect))
    }
    it(
      "10.n Select.instancesOf for typed BaseModules even type is specified wrongly (should be ignored, even though we wish it weren't)"
    ) {
      val aspect = aop.inspecting.InspectingAspect({ m: HasMultipleTypeParamsInside =>
        val targets = aop.Select.instancesOf[HasTypeParams[SInt]](m.toDefinition).map { i: Instance[HasTypeParams[_]] =>
          i.toTarget
        }
        targets should be(
          Seq(
            "~HasMultipleTypeParamsInside|HasMultipleTypeParamsInside/i00:HasTypeParams".it,
            "~HasMultipleTypeParamsInside|HasMultipleTypeParamsInside/i01:HasTypeParams".it,
            "~HasMultipleTypeParamsInside|HasMultipleTypeParamsInside/i10:HasTypeParams_1".it,
            "~HasMultipleTypeParamsInside|HasMultipleTypeParamsInside/i11:HasTypeParams_1".it
          )
        )
      })
      getFirrtlAndAnnos(new HasMultipleTypeParamsInside, Seq(aspect))
    }
  }
}
