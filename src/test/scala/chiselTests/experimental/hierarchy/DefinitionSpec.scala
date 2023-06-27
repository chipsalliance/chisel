// SPDX-License-Identifier: Apache-2.0

package chiselTests
package experimental.hierarchy

import chisel3._
import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance}

// TODO/Notes
// - In backport, clock/reset are not automatically assigned. I think this is fixed in 3.5
// - CircuitTarget for annotations on the definition are wrong - needs to be fixed.
class DefinitionSpec extends ChiselFunSpec with Utils {
  import Annotations._
  import Examples._
  describe("(0): Definition instantiation") {
    it("(0.a): module name of a definition should be correct") {
      class Top extends Module {
        val definition = Definition(new AddOne)
      }
      val (chirrtl, _) = getFirrtlAndAnnos(new Top)
      chirrtl.serialize should include("module AddOne :")
    }
    it("(0.b): accessing internal fields through non-generated means is hard to do") {
      class Top extends Module {
        val definition = Definition(new AddOne)
        //definition.lookup(_.in) // Uncommenting this line will give the following error:
        //"You are trying to access a macro-only API. Please use the @public annotation instead."
        definition.in
      }
      val (chirrtl, _) = getFirrtlAndAnnos(new Top)
      chirrtl.serialize should include("module AddOne :")
    }
    it("(0.c): reset inference is not defaulted to Bool for definitions") {
      class Top extends Module with RequireAsyncReset {
        val definition = Definition(new HasUninferredReset)
        val i0 = Instance(definition)
        i0.in := 0.U
      }
      val (chirrtl, _) = getFirrtlAndAnnos(new Top)
      chirrtl.serialize should include("inst i0 of HasUninferredReset")
    }
    it("(0.d): module names of repeated definition should be sequential") {
      class Top extends Module {
        val k = Module(
          new AddTwoParameterized(
            4,
            (x: Int) =>
              Seq.tabulate(x) { j =>
                val addOneDef = Definition(new AddOneParameterized(x + j))
                val addOne = Instance(addOneDef)
                addOne
              }
          )
        )
      }
      val (chirrtl, _) = getFirrtlAndAnnos(new Top)
      chirrtl.serialize should include("module AddOneParameterized :")
      chirrtl.serialize should include("module AddOneParameterized_1 :")
      chirrtl.serialize should include("module AddOneParameterized_2 :")
      chirrtl.serialize should include("module AddOneParameterized_3 :")
    }
    it("(0.e): multiple instantiations should have sequential names") {
      class Top extends Module {
        val addOneDef = Definition(new AddOneParameterized(4))
        val addOne = Instance(addOneDef)
        val otherAddOne = Module(new AddOneParameterized(4))
      }
      val (chirrtl, _) = getFirrtlAndAnnos(new Top)
      chirrtl.serialize should include("module AddOneParameterized :")
      chirrtl.serialize should include("module AddOneParameterized_1 :")
    }
    it("(0.f): nested definitions should have sequential names") {
      class Top extends Module {
        val k = Module(
          new AddTwoWithNested(
            4,
            (x: Int) =>
              Seq.tabulate(x) { j =>
                val addOneDef = Definition(new AddOneWithNested(x + j))
                val addOne = Instance(addOneDef)
                addOne
              }
          )
        )
      }
      val (chirrtl, _) = getFirrtlAndAnnos(new Top)
      chirrtl.serialize should include("module AddOneWithNested :")
      chirrtl.serialize should include("module AddOneWithNested_1 :")
      chirrtl.serialize should include("module AddOneWithNested_2 :")
      chirrtl.serialize should include("module AddOneWithNested_3 :")
    }
  }
  describe("(1): Annotations on definitions in same chisel compilation") {
    it("(1.a): should work on a single definition, annotating the definition") {
      class Top extends Module {
        val definition: Definition[AddOne] = Definition(new AddOne)
        mark(definition, "mark")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddOne".mt, "mark"))
    }
    it("(1.b): should work on a single definition, annotating an inner wire") {
      class Top extends Module {
        val definition: Definition[AddOne] = Definition(new AddOne)
        mark(definition.innerWire, "i0.innerWire")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddOne>innerWire".rt, "i0.innerWire"))
    }
    it("(1.c): should work on a two nested definitions, annotating the definition") {
      class Top extends Module {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        mark(definition.definition, "i0.i0")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddOne".mt, "i0.i0"))
    }
    it("(1.d): should work on an instance in a definition, annotating the instance") {
      class Top extends Module {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        mark(definition.i0, "i0.i0")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddTwo/i0:AddOne".it, "i0.i0"))
    }
    it("(1.e): should work on a definition in an instance, annotating the definition") {
      class Top extends Module {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        val i0 = Instance(definition)
        mark(i0.definition, "i0.i0")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddOne".mt, "i0.i0"))
    }
    it("(1.f): should work on a wire in an instance in a definition") {
      class Top extends Module {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        mark(definition.i0.innerWire, "i0.i0.innerWire")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddTwo/i0:AddOne>innerWire".rt, "i0.i0.innerWire"))
    }
    it("(1.g): should work on a nested module in a definition, annotating the module") {
      class Top extends Module {
        val definition: Definition[AddTwoMixedModules] = Definition(new AddTwoMixedModules)
        mark(definition.i1, "i0.i1")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddTwoMixedModules/i1:AddOne_1".it, "i0.i1"))
    }
    // Can you define an instantiable container? I think not.
    // Instead, we can test the instantiable container in a definition
    it("(1.h): should work on an instantiable container, annotating a wire in the defintion") {
      class Top extends Module {
        val definition: Definition[AddOneWithInstantiableWire] = Definition(new AddOneWithInstantiableWire)
        mark(definition.wireContainer.innerWire, "i0.innerWire")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddOneWithInstantiableWire>innerWire".rt, "i0.innerWire"))
    }
    it("(1.i): should work on an instantiable container, annotating a module") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableModule)
        mark(definition.moduleContainer.i0, "i0.i0")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddOneWithInstantiableModule/i0:AddOne".it, "i0.i0"))
    }
    it("(1.j): should work on an instantiable container, annotating an instance") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableInstance)
        mark(definition.instanceContainer.i0, "i0.i0")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddOneWithInstantiableInstance/i0:AddOne".it, "i0.i0"))
    }
    it("(1.k): should work on an instantiable container, annotating an instantiable container's module") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableInstantiable)
        mark(definition.containerContainer.container.i0, "i0.i0")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddOneWithInstantiableInstantiable/i0:AddOne".it, "i0.i0"))
    }
    it("(1.l): should work on public member which references public member of another instance") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableInstantiable)
        mark(definition.containerContainer.container.i0, "i0.i0")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddOneWithInstantiableInstantiable/i0:AddOne".it, "i0.i0"))
    }
    it("(1.m): should work for targets on definition to have correct circuit name") {
      class Top extends Module {
        val definition = Definition(new AddOneWithAnnotation)
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddOneWithAnnotation>innerWire".rt, "innerWire"))
    }
  }
  describe("(2): Annotations on designs not in the same chisel compilation") {
    it("(2.a): should work on an innerWire, marked in a different compilation") {
      val first = elaborateAndGetModule(new AddTwo)
      class Top(x: AddTwo) extends Module {
        val parent = Definition(new ViewerParent(x, false, true))
      }
      val (_, annos) = getFirrtlAndAnnos(new Top(first))
      annos should contain(MarkAnnotation("~AddTwo|AddTwo/i0:AddOne>innerWire".rt, "first"))
    }
    it("(2.b): should work on an innerWire, marked in a different compilation, in instanced instantiable") {
      val first = elaborateAndGetModule(new AddTwo)
      class Top(x: AddTwo) extends Module {
        val parent = Definition(new ViewerParent(x, true, false))
      }
      val (_, annos) = getFirrtlAndAnnos(new Top(first))
      annos should contain(MarkAnnotation("~AddTwo|AddTwo/i0:AddOne>innerWire".rt, "second"))
    }
    it("(2.c): should work on an innerWire, marked in a different compilation, in instanced module") {
      val first = elaborateAndGetModule(new AddTwo)
      class Top(x: AddTwo) extends Module {
        val parent = Definition(new ViewerParent(x, false, false))
        mark(parent.viewer.x.i0.innerWire, "third")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top(first))
      annos should contain(MarkAnnotation("~AddTwo|AddTwo/i0:AddOne>innerWire".rt, "third"))
    }
  }
  describe("(3): @public") {
    it("(3.a): should work on multi-vals") {
      class Top() extends Module {
        val mv = Definition(new MultiVal())
        mark(mv.x, "mv.x")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|MultiVal>x".rt, "mv.x"))
    }
    it("(3.b): should work on lazy vals") {
      class Top() extends Module {
        val lv = Definition(new LazyVal())
        mark(lv.x, lv.y)
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|LazyVal>x".rt, "Hi"))
    }
    it("(3.c): should work on islookupables") {
      class Top() extends Module {
        val p = Parameters("hi", 0)
        val up = Definition(new UsesParameters(p))
        mark(up.x, up.y.string + up.y.int)
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|UsesParameters>x".rt, "hi0"))
    }
    it("(3.d): should work on lists") {
      class Top() extends Module {
        val i = Definition(new HasList())
        mark(i.x(1), i.y(1).toString)
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|HasList>x_1".rt, "2"))
    }
    it("(3.e): should work on seqs") {
      class Top() extends Module {
        val i = Definition(new HasSeq())
        mark(i.x(1), i.y(1).toString)
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|HasSeq>x_1".rt, "2"))
    }
    it("(3.f): should work on options") {
      class Top() extends Module {
        val i = Definition(new HasOption())
        i.x.map(x => mark(x, "x"))
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|HasOption>x".rt, "x"))
    }
    it("(3.g): should work on vecs") {
      class Top() extends Module {
        val i = Definition(new HasVec())
        mark(i.x, "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|HasVec>x".rt, "blah"))
    }
    it("(3.h): should work on statically indexed vectors external to module") {
      class Top() extends Module {
        val i = Definition(new HasVec())
        mark(i.x(1), "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|HasVec>x[1]".rt, "blah"))
    }
    it("(3.i): should work on statically indexed vectors internal to module") {
      class Top() extends Module {
        val i = Definition(new HasIndexedVec())
        mark(i.y, "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|HasIndexedVec>x[1]".rt, "blah"))
    }
    ignore("(3.j): should work on vals in constructor arguments") {
      class Top() extends Module {
        val i = Definition(new HasPublicConstructorArgs(10))
        //mark(i.x, i.int.toString)
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|HasPublicConstructorArgs>x".rt, "10"))
    }
    it("(3.k): should work on unimplemented vals in abstract classes/traits") {
      class Top() extends Module {
        val i = Definition(new ConcreteHasBlah())
        def f(d: Definition[HasBlah]): Unit = {
          mark(d, d.blah.toString)
        }
        f(i)
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|ConcreteHasBlah".mt, "10"))
    }
    it("(3.l): should work on eithers") {
      class Top() extends Module {
        val i = Definition(new HasEither())
        i.x.map(x => mark(x, "xright")).left.map(x => mark(x, "xleft"))
        i.y.map(x => mark(x, "yright")).left.map(x => mark(x, "yleft"))
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|HasEither>x".rt, "xright"))
      annos should contain(MarkAnnotation("~Top|HasEither>y".rt, "yleft"))
    }
    it("(3.m): should work on tuple2") {
      class Top() extends Module {
        val i = Definition(new HasTuple2())
        mark(i.xy._1, "x")
        mark(i.xy._2, "y")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|HasTuple2>x".rt, "x"))
      annos should contain(MarkAnnotation("~Top|HasTuple2>y".rt, "y"))
    }
    it("(3.n): should work on Mems/SyncReadMems") {
      class Top() extends Module {
        val i = Definition(new HasMems())
        mark(i.mem, "Mem")
        mark(i.syncReadMem, "SyncReadMem")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|HasMems>mem".rt, "Mem"))
      annos should contain(MarkAnnotation("~Top|HasMems>syncReadMem".rt, "SyncReadMem"))
    }
    it("(3.o): should not create memory ports") {
      class Top() extends Module {
        val i = Definition(new HasMems())
        i.mem(0) := 100.U // should be illegal!
      }
      val failure = intercept[ChiselException] {
        getFirrtlAndAnnos(new Top)
      }
      assert(
        failure.getMessage ==
          "Cannot create a memory port in a different module (Top) than where the memory is (HasMems)."
      )
    }
  }
  describe("(4): toDefinition") {
    it("(4.a): should work on modules") {
      class Top() extends Module {
        val i = Module(new AddOne())
        f(i.toDefinition)
      }
      def f(i: Definition[AddOne]): Unit = mark(i.innerWire, "blah")
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddOne>innerWire".rt, "blah"))
    }
    it("(4.b): should work on seqs of modules") {
      class Top() extends Module {
        val is = Seq(Module(new AddTwo()), Module(new AddTwo())).map(_.toDefinition)
        mark(f(is), "blah")
      }
      def f(i: Seq[Definition[AddTwo]]): Data = i.head.i0.innerWire
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddTwo/i0:AddOne>innerWire".rt, "blah"))
    }
    it("(4.c): should work on options of modules") {
      class Top() extends Module {
        val is: Option[Definition[AddTwo]] = Some(Module(new AddTwo())).map(_.toDefinition)
        mark(f(is), "blah")
      }
      def f(i: Option[Definition[AddTwo]]): Data = i.get.i0.innerWire
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddTwo/i0:AddOne>innerWire".rt, "blah"))
    }
  }
  describe("(5): Absolute Targets should work as expected") {
    it("(5.a): toAbsoluteTarget on a port of a definition") {
      class Top() extends Module {
        val i = Definition(new AddTwo())
        amark(i.in, "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddTwo>in".rt, "blah"))
    }
    it("(5.b): toAbsoluteTarget on a subinstance's data within a definition") {
      class Top() extends Module {
        val i = Definition(new AddTwo())
        amark(i.i0.innerWire, "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddTwo/i0:AddOne>innerWire".rt, "blah"))
    }
    it("(5.c): toAbsoluteTarget on a submodule's data within a definition") {
      class Top() extends Module {
        val i = Definition(new AddTwoMixedModules())
        amark(i.i1.in, "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|AddTwoMixedModules/i1:AddOne_1>in".rt, "blah"))
    }
    it("(5.d): toAbsoluteTarget on a submodule's data, in an aggregate, within a definition") {
      class Top() extends Module {
        val i = Definition(new InstantiatesHasVec())
        amark(i.i1.x.head, "blah")
      }
      val (_, annos) = getFirrtlAndAnnos(new Top)
      annos should contain(MarkAnnotation("~Top|InstantiatesHasVec/i1:HasVec_1>x[0]".rt, "blah"))
    }
  }
  describe("(6): @instantiable traits should work as expected") {
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

    it("(6.a): A Module that implements an @instantiable trait should be definable as that trait") {
      class Top extends Module {
        val i: Definition[ModuleIntf] = Definition(new ModuleWithCommonIntf)
        mark(i.io.in, "gotcha")
        mark(i, "inst")
      }
      val expected = List(
        "~Top|ModuleWithCommonIntf>io.in".rt -> "gotcha",
        "~Top|ModuleWithCommonIntf".mt -> "inst"
      )
      val (chirrtl, annos) = getFirrtlAndAnnos(new Top)
      for (e <- expected.map(MarkAnnotation.tupled)) {
        annos should contain(e)
      }
    }
    it(
      "(6.b): An @instantiable Module implementing an @instantiable trait should be able to use extension methods from both"
    ) {
      class Top extends Module {
        val i: Definition[ModuleWithCommonIntf] = Definition(new ModuleWithCommonIntf)
        mark(i.io.in, "gotcha")
        mark(i.sum, "also this")
        mark(i, "inst")
      }
      val expected = List(
        "~Top|ModuleWithCommonIntf>io.in".rt -> "gotcha",
        "~Top|ModuleWithCommonIntf>sum".rt -> "also this",
        "~Top|ModuleWithCommonIntf".mt -> "inst"
      )
      val (chirrtl, annos) = getFirrtlAndAnnos(new Top)
      for (e <- expected.map(MarkAnnotation.tupled)) {
        annos should contain(e)
      }
    }
    it("(6.c): A BlackBox that implements an @instantiable trait should be instantiable as that trait") {
      class Top extends Module {
        val m: ModuleIntf = Module(new BlackBoxWithCommonIntf)
        val d: Definition[ModuleIntf] = m.toDefinition
        mark(d.io.in, "gotcha")
        mark(d, "module")
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
        val definition = Definition(new ModuleWithCommonIntf("X"))
        val insts: Seq[Definition[ModuleIntf]] = Vector(
          Module(new ModuleWithCommonIntf("Y")).toDefinition,
          Module(new BlackBoxWithCommonIntf).toDefinition,
          definition
        )
        mark(insts(0).io.in, "foo")
        mark(insts(1).io.in, "bar")
        mark(insts(2).io.in, "fizz")
      }
      val expected = List(
        "~Top|ModuleWithCommonIntfY>io.in".rt -> "foo",
        "~Top|BlackBoxWithCommonIntf>in".rt -> "bar",
        "~Top|ModuleWithCommonIntfX>io.in".rt -> "fizz"
      )
      val (chirrtl, annos) = getFirrtlAndAnnos(new Top)
      for (e <- expected.map(MarkAnnotation.tupled)) {
        annos should contain(e)
      }
    }
  }
  describe("(7): @instantiable and @public should compose with DataView") {
    import chisel3.experimental.dataview._
    ignore("(7.a): should work on simple Views") {
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
        val d = Definition(new MyModule)
        val i = Instance(d)
        i.foo := foo
        bar := i.out
        mark(d.out, "out")
        mark(d.foo, "foo")
        mark(d.bar, "bar")
      }
      val expectedAnnos = List(
        "~Top|MyModule>out".rt -> "out",
        "~Top|MyModule>in".rt -> "foo",
        "~Top|MyModule>sum".rt -> "bar"
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
    ignore("(7.b): should work on Aggregate Views that are mapped 1:1") {
      import chiselTests.experimental.SimpleBundleDataView._
      @instantiable
      class MyModule extends RawModule {
        private val a = IO(Input(new BundleA(8)))
        private val b = IO(Output(new BundleA(8)))
        @public val in = a.viewAs[BundleB]
        @public val out = b.viewAs[BundleB]
        out := in
      }
      class Top extends RawModule {
        val foo = IO(Input(new BundleB(8)))
        val bar = IO(Output(new BundleB(8)))
        val d = Definition(new MyModule)
        val i = Instance(d)
        i.in := foo
        bar.bar := i.out.bar
        mark(d.in, "in")
        mark(d.in.bar, "in_bar")
      }
      val expectedAnnos = List(
        "~Top|MyModule>a".rt -> "in",
        "~Top|MyModule>a.foo".rt -> "in_bar"
      )
      val expectedLines = List(
        "connect i.a, foo",
        "connect bar, i.b.foo"
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
}
