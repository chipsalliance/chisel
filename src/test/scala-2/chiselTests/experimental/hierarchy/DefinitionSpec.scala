// SPDX-License-Identifier: Apache-2.0

package chiselTests
package experimental.hierarchy

import chisel3._
import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance}
import chiselTests.experimental.ExtensionMethods.ChiselStageHelpers
import circt.stage.ChiselStage
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

// TODO/Notes
// - In backport, clock/reset are not automatically assigned. I think this is fixed in 3.5
// - CircuitTarget for annotations on the definition are wrong - needs to be fixed.
class DefinitionSpec extends AnyFunSpec with Matchers with FileCheck {
  import Annotations._
  import Examples._
  describe("(0): Definition instantiation") {
    it("(0.a): module name of a definition should be correct") {
      class Top extends Module {
        val definition = Definition(new AddOne)
      }
      ChiselStage.emitCHIRRTL(new Top) should include("module AddOne :")
    }
    it("(0.b): accessing internal fields through non-generated means is hard to do") {
      class Top extends Module {
        val definition = Definition(new AddOne)
        // definition.lookup(_.in) // Uncommenting this line will give the following error:
        // "You are trying to access a macro-only API. Please use the @public annotation instead."
        definition.in
      }
      ChiselStage.emitCHIRRTL(new Top) should include("module AddOne :")
    }
    it("(0.c): reset inference is not defaulted to Bool for definitions") {
      class Top extends Module with RequireAsyncReset {
        val definition = Definition(new HasUninferredReset)
        val i0 = Instance(definition)
        i0.in := 0.U
      }
      ChiselStage.emitCHIRRTL(new Top) should include("inst i0 of HasUninferredReset")
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
      val chirrtl = ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("module AddOneParameterized :")
      chirrtl should include("module AddOneParameterized_1 :")
      chirrtl should include("module AddOneParameterized_2 :")
      chirrtl should include("module AddOneParameterized_3 :")
    }
    it("(0.e): multiple instantiations should have sequential names") {
      class Top extends Module {
        val addOneDef = Definition(new AddOneParameterized(4))
        val addOne = Instance(addOneDef)
        val otherAddOne = Module(new AddOneParameterized(4))
      }
      val chirrtl = ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("module AddOneParameterized :")
      chirrtl should include("module AddOneParameterized_1 :")
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
      val chirrtl = ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("module AddOneWithNested :")
      chirrtl should include("module AddOneWithNested_1 :")
      chirrtl should include("module AddOneWithNested_2 :")
      chirrtl should include("module AddOneWithNested_3 :")
    }

    it("(0.g): definitions should work as arguments to definitions") {
      class Top extends Module {
        val addOne = Definition(new AddOne)
        val addTwo = Definition(new AddTwoDefinitionArgument(addOne))
        val inst = Instance(addTwo)
        inst.in := 12.U
      }
      val chirrtl = ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("module AddOne :")
      chirrtl shouldNot include("module AddOne_")
      chirrtl should include("module AddTwoDefinitionArgument :")
      chirrtl should include("module Top :")
    }
    it("(0.h): definitions created from Modules should work as arguments to definitions") {
      class Top extends Module {
        val addOne = Module(new AddOne)
        val addTwo = Definition(new AddTwoDefinitionArgument(addOne.toDefinition))
        val inst = Instance(addTwo)
        inst.in := 12.U
      }
      val chirrtl = ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("module AddOne :")
      chirrtl shouldNot include("module AddOne_")
      chirrtl should include("module AddTwoDefinitionArgument :")
      chirrtl should include("module Top :")
      chirrtl should include("inst addOne of AddOne")
    }
  }
  describe("(1): Annotations on definitions in same chisel compilation") {
    it("(1.a): should work on a single definition, annotating the definition") {
      class Top extends Module {
        val definition: Definition[AddOne] = Definition(new AddOne)
        mark(definition, "mark")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddOne"
           |CHECK-NEXT: "tag":"mark"
           |""".stripMargin
      )
    }
    it("(1.b): should work on a single definition, annotating an inner wire") {
      class Top extends Module {
        val definition: Definition[AddOne] = Definition(new AddOne)
        mark(definition.innerWire, "i0.innerWire")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddOne>innerWire"
           |CHECK-NEXT: "tag":"i0.innerWire"
           |""".stripMargin
      )
    }
    it("(1.c): should work on a two nested definitions, annotating the definition") {
      class Top extends Module {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        mark(definition.definition, "i0.i0")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddOne"
           |CHECK-NEXT: "tag":"i0.i0"
           |""".stripMargin
      )
    }
    it("(1.d): should work on an instance in a definition, annotating the instance") {
      class Top extends Module {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        mark(definition.i0, "i0.i0")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddTwo/i0:AddOne"
           |CHECK-NEXT: "tag":"i0.i0"
           |""".stripMargin
      )
    }
    it("(1.e): should work on a definition in an instance, annotating the definition") {
      class Top extends Module {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        val i0 = Instance(definition)
        mark(i0.definition, "i0.i0")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddOne"
           |CHECK-NEXT: "tag":"i0.i0"
           |""".stripMargin
      )
    }
    it("(1.f): should work on a wire in an instance in a definition") {
      class Top extends Module {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        mark(definition.i0.innerWire, "i0.i0.innerWire")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddTwo/i0:AddOne>innerWire"
           |CHECK-NEXT: "tag":"i0.i0.innerWire"
           |""".stripMargin
      )
    }
    it("(1.g): should work on a nested module in a definition, annotating the module") {
      class Top extends Module {
        val definition: Definition[AddTwoMixedModules] = Definition(new AddTwoMixedModules)
        mark(definition.i1, "i0.i1")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddTwoMixedModules/i1:AddOne_1"
           |CHECK-NEXT: "tag":"i0.i1"
           |""".stripMargin
      )
    }
    // Can you define an instantiable container? I think not.
    // Instead, we can test the instantiable container in a definition
    it("(1.h): should work on an instantiable container, annotating a wire in the defintion") {
      class Top extends Module {
        val definition: Definition[AddOneWithInstantiableWire] = Definition(new AddOneWithInstantiableWire)
        mark(definition.wireContainer.innerWire, "i0.innerWire")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddOneWithInstantiableWire>innerWire"
           |CHECK-NEXT: "tag":"i0.innerWire"
           |""".stripMargin
      )
    }
    it("(1.i): should work on an instantiable container, annotating a module") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableModule)
        mark(definition.moduleContainer.i0, "i0.i0")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddOneWithInstantiableModule/i0:AddOne"
           |CHECK-NEXT: "tag":"i0.i0"
           |""".stripMargin
      )
    }
    it("(1.j): should work on an instantiable container, annotating an instance") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableInstance)
        mark(definition.instanceContainer.i0, "i0.i0")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddOneWithInstantiableInstance/i0:AddOne"
           |CHECK-NEXT: "tag":"i0.i0"
           |""".stripMargin
      )
    }
    it("(1.k): should work on an instantiable container, annotating an instantiable container's module") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableInstantiable)
        mark(definition.containerContainer.container.i0, "i0.i0")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddOneWithInstantiableInstantiable/i0:AddOne"
           |CHECK-NEXT: "tag":"i0.i0"
           |""".stripMargin
      )
    }
    it("(1.l): should work on public member which references public member of another instance") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableInstantiable)
        mark(definition.containerContainer.container.i0, "i0.i0")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddOneWithInstantiableInstantiable/i0:AddOne"
           |CHECK-NEXT: "tag":"i0.i0"
           |""".stripMargin
      )
    }
    it("(1.m): should work for targets on definition to have correct circuit name") {
      class Top extends Module {
        val definition = Definition(new AddOneWithAnnotation)
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddOneWithAnnotation>innerWire"
           |CHECK-NEXT: "tag":"innerWire"
           |""".stripMargin
      )
    }
    it("(1.n): should work on user-defined types that provide Lookupable") {
      class Top extends Module {
        val defn = Definition(new HasUserDefinedType)
        defn.simple.name should be("foo")
        defn.parameterized.value should be(List(1, 2, 3))
        mark(defn.simple.data, "data")
        mark(defn.simple.inst, "inst")
        mark(defn.parameterized.inst, "inst2")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasUserDefinedType>wire"
           |CHECK-NEXT: "tag":"data"
           |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasUserDefinedType/inst0:AddOne"
           |CHECK-NEXT: "tag":"inst"
           |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasUserDefinedType/inst1:AddOne"
           |CHECK-NEXT: "tag":"inst2"
           |""".stripMargin
      )
    }
  }
  describe("(2): Annotations on designs not in the same chisel compilation") {
    // Extract the built `AddTwo` module for use in other tests.
    val first = ChiselStage.getModule(new AddTwo)
    it("(2.a): should work on an innerWire, marked in a different compilation") {
      class Top(x: AddTwo) extends Module {
        val parent = Definition(new ViewerParent(x, false, true))
      }
      generateFirrtlAndFileCheck(new Top(first))(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~AddTwo|AddTwo/i0:AddOne>innerWire"
           |CHECK-NEXT: "tag":"first"
           |""".stripMargin
      )
    }
    it("(2.b): should work on an innerWire, marked in a different compilation, in instanced instantiable") {
      class Top(x: AddTwo) extends Module {
        val parent = Definition(new ViewerParent(x, true, false))
      }
      generateFirrtlAndFileCheck(new Top(first))(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~AddTwo|AddTwo/i0:AddOne>innerWire"
           |CHECK-NEXT: "tag":"second"
           |""".stripMargin
      )
    }
    it("(2.c): should work on an innerWire, marked in a different compilation, in instanced module") {
      class Top(x: AddTwo) extends Module {
        val parent = Definition(new ViewerParent(x, false, false))
        mark(parent.viewer.x.i0.innerWire, "third")
      }
      generateFirrtlAndFileCheck(new Top(first))(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~AddTwo|AddTwo/i0:AddOne>innerWire"
           |CHECK-NEXT: "tag":"third"
           |""".stripMargin
      )
    }
  }
  describe("(3): @public") {
    it("(3.a): should work on multi-vals") {
      class Top() extends Module {
        val mv = Definition(new MultiVal())
        mark(mv.x, "mv.x")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|MultiVal>x"
           |CHECK-NEXT: "tag":"mv.x"
           |""".stripMargin
      )
    }
    it("(3.b): should work on lazy vals") {
      class Top() extends Module {
        val lv = Definition(new LazyVal())
        mark(lv.x, lv.y)
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|LazyVal>x"
           |CHECK-NEXT: "tag":"Hi"
           |""".stripMargin
      )
    }
    it("(3.c): should work on islookupables") {
      class Top() extends Module {
        val p = Parameters("hi", 0)
        val up = Definition(new UsesParameters(p))
        mark(up.x, up.y.string + up.y.int)
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|UsesParameters>x"
           |CHECK-NEXT: "tag":"hi0"
           |""".stripMargin
      )
    }
    it("(3.d): should work on lists") {
      class Top() extends Module {
        val i = Definition(new HasList())
        mark(i.x(1), i.y(1).toString)
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasList>x_1"
           |CHECK-NEXT: "tag":"2"
           |""".stripMargin
      )
    }
    it("(3.e): should work on seqs") {
      class Top() extends Module {
        val i = Definition(new HasSeq())
        mark(i.x(1), i.y(1).toString)
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasSeq>x_1"
           |CHECK-NEXT: "tag":"2"
           |""".stripMargin
      )
    }
    it("(3.f): should work on options") {
      class Top() extends Module {
        val i = Definition(new HasOption())
        i.x.map(x => mark(x, "x"))
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasOption>x"
           |CHECK-NEXT: "tag":"x"
           |""".stripMargin
      )
    }
    it("(3.g): should work on vecs") {
      class Top() extends Module {
        val i = Definition(new HasVec())
        mark(i.x, "blah")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasVec>x"
           |CHECK-NEXT: "tag":"blah"
           |""".stripMargin
      )
    }
    it("(3.h): should work on statically indexed vectors external to module") {
      class Top() extends Module {
        val i = Definition(new HasVec())
        mark(i.x(1), "blah")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasVec>x[1]"
           |CHECK-NEXT: "tag":"blah"
           |""".stripMargin
      )
    }
    it("(3.i): should work on statically indexed vectors internal to module") {
      class Top() extends Module {
        val i = Definition(new HasIndexedVec())
        mark(i.y, "blah")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasIndexedVec>x[1]"
           |CHECK-NEXT: "tag":"blah"
           |""".stripMargin
      )
    }
    ignore("(3.j): should work on vals in constructor arguments") {
      class Top() extends Module {
        val i = Definition(new HasPublicConstructorArgs(10))
        // mark(i.x, i.int.toString)
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasPublicConstructorArgs>x"
           |CHECK-NEXT: "tag":"10"
           |""".stripMargin
      )
    }
    it("(3.k): should work on unimplemented vals in abstract classes/traits") {
      class Top() extends Module {
        val i = Definition(new ConcreteHasBlah())
        def f(d: Definition[HasBlah]): Unit = {
          mark(d, d.blah.toString)
        }
        f(i)
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|ConcreteHasBlah"
           |CHECK-NEXT: "tag":"10"
           |""".stripMargin
      )
    }
    it("(3.l): should work on eithers") {
      class Top() extends Module {
        val i = Definition(new HasEither())
        i.x.map(x => mark(x, "xright")).left.map(x => mark(x, "xleft"))
        i.y.map(x => mark(x, "yright")).left.map(x => mark(x, "yleft"))
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasEither>x"
           |CHECK-NEXT: "tag":"xright"
           |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasEither>y"
           |CHECK-NEXT: "tag":"yleft"
           |""".stripMargin
      )
    }
    it("(3.m): should work on tuple2") {
      class Top() extends Module {
        val i = Definition(new HasTuple2())
        mark(i.xy._1, "x")
        mark(i.xy._2, "y")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasTuple2>x"
           |CHECK-NEXT: "tag":"x"
           |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasTuple2>y"
           |CHECK-NEXT: "tag":"y"
           |""".stripMargin
      )
    }
    it("(3.n): should work on Mems/SyncReadMems") {
      class Top() extends Module {
        val i = Definition(new HasMems())
        mark(i.mem, "Mem")
        mark(i.syncReadMem, "SyncReadMem")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasMems>mem"
           |CHECK-NEXT: "tag":"Mem"
           |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasMems>syncReadMem"
           |CHECK-NEXT: "tag":"SyncReadMem"
           |""".stripMargin
      )
    }
    it("(3.o): should not create memory ports") {
      class Top() extends Module {
        val i = Definition(new HasMems())
        i.mem(0) := 100.U // should be illegal!
      }
      intercept[ChiselException] {
        ChiselStage.convert(new Top)
      }.getMessage should include(
        "Cannot create a memory port in a different module (Top) than where the memory is (HasMems)."
      )
    }
    it("(3.p): should work on HasTarget") {
      class Top() extends Module {
        val i = Definition(new HasHasTarget)
        mark(i.x, "x")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasHasTarget>sram_sram"
           |CHECK-NEXT: "tag":"x"
           |""".stripMargin
      )
    }
    it("(3.q): should work on Tuple5 with a Module in it") {
      class Top() extends Module {
        val defn = Definition(new HasTuple5())
        val (3, w: UInt, "hi", inst: Instance[AddOne], l) = defn.tup
        l should be(List(1, 2, 3))
        mark(w, "wire")
        mark(inst, "inst")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasTuple5>wire"
           |CHECK-NEXT: "tag":"wire"
           |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|HasTuple5/inst:AddOne"
           |CHECK-NEXT: "tag":"inst"
           |""".stripMargin
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
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddOne>innerWire"
           |CHECK-NEXT: "tag":"blah"
           |""".stripMargin
      )
    }
    it("(4.b): should work on seqs of modules") {
      class Top() extends Module {
        val is = Seq(Module(new AddTwo()), Module(new AddTwo())).map(_.toDefinition)
        mark(f(is), "blah")
      }
      def f(i: Seq[Definition[AddTwo]]): Data = i.head.i0.innerWire
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddTwo/i0:AddOne>innerWire"
           |CHECK-NEXT: "tag":"blah"
           |""".stripMargin
      )
    }
    it("(4.c): should work on options of modules") {
      class Top() extends Module {
        val is: Option[Definition[AddTwo]] = Some(Module(new AddTwo())).map(_.toDefinition)
        mark(f(is), "blah")
      }
      def f(i: Option[Definition[AddTwo]]): Data = i.get.i0.innerWire
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddTwo/i0:AddOne>innerWire"
           |CHECK-NEXT: "tag":"blah"
           |""".stripMargin
      )
    }
  }
  describe("(5): Absolute Targets should work as expected") {
    it("(5.a): toAbsoluteTarget on a port of a definition") {
      class Top() extends Module {
        val i = Definition(new AddTwo())
        amark(i.in, "blah")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddTwo>in"
           |CHECK-NEXT: "tag":"blah"
           |""".stripMargin
      )
    }
    it("(5.b): toAbsoluteTarget on a subinstance's data within a definition") {
      class Top() extends Module {
        val i = Definition(new AddTwo())
        amark(i.i0.innerWire, "blah")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddTwo/i0:AddOne>innerWire"
           |CHECK-NEXT: "tag":"blah"
           |""".stripMargin
      )
    }
    it("(5.c): toAbsoluteTarget on a submodule's data within a definition") {
      class Top() extends Module {
        val i = Definition(new AddTwoMixedModules())
        amark(i.i1.in, "blah")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|AddTwoMixedModules/i1:AddOne_1>in"
           |CHECK-NEXT: "tag":"blah"
           |""".stripMargin
      )
    }
    it("(5.d): toAbsoluteTarget on a submodule's data, in an aggregate, within a definition") {
      class Top() extends Module {
        val i = Definition(new InstantiatesHasVec())
        amark(i.i1.x.head, "blah")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|InstantiatesHasVec/i1:HasVec_1>x[0]"
           |CHECK-NEXT: "tag":"blah"
           |""".stripMargin
      )
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
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|ModuleWithCommonIntf>io.in"
           |CHECK-NEXT: "tag":"gotcha"
           |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|ModuleWithCommonIntf"
           |CHECK-NEXT: "tag":"inst"
           |""".stripMargin
      )
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
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|ModuleWithCommonIntf>io.in"
           |CHECK-NEXT: "tag":"gotcha"
           |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|ModuleWithCommonIntf>sum"
           |CHECK-NEXT: "tag":"also this"
           |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|ModuleWithCommonIntf"
           |CHECK-NEXT: "tag":"inst"
           |""".stripMargin
      )
    }
    it("(6.c): A BlackBox that implements an @instantiable trait should be instantiable as that trait") {
      class Top extends Module {
        val m: ModuleIntf = Module(new BlackBoxWithCommonIntf)
        val d: Definition[ModuleIntf] = m.toDefinition
        mark(d.io.in, "gotcha")
        mark(d, "module")
      }
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|BlackBoxWithCommonIntf>in"
           |CHECK-NEXT: "tag":"gotcha"
           |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|BlackBoxWithCommonIntf"
           |CHECK-NEXT: "tag":"module"
           |""".stripMargin
      )
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
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|ModuleWithCommonIntfY>io.in"
           |CHECK-NEXT: "tag":"foo"
           |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|BlackBoxWithCommonIntf>in"
           |CHECK-NEXT: "tag":"bar"
           |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|ModuleWithCommonIntfX>io.in"
           |CHECK-NEXT: "tag":"fizz"
           |""".stripMargin
      )
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
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|MyModule>out"
           |CHECK-NEXT: "tag":"out"
           |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|MyModule>in"
           |CHECK-NEXT: "tag":"foo"
           |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|MyModule>sum"
           |CHECK-NEXT: "tag":"bar"
           |
           |CHECK:      public module Top :
           |CHECK:        connect i.in, foo
           |CHECK:        connect bar, i.out
           |""".stripMargin
      )
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
      generateFirrtlAndFileCheck(new Top)(
        """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|MyModule>a"
           |CHECK-NEXT: "tag":"in"
           |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
           |CHECK-NEXT: "target":"~Top|MyModule>a.foo"
           |CHECK-NEXT: "tag":"in_bar"
           |
           |CHECK:      public module Top :
           |CHECK:        connect i.a, foo
           |CHECK:        connct bar, i.b.foo
           |""".stripMargin
      )
    }
  }
}
