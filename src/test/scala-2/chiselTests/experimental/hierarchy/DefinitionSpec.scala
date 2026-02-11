// SPDX-License-Identifier: Apache-2.0

package chiselTests
package experimental.hierarchy

import chisel3._
import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance}
import chisel3.testing.scalatest.FileCheck
import chiselTests.experimental.ExtensionMethods.ChiselStageHelpers
import circt.stage.ChiselStage
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

// TODO/Notes
// - In backport, clock/reset are not automatically assigned. I think this is fixed in 3.5
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddOne"
             |CHECK-NEXT: "tag":"mark"
             |""".stripMargin
        )
    }
    it("(1.b): should work on a single definition, annotating an inner wire") {
      class Top extends Module {
        val definition: Definition[AddOne] = Definition(new AddOne)
        mark(definition.innerWire, "i0.innerWire")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddOne>innerWire"
             |CHECK-NEXT: "tag":"i0.innerWire"
             |""".stripMargin
        )
    }
    it("(1.c): should work on a two nested definitions, annotating the definition") {
      class Top extends Module {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        mark(definition.definition, "i0.i0")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddOne"
             |CHECK-NEXT: "tag":"i0.i0"
             |""".stripMargin
        )
    }
    it("(1.d): should work on an instance in a definition, annotating the instance") {
      class Top extends Module {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        mark(definition.i0, "i0.i0")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddTwo/i0:AddOne"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddOne"
             |CHECK-NEXT: "tag":"i0.i0"
             |""".stripMargin
        )
    }
    it("(1.f): should work on a wire in an instance in a definition") {
      class Top extends Module {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        mark(definition.i0.innerWire, "i0.i0.innerWire")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddTwo/i0:AddOne>innerWire"
             |CHECK-NEXT: "tag":"i0.i0.innerWire"
             |""".stripMargin
        )
    }
    it("(1.g): should work on a nested module in a definition, annotating the module") {
      class Top extends Module {
        val definition: Definition[AddTwoMixedModules] = Definition(new AddTwoMixedModules)
        mark(definition.i1, "i0.i1")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddTwoMixedModules/i1:AddOne_1"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddOneWithInstantiableWire>innerWire"
             |CHECK-NEXT: "tag":"i0.innerWire"
             |""".stripMargin
        )
    }
    it("(1.i): should work on an instantiable container, annotating a module") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableModule)
        mark(definition.moduleContainer.i0, "i0.i0")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddOneWithInstantiableModule/i0:AddOne"
             |CHECK-NEXT: "tag":"i0.i0"
             |""".stripMargin
        )
    }
    it("(1.j): should work on an instantiable container, annotating an instance") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableInstance)
        mark(definition.instanceContainer.i0, "i0.i0")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddOneWithInstantiableInstance/i0:AddOne"
             |CHECK-NEXT: "tag":"i0.i0"
             |""".stripMargin
        )
    }
    it("(1.k): should work on an instantiable container, annotating an instantiable container's module") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableInstantiable)
        mark(definition.containerContainer.container.i0, "i0.i0")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddOneWithInstantiableInstantiable/i0:AddOne"
             |CHECK-NEXT: "tag":"i0.i0"
             |""".stripMargin
        )
    }
    it("(1.l): should work on public member which references public member of another instance") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableInstantiable)
        mark(definition.containerContainer.container.i0, "i0.i0")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddOneWithInstantiableInstantiable/i0:AddOne"
             |CHECK-NEXT: "tag":"i0.i0"
             |""".stripMargin
        )
    }
    it("(1.m): should work for targets on definition to have correct circuit name") {
      class Top extends Module {
        val definition = Definition(new AddOneWithAnnotation)
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddOneWithAnnotation>innerWire"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasUserDefinedType>wire"
             |CHECK-NEXT: "tag":"data"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasUserDefinedType/inst0:AddOne"
             |CHECK-NEXT: "tag":"inst"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasUserDefinedType/inst1:AddOne"
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
      ChiselStage
        .emitCHIRRTL(new Top(first))
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddTwo/i0:AddOne>innerWire"
             |CHECK-NEXT: "tag":"first"
             |""".stripMargin
        )
    }
    it("(2.b): should work on an innerWire, marked in a different compilation, in instanced instantiable") {
      class Top(x: AddTwo) extends Module {
        val parent = Definition(new ViewerParent(x, true, false))
      }
      ChiselStage
        .emitCHIRRTL(new Top(first))
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddTwo/i0:AddOne>innerWire"
             |CHECK-NEXT: "tag":"second"
             |""".stripMargin
        )
    }
    it("(2.c): should work on an innerWire, marked in a different compilation, in instanced module") {
      class Top(x: AddTwo) extends Module {
        val parent = Definition(new ViewerParent(x, false, false))
        mark(parent.viewer.x.i0.innerWire, "third")
      }
      ChiselStage
        .emitCHIRRTL(new Top(first))
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddTwo/i0:AddOne>innerWire"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|MultiVal>x"
             |CHECK-NEXT: "tag":"mv.x"
             |""".stripMargin
        )
    }
    it("(3.b): should work on lazy vals") {
      class Top() extends Module {
        val lv = Definition(new LazyVal())
        mark(lv.x, lv.y)
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|LazyVal>x"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|UsesParameters>x"
             |CHECK-NEXT: "tag":"hi0"
             |""".stripMargin
        )
    }
    it("(3.d): should work on lists") {
      class Top() extends Module {
        val i = Definition(new HasList())
        mark(i.x(1), i.y(1).toString)
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasList>x_1"
             |CHECK-NEXT: "tag":"2"
             |""".stripMargin
        )
    }
    it("(3.e): should work on seqs") {
      class Top() extends Module {
        val i = Definition(new HasSeq())
        mark(i.x(1), i.y(1).toString)
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasSeq>x_1"
             |CHECK-NEXT: "tag":"2"
             |""".stripMargin
        )
    }
    it("(3.f): should work on options") {
      class Top() extends Module {
        val i = Definition(new HasOption())
        i.x.map(x => mark(x, "x"))
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasOption>x"
             |CHECK-NEXT: "tag":"x"
             |""".stripMargin
        )
    }
    it("(3.g): should work on vecs") {
      class Top() extends Module {
        val i = Definition(new HasVec())
        mark(i.x, "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasVec>x"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(3.h): should work on statically indexed vectors external to module") {
      class Top() extends Module {
        val i = Definition(new HasVec())
        mark(i.x(1), "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasVec>x[1]"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(3.i): should work on statically indexed vectors internal to module") {
      class Top() extends Module {
        val i = Definition(new HasIndexedVec())
        mark(i.y, "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasIndexedVec>x[1]"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    ignore("(3.j): should work on vals in constructor arguments") {
      class Top() extends Module {
        val i = Definition(new HasPublicConstructorArgs(10))
        // mark(i.x, i.int.toString)
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasPublicConstructorArgs>x"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|ConcreteHasBlah"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasEither>x"
             |CHECK-NEXT: "tag":"xright"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasEither>y"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasTuple2>x"
             |CHECK-NEXT: "tag":"x"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasTuple2>y"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasMems>mem"
             |CHECK-NEXT: "tag":"Mem"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasMems>syncReadMem"
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
        ChiselStage.elaborate(new Top)
      }.getMessage should include(
        "Cannot create a memory port in a different module (Top) than where the memory is (HasMems)."
      )
    }
    it("(3.p): should work on HasTarget") {
      class Top() extends Module {
        val i = Definition(new HasHasTarget)
        mark(i.x, "x")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasHasTarget>sram_sram"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasTuple5>wire"
             |CHECK-NEXT: "tag":"wire"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasTuple5/inst:AddOne"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddOne>innerWire"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddTwo/i0:AddOne>innerWire"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddTwo/i0:AddOne>innerWire"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddTwo>in"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(5.b): toAbsoluteTarget on a subinstance's data within a definition") {
      class Top() extends Module {
        val i = Definition(new AddTwo())
        amark(i.i0.innerWire, "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddTwo/i0:AddOne>innerWire"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(5.c): toAbsoluteTarget on a submodule's data within a definition") {
      class Top() extends Module {
        val i = Definition(new AddTwoMixedModules())
        amark(i.i1.in, "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddTwoMixedModules/i1:AddOne_1>in"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(5.d): toAbsoluteTarget on a submodule's data, in an aggregate, within a definition") {
      class Top() extends Module {
        val i = Definition(new InstantiatesHasVec())
        amark(i.i1.x.head, "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|InstantiatesHasVec/i1:HasVec_1>x[0]"
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
    class BlackBoxWithCommonIntf extends ExtModule with ModuleIntf

    it("(6.a): A Module that implements an @instantiable trait should be definable as that trait") {
      class Top extends Module {
        val i: Definition[ModuleIntf] = Definition(new ModuleWithCommonIntf)
        mark(i.io.in, "gotcha")
        mark(i, "inst")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|ModuleWithCommonIntf>io.in"
             |CHECK-NEXT: "tag":"gotcha"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|ModuleWithCommonIntf"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|ModuleWithCommonIntf>io.in"
             |CHECK-NEXT: "tag":"gotcha"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|ModuleWithCommonIntf>sum"
             |CHECK-NEXT: "tag":"also this"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|ModuleWithCommonIntf"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|BlackBoxWithCommonIntf>io.in"
             |CHECK-NEXT: "tag":"gotcha"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|BlackBoxWithCommonIntf"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|ModuleWithCommonIntfY>io.in"
             |CHECK-NEXT: "tag":"foo"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|BlackBoxWithCommonIntf>io.in"
             |CHECK-NEXT: "tag":"bar"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|ModuleWithCommonIntfX>io.in"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|MyModule>out"
             |CHECK-NEXT: "tag":"out"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|MyModule>in"
             |CHECK-NEXT: "tag":"foo"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|MyModule>sum"
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|MyModule>a"
             |CHECK-NEXT: "tag":"in"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|MyModule>a.foo"
             |CHECK-NEXT: "tag":"in_bar"
             |
             |CHECK:      public module Top :
             |CHECK:        connect i.a, foo
             |CHECK:        connct bar, i.b.foo
             |""".stripMargin
        )
    }
  }
  describe("(8): A Definition under an elideBlock scope") {
    it("(8.a): should elide layer blocks") {
      class Bar extends RawModule {
        layer.block(layers.Verification) {
          val a = WireInit(false.B)
        }
      }
      class Foo extends RawModule {
        layer.elideBlocks {
          val barDef = Definition(new Bar)
        }
      }
      ChiselStage
        .emitCHIRRTL(new Foo)
        .fileCheck()(
          """|CHECK: module Bar :
             |CHECK-NOT: layerblock
             |""".stripMargin
        )
    }
  }
  describe("(9): Calling .toDefinition should update Builder.definitions") {
    it("(9.a): calling .toDefinition from another definition's child Instance should work as expected") {
      class Bar extends RawModule {
        val a = WireInit(false.B)
      }
      @instantiable
      class Foo extends RawModule {
        @public val bar = Module(new Bar)
      }
      class Baz(d: Definition[Foo]) extends RawModule {
        val bar = Instance(d.bar.toDefinition)
      }
      ChiselStage
        .emitCHIRRTL(
          {
            val d = Definition(new Foo)
            new Baz(d)
          },
          Array("--full-stacktrace")
        )
        .fileCheck()(
          """|CHECK: module Bar :
             |CHECK: module Baz :
             |""".stripMargin
        )
    }
    it("(9.b): calling .toDefinition twice from anther definition's child Instance should work as expected") {
      class Bar extends RawModule {
        val a = WireInit(false.B)
      }
      @instantiable
      class Foo extends RawModule {
        @public val bar = Module(new Bar)
      }
      class Baz(d: Definition[Foo]) extends RawModule {
        val bar = Instance(d.bar.toDefinition)
        val bar2 = Instance(d.bar.toDefinition)
      }
      val x = ChiselStage
        .emitCHIRRTL(
          {
            val d = Definition(new Foo)
            new Baz(d)
          },
          Array("--full-stacktrace")
        )
        .fileCheck()(
          """|CHECK: module Bar :
             |CHECK-NOT: module Bar_
             |CHECK: module Baz :
             |""".stripMargin
        )
    }
    it(
      "(9.c): calling .toDefinition on an Instance of a module that was already imported from another Definition should work"
    ) {
      import chisel3.experimental.hierarchy.core.ImportDefinitionAnnotation
      import chisel3.stage.{ChiselGeneratorAnnotation, DesignAnnotation}

      // First elaboration: Create FooForImport with BarForImport inside
      val fooAnnos = ChiselGeneratorAnnotation(() => new FooForImport).elaborate
      val fooDef = fooAnnos.collectFirst { case DesignAnnotation(d: FooForImport, _) => d.toDefinition }.get
      // Also get the Bar definition from the first elaboration
      val barDef = fooDef.bar.toDefinition

      // Second elaboration: Create Baz that imports Foo's Definition and calls .toDefinition on bar
      class Baz(importedFooDef: Definition[FooForImport]) extends RawModule {
        val fooInst = Instance(importedFooDef)

        // This tests calling .toDefinition on an Instance's child module
        // when that Instance came from an imported Definition
        val barDefFromInst = importedFooDef.bar.toDefinition
        val barInst = Instance(barDefFromInst)
      }

      val annos = (new circt.stage.ChiselStage).execute(
        Array("--target", "chirrtl"),
        Seq(
          ChiselGeneratorAnnotation(() => new Baz(fooDef)),
          ImportDefinitionAnnotation(fooDef),
          ImportDefinitionAnnotation(barDef)
        )
      )

      // Both FooForImport and BarForImport should be extmodules (imported from previous elaboration)
      // Baz is the only actual module
      annos.collectFirst { case a: chisel3.stage.ChiselCircuitAnnotation => a.elaboratedCircuit.serialize }.get
        .fileCheck()(
          """|CHECK: extmodule FooForImport
             |CHECK: extmodule BarForImport
             |CHECK: module Baz :
             |CHECK: inst fooInst of FooForImport
             |CHECK: inst barInst of BarForImport
             |""".stripMargin
        )
    }
    it("(9.d): Definition equality should work correctly for deduplication") {
      // This test verifies that Definitions with the same proto are properly deduplicated
      // in Builder.definitions (which uses LinkedHashSet).
      //
      // Both definitionsIn and definitionsOf should create Definitions that can be
      // properly compared for equality when they reference the same underlying proto.
      import chisel3.aop.Select
      import chisel3.stage.{ChiselGeneratorAnnotation, DesignAnnotation}

      class TestHarness extends Module {
        val addOne = Module(new AddOne)
      }

      // Elaborate the design
      val dut = ChiselGeneratorAnnotation(() => {
        new TestHarness
      }).elaborate(1).asInstanceOf[DesignAnnotation[TestHarness]].design

      // Use Select.definitionsIn and Select.definitionsOf
      val definitionsFromIn = Select.definitionsIn(dut.toDefinition)
      val definitionsFromOf = Select.definitionsOf[AddOne](dut.toDefinition)

      // Both should find the same AddOne module
      definitionsFromIn.size should be(1)
      definitionsFromOf.size should be(1)

      // Both should be equal since they reference the same proto
      // (Both use Proto wrapper now, ensuring proper equality)
      (definitionsFromIn.head == definitionsFromOf.head) should be(true)
    }
  }

}
