// SPDX-License-Identifier: Apache-2.0

package chiselTests
package experimental.hierarchy

import chisel3._
import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance}
import chisel3.testing.scalatest.FileCheck
import chisel3.util.{Decoupled, DecoupledIO, Valid}
import chisel3.experimental.{attach, Analog}
import chisel3.stage.{ChiselGeneratorAnnotation, DesignAnnotation}
import circt.stage.ChiselStage
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

// TODO/Notes
// - In backport, clock/reset are not automatically assigned. I think this is fixed in 3.5
class InstanceSpec extends AnyFunSpec with Matchers with Utils with FileCheck {
  import Annotations._
  import Examples._
  describe("(0) Instance instantiation") {
    it("(0.a): name of an instance should be correct") {
      class Top extends Module {
        val definition = Definition(new AddOne)
        val i0 = Instance(definition)
      }
      val chirrtl = ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("inst i0 of AddOne")
    }
    it("(0.b): name of an instanceclone should not error") {
      class Top extends Module {
        val definition = Definition(new AddTwo)
        val i0 = Instance(definition)
        val i = i0.i0 // This should not error
      }
      val chirrtl = ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("inst i0 of AddTwo")
    }
    it("(0.c): accessing internal fields through non-generated means is hard to do") {
      class Top extends Module {
        val definition = Definition(new AddOne)
        val i0 = Instance(definition)
        // i0.lookup(_.in) // Uncommenting this line will give the following error:
        // "You are trying to access a macro-only API. Please use the @public annotation instead."
        i0.in
      }
      val chirrtl = ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("inst i0 of AddOne")
    }
    it("(0.d): ExtModules should be supported") {
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
      val chirrtl = ChiselStage.emitCHIRRTL(new Top)
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
      val chirrtl = ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("connect i0.realIn.aminusx, UInt<7>(0h7b)")
    }
    it("(0.f): access under when does not crash") {
      class Top extends Module {
        val definition = Definition(new AddTwo)
        val i0 = Instance(definition)
        when(true.B) {
          val i = i0.i0 // This should not error
        }
      }
      ChiselStage.emitCHIRRTL(new Top)
    }
    it("(0.g): access under layer does not crash") {
      object A extends layer.Layer(layer.LayerConfig.Extract())
      class Top extends Module {
        val definition = Definition(new AddTwo)
        val i0 = Instance(definition)
        layer.block(A) {
          val i = i0.i0 // This should not error
        }
        layer.block(A) {
          val i = i0.i1 // This should not error
        }
      }
      ChiselStage.emitCHIRRTL(new Top)
    }
  }
  describe("(1) Annotations on instances in same chisel compilation") {
    it("(1.a): should work on a single instance, annotating the instance") {
      class Top extends Module {
        val definition: Definition[AddOne] = Definition(new AddOne)
        val i0:         Instance[AddOne] = Instance(definition)
        mark(i0, "i0")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i0:AddOne"
             |CHECK-NEXT: "tag":"i0"
             |""".stripMargin
        )
    }
    it("(1.b): should work on a single instance, annotating an inner wire") {
      class Top extends Module {
        val definition: Definition[AddOne] = Definition(new AddOne)
        val i0:         Instance[AddOne] = Instance(definition)
        mark(i0.innerWire, "i0.innerWire")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i0:AddOne>innerWire"
             |CHECK-NEXT: "tag":"i0.innerWire"
             |""".stripMargin
        )
    }
    it("(1.c): should work on a two nested instances, annotating the instance") {
      class Top extends Module {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        val i0:         Instance[AddTwo] = Instance(definition)
        mark(i0.i0, "i0.i0")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i0:AddTwo/i0:AddOne"
             |CHECK-NEXT: "tag":"i0.i0"
             |""".stripMargin
        )
    }
    it("(1.d): should work on a two nested instances, annotating the inner wire") {
      class Top extends Module {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        val i0:         Instance[AddTwo] = Instance(definition)
        mark(i0.i0.innerWire, "i0.i0.innerWire")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i0:AddTwo/i0:AddOne>innerWire"
             |CHECK-NEXT: "tag":"i0.i0.innerWire"
             |""".stripMargin
        )
    }
    it("(1.e): should work on a nested module in an instance, annotating the module") {
      class Top extends Module {
        val definition: Definition[AddTwoMixedModules] = Definition(new AddTwoMixedModules)
        val i0:         Instance[AddTwoMixedModules] = Instance(definition)
        mark(i0.i1, "i0.i1")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i0:AddTwoMixedModules/i1:AddOne_1"
             |CHECK-NEXT: "tag":"i0.i1"
             |""".stripMargin
        )
    }
    it("(1.f): should work on an instantiable container, annotating a wire") {
      class Top extends Module {
        val definition: Definition[AddOneWithInstantiableWire] = Definition(new AddOneWithInstantiableWire)
        val i0:         Instance[AddOneWithInstantiableWire] = Instance(definition)
        mark(i0.wireContainer.innerWire, "i0.innerWire")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i0:AddOneWithInstantiableWire>innerWire"
             |CHECK-NEXT: "tag":"i0.innerWire"
             |""".stripMargin
        )
    }
    it("(1.g): should work on an instantiable container, annotating a module") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableModule)
        val i0 = Instance(definition)
        mark(i0.moduleContainer.i0, "i0.i0")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i0:AddOneWithInstantiableModule/i0:AddOne"
             |CHECK-NEXT: "tag":"i0.i0"
             |""".stripMargin
        )
    }
    it("(1.h): should work on an instantiable container, annotating an instance") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableInstance)
        val i0 = Instance(definition)
        mark(i0.instanceContainer.i0, "i0.i0")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i0:AddOneWithInstantiableInstance/i0:AddOne"
             |CHECK-NEXT: "tag":"i0.i0"
             |""".stripMargin
        )
    }
    it("(1.i): should work on an instantiable container, annotating an instantiable container's module") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableInstantiable)
        val i0 = Instance(definition)
        mark(i0.containerContainer.container.i0, "i0.i0")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i0:AddOneWithInstantiableInstantiable/i0:AddOne"
             |CHECK-NEXT: "tag":"i0.i0"
             |""".stripMargin
        )
    }
    it("(1.j): should work on public member which references public member of another instance") {
      class Top extends Module {
        val definition = Definition(new AddOneWithInstantiableInstantiable)
        val i0 = Instance(definition)
        mark(i0.containerContainer.container.i0, "i0.i0")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i0:AddOneWithInstantiableInstantiable/i0:AddOne"
             |CHECK-NEXT: "tag":"i0.i0"
             |""".stripMargin
        )
    }
    it("(1.k): should work for targets on definition to have correct circuit name") {
      class Top extends Module {

        val definition = Definition(new AddOneWithAnnotation)
        val i0 = Instance(definition)
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
    it("(1.l): should work on things with type parameters") {
      class Top extends Module {
        val definition = Definition(new HasTypeParams[UInt](UInt(3.W)))
        val i0 = Instance(definition)
        mark(i0.blah, "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i0:HasTypeParams>blah"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(1.m): should work on Analog wires") {
      class Top extends Module {
        val port = IO(Analog(8.W))
        val definition = Definition(new HasAnalogWire)
        val i0 = Instance(definition)
        attach(port, i0.port)
        mark(i0.wire, "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i0:HasAnalogWire>wire"
             |CHECK-NEXT: "tag":"blah"
             |
             |CHECK:      public module Top :
             |CHECK:        attach (port, i0.port)
             |""".stripMargin
        )
    }
    it("(1.n): should work on user-defined types that provide Lookupable") {
      class Top extends Module {
        val definition = Definition(new HasUserDefinedType)
        val i0 = Instance(definition)
        i0.simple.name should be("foo")
        i0.parameterized.value should be(List(1, 2, 3))
        mark(i0.simple.data, "data")
        mark(i0.simple.inst, "inst")
        mark(i0.parameterized.inst, "inst2")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i0:HasUserDefinedType>wire"
             |CHECK-NEXT: "tag":"data"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i0:HasUserDefinedType/inst0:AddOne"
             |CHECK-NEXT: "tag":"inst"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i0:HasUserDefinedType/inst1:AddOne"
             |CHECK-NEXT: "tag":"inst2"
             |""".stripMargin
        )
    }
  }
  describe("(2) Annotations on designs not in the same chisel compilation") {
    // Extract the built `AddTwo` module for use in other tests.
    val first = {
      var result: AddTwo = null
      ChiselStage.emitCHIRRTL {
        result = new AddTwo
        result
      }
      result
    }
    it("(2.a): should work on an innerWire, marked in a different compilation") {
      class Top(x: AddTwo) extends Module {
        val parent = Instance(Definition(new ViewerParent(x, false, true)))
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
        val parent = Instance(Definition(new ViewerParent(x, true, false)))
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
        val d = Definition(new ViewerParent(x, false, false))
        val parent = Instance(d)
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
  describe("(3) @public") {
    it("(3.a): should work on multi-vals") {
      class Top() extends Module {
        val mv = Instance(Definition(new MultiVal()))
        mark(mv.x, "mv.x")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/mv:MultiVal>x"
             |CHECK-NEXT: "tag":"mv.x"
             |""".stripMargin
        )
    }
    it("(3.b): should work on lazy vals") {
      class Top() extends Module {
        val lv = Instance(Definition(new LazyVal()))
        mark(lv.x, lv.y)
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/lv:LazyVal>x"
             |CHECK-NEXT: "tag":"Hi"
             |""".stripMargin
        )
    }
    it("(3.c): should work on islookupables") {
      class Top() extends Module {
        val p = Parameters("hi", 0)
        val up = Instance(Definition(new UsesParameters(p)))
        mark(up.x, up.y.string + up.y.int)
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/up:UsesParameters>x"
             |CHECK-NEXT: "tag":"hi0"
             |""".stripMargin
        )
    }
    it("(3.d): should work on lists") {
      class Top() extends Module {
        val i = Instance(Definition(new HasList()))
        mark(i.x(1), i.y(1).toString)
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasList>x_1"
             |CHECK-NEXT: "tag":"2"
             |""".stripMargin
        )
    }
    it("(3.e): should work on seqs") {
      class Top() extends Module {
        val i = Instance(Definition(new HasSeq()))
        mark(i.x(1), i.y(1).toString)
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasSeq>x_1"
             |CHECK-NEXT: "tag":"2"
             |""".stripMargin
        )
    }
    it("(3.f): should work on options") {
      class Top() extends Module {
        val i = Instance(Definition(new HasOption()))
        i.x.map(x => mark(x, "x"))
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasOption>x"
             |CHECK-NEXT: "tag":"x"
             |""".stripMargin
        )
    }
    it("(3.g): should work on vecs") {
      class Top() extends Module {
        val i = Instance(Definition(new HasVec()))
        mark(i.x, "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasVec>x"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(3.h): should work on statically indexed vectors external to module") {
      class Top() extends Module {
        val i = Instance(Definition(new HasVec()))
        mark(i.x(1), "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasVec>x[1]"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(3.i): should work on statically indexed vectors internal to module") {
      class Top() extends Module {
        val i = Instance(Definition(new HasIndexedVec()))
        mark(i.y, "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasIndexedVec>x[1]"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasSubFieldAccess>in.valid"
             |CHECK-NEXT: "tag":"valid"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasSubFieldAccess>in.bits"
             |CHECK-NEXT: "tag":"bits"
             |
             |CHECK:      public module Top :
             |CHECK:        connect i.in.valid, input.valid
             |CHECK:        connect i.in.bits, input.bits
             |""".stripMargin
        )
    }
    ignore("(3.k): should work on vals in constructor arguments") {
      class Top() extends Module {
        val i = Instance(Definition(new HasPublicConstructorArgs(10)))
        // mark(i.x, i.int.toString)
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasPublicConstructorArgs>x"
             |CHECK-NEXT: "tag":"10"
             |""".stripMargin
        )
    }
    it("(3.l): should work on eithers") {
      class Top() extends Module {
        val i = Instance(Definition(new HasEither()))
        i.x.map(x => mark(x, "xright")).left.map(x => mark(x, "xleft"))
        i.y.map(x => mark(x, "yright")).left.map(x => mark(x, "yleft"))
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasEither>x"
             |CHECK-NEXT: "tag":"xright"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasEither>y"
             |CHECK-NEXT: "tag":"yleft"
             |""".stripMargin
        )
    }
    it("(3.m): should work on tuple2") {
      class Top() extends Module {
        val i = Instance(Definition(new HasTuple2()))
        mark(i.xy._1, "x")
        mark(i.xy._2, "y")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasTuple2>x"
             |CHECK-NEXT: "tag":"x"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasTuple2>y"
             |CHECK-NEXT: "tag":"y"
             |""".stripMargin
        )
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
        // @public private val privateVal = 10
        // This errors
        // @public protected val protectedVal = 10
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasMems>mem"
             |CHECK-NEXT: "tag":"Mem"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasMems>syncReadMem"
             |CHECK-NEXT: "tag":"SyncReadMem"
             |""".stripMargin
        )
    }
    it("(3.p): should make connectable IOs on nested IsInstantiables that have IO Datas in them") {
      ChiselStage
        .emitCHIRRTL(new AddTwoNestedInstantiableData(4))
        .fileCheck()(
          """|CHECK-COUNT-3: connect i1.in, i0.out
             |CHECK-NOT:     connect i1.in, i0.out
             |""".stripMargin
        )
    }
    it(
      "(3.q): should make connectable IOs on nested IsInstantiables's Data when the Instance and Definition do not have the same parent"
    ) {
      ChiselStage
        .emitCHIRRTL(new AddTwoNestedInstantiableDataWrapper(4))
        .fileCheck()("""|CHECK-COUNT-3: connect i1.in, i0.out
                        |CHECK-NOT:     connect i1.in, i0.out
                        |""".stripMargin)
    }
    it("(3.r): should work on HasTarget") {
      class Top() extends Module {
        val i = Instance(Definition(new HasHasTarget))
        mark(i.x, "x")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasHasTarget>sram_sram"
             |CHECK-NEXT: "tag":"x"
             |""".stripMargin
        )
    }
    it("(3.s): should work on Unit") {
      class Top extends Module {
        val i = Instance(Definition(new HasPublicUnit))
        i.x should be(())
        mark(i.y._1, "y_1")
        i.y._2 should be(())
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasPublicUnit>y_1"
             |CHECK-NEXT: "tag":"y_1"
             |""".stripMargin
        )
    }
    it("(3.t): should work on Tuple5 with a Module in it") {
      class Top() extends Module {
        val i = Instance(Definition(new HasTuple5()))
        val (3, w: UInt, "hi", inst: Instance[AddOne], l) = i.tup
        l should be(List(1, 2, 3))
        mark(w, "wire")
        mark(inst, "inst")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasTuple5>wire"
             |CHECK-NEXT: "tag":"wire"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasTuple5/inst:AddOne"
             |CHECK-NEXT: "tag":"inst"
             |""".stripMargin
        )
    }
    it("(3.u): should work on ActualDirection") {
      class Top extends Module {
        val i = Instance(Definition(new HasPublicActualDirection))
        i.inputDirection should be(ActualDirection.Input)
        i.outputDirection should be(ActualDirection.Output)
        i.bundleDirection should be(ActualDirection.Bidirectional(ActualDirection.Default))
      }
      ChiselStage.emitCHIRRTL(new Top)
    }
  }
  describe("(4) toInstance") {
    it("(4.a): should work on modules") {
      class Top() extends Module {
        val i = Module(new AddOne())
        f(i.toInstance)
      }
      def f(i: Instance[AddOne]): Unit = mark(i.innerWire, "blah")
      // TODO: Should this be ~|Top/i:AddOne>innerWire ???
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddOne>innerWire"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(4.b): should work on IsInstantiable") {
      class Top() extends Module {
        val i = Module(new AddTwo())
        val v = new Viewer(i, false)
        mark(f(v.toInstance), "blah")
      }
      def f(i: Instance[Viewer]): Data = i.x.i0.innerWire
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddTwo/i0:AddOne>innerWire"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(4.c): should work on seqs of modules") {
      class Top() extends Module {
        val is = Seq(Module(new AddTwo()), Module(new AddTwo())).map(_.toInstance)
        mark(f(is), "blah")
      }
      def f(i: Seq[Instance[AddTwo]]): Data = i.head.i0.innerWire
      // TODO: Should this be ~|Top... ??
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddTwo/i0:AddOne>innerWire"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(4.d): should work on seqs of IsInstantiable") {
      class Top() extends Module {
        val i = Module(new AddTwo())
        val vs = Seq(new Viewer(i, false), new Viewer(i, false)).map(_.toInstance)
        mark(f(vs), "blah")
      }
      def f(i: Seq[Instance[Viewer]]): Data = i.head.x.i0.innerWire
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddTwo/i0:AddOne>innerWire"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(4.e): should work on options of modules") {
      class Top() extends Module {
        val is: Option[Instance[AddTwo]] = Some(Module(new AddTwo())).map(_.toInstance)
        mark(f(is), "blah")
      }
      def f(i: Option[Instance[AddTwo]]): Data = i.get.i0.innerWire
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
  describe("(5) Absolute Targets should work as expected") {
    it("(5.a): toAbsoluteTarget on a port of an instance") {
      class Top() extends Module {
        val i = Instance(Definition(new AddTwo()))
        amark(i.in, "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:AddTwo>in"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(5.b): toAbsoluteTarget on a subinstance's data within an instance") {
      class Top() extends Module {
        val i = Instance(Definition(new AddTwo()))
        amark(i.i0.innerWire, "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:AddTwo/i0:AddOne>innerWire"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(5.c): toAbsoluteTarget on a submodule's data within an instance") {
      class Top() extends Module {
        val i = Instance(Definition(new AddTwoMixedModules()))
        amark(i.i1.in, "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:AddTwoMixedModules/i1:AddOne_1>in"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(5.d): toAbsoluteTarget on a submodule's data, in an aggregate, within an instance") {
      class Top() extends Module {
        val i = Instance(Definition(new InstantiatesHasVec()))
        amark(i.i1.x.head, "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:InstantiatesHasVec/i1:HasVec_1>x[0]"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:InstantiatesHasVec/i1:HasVec_1>x[0].x"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(5.f): toAbsoluteTarget on a subinstance") {
      class Top() extends Module {
        val i = Instance(Definition(new AddTwo()))
        amark(i.i1, "blah")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:AddTwo/i1:AddOne"
             |CHECK-NEXT: "tag":"blah"
             |""".stripMargin
        )
    }
    it("(5.g): should work for absolute targets on definition to have correct circuit name") {
      class Top extends Module {
        val definition = Definition(new AddOneWithAbsoluteAnnotation)
        val i0 = Instance(definition)
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|AddOneWithAbsoluteAnnotation>innerWire"
             |CHECK-NEXT: "tag":"innerWire"
             |""".stripMargin
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
    class BlackBoxWithCommonIntf extends ExtModule with ModuleIntf

    it("(6.a): A Module that implements an @instantiable trait should be instantiable as that trait") {
      class Top extends Module {
        val i: Instance[ModuleIntf] = Instance(Definition(new ModuleWithCommonIntf))
        mark(i.io.in, "gotcha")
        mark(i, "inst")
      }
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:ModuleWithCommonIntf>io.in"
             |CHECK-NEXT: "tag":"gotcha"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:ModuleWithCommonIntf"
             |CHECK-NEXT: "tag":"inst"
             |""".stripMargin
        )
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:ModuleWithCommonIntf>io.in"
             |CHECK-NEXT: "tag":"gotcha"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:ModuleWithCommonIntf>sum"
             |CHECK-NEXT: "tag":"also this"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:ModuleWithCommonIntf"
             |CHECK-NEXT: "tag":"inst"
             |""".stripMargin
        )
    }
    it("(6.c): A BlackBox that implements an @instantiable trait should be instantiable as that trait") {
      class Top extends Module {
        val i: Instance[ModuleIntf] = Module(new BlackBoxWithCommonIntf).toInstance
        mark(i.io.in, "gotcha")
        mark(i, "module")
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
             |CHECK-NEXT: "target":"~|Top/insts_2:ModuleWithCommonIntfX>io.in"
             |CHECK-NEXT: "tag":"fizz"
             |""".stripMargin
        )
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyModule>out"
             |CHECK-NEXT: "tag":"out"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyModule>in"
             |CHECK-NEXT: "tag":"foo"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyModule>sum"
             |CHECK-NEXT: "tag":"bar"
             |
             |CHECK:      public module Top :
             |CHECK:        connect i.in, foo
             |CHECK:        connect bar, i.out
             |""".stripMargin
        )
    }

    ignore("(7.b): should work on Aggregate Views") {
      import chiselTests.experimental.FlatDecoupledDataView._
      type RegDecoupled = DecoupledIO[FizzBuzz]
      val RegDecoupled = Decoupled
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
        val foo = IO(Flipped(RegDecoupled(new FizzBuzz)))
        val bar = IO(RegDecoupled(new FizzBuzz))
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyModule>a"
             |CHECK-NEXT: "tag":"enq"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyModule>a.fizz"
             |CHECK-NEXT: "tag":"enq.bits"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyModule>a.buzz"
             |CHECK-NEXT: "tag":"enq.bits"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyModule>b.fizz"
             |CHECK-NEXT: "tag":"deq.bits.fizz"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyModule>a.valid"
             |CHECK-NEXT: "tag":"enq_valid"
             |
             |CHECK:      public module Top :
             |CHECK:        connect i.a.valid, foo.valid
             |CHECK:        connect foo.ready, i.a.ready
             |CHECK:        connect i.a.fizz, foo.bits.fizz
             |CHECK:        connect i.a.buzz, foo.bits.buzz
             |CHECK:        connect bar.valid, i.b.valid
             |CHECK:        connect i.b.ready, bar.ready
             |CHECK:        connect bar.bits.fizz, i.b.fizz
             |CHECK:        connect bar.bits.buzz, i.b.buzz
             |""".stripMargin
        )
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyModule>a"
             |CHECK-NEXT: "tag":"in"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyModule>b.foo"
             |CHECK-NEXT: "tag":"out_bar"
             |
             |CHECK:      public module Top :
             |CHECK:        connect i.a, foo
             |CHECK:        connect bar.bar, i.b.foo
             |""".stripMargin
        )
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyModule>a"
             |CHECK-NEXT: "tag":"i.ports"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyModule>b"
             |CHECK-NEXT: "tag":"i.ports"
             |
             |CHECK:      public module Top :
             |CHECK:        connect i.a, foo
             |CHECK:        connect bar, i.b
             |""".stripMargin
        )
    }

    it("(7.e): should work on Views of ExtModules") {
      @instantiable
      class MyBlackBox extends ExtModule {
        @public val io = FlatIO(new Bundle {
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyBlackBox>in"
             |CHECK-NEXT: "tag":"i.foo"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyBlackBox>out"
             |CHECK-NEXT: "tag":"i.bar"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyBlackBox>in"
             |CHECK-NEXT: "tag":"i.innerView.in"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:MyBlackBox>out"
             |CHECK-NEXT: "tag":"outerView.out"
             |
             |CHECK:      public module Top :
             |CHECK:        connect i.in, foo
             |CHECK:        connect bar, i.out
             |""".stripMargin
        )
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasCMAR/c:AggregatePortModule>io"
             |CHECK-NEXT: "tag":"c.io"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|HasCMAR/c:AggregatePortModule>io.out"
             |CHECK-NEXT: "tag":"c.io.out"
             |""".stripMargin
        )
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
      ChiselStage
        .emitCHIRRTL(new Top)
        .fileCheck()(
          """|CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasCMAR/c:AggregatePortModule>io"
             |CHECK-NEXT: "tag":"i.c.io"
             |CHECK:      "class":"chiselTests.experimental.hierarchy.Annotations$MarkAnnotation"
             |CHECK-NEXT: "target":"~|Top/i:HasCMAR/c:AggregatePortModule>io.out"
             |CHECK-NEXT: "tag":"i.c.io.out"
             |""".stripMargin
        )
    }
  }
  describe("(9) isA[..]") {
    it("(9.a): it should work on simple classes") {
      class Top extends Module {
        val d = Definition(new AddOne)
        require(d.isA[AddOne])
      }
      ChiselStage.emitCHIRRTL(new Top)
    }
    it("(9.b): it should not work on inner classes") {
      class InnerClass extends Module
      class Top extends Module {
        val d = Definition(new InnerClass)
        "require(d.isA[Module])" should compile // ensures that the test below is checking something useful
        "require(d.isA[InnerClass])" shouldNot compile
      }
      ChiselStage.emitCHIRRTL(new Top)
    }
    it("(9.c): it should work on super classes") {
      class InnerClass extends Module
      class Top extends Module {
        val d = Definition(new InnerClass)
        require(d.isA[Module])
      }
      ChiselStage.emitCHIRRTL(new Top)
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
      ChiselStage.emitCHIRRTL(new Top)
    }
    it("(9.e): it should ignore type parameters (even though it would be nice if it didn't)") {
      class Top extends Module {
        val d0: Definition[Module] = Definition(new HasTypeParams(Bool()))
        require(d0.isA[HasTypeParams[Bool]])
        require(d0.isA[HasTypeParams[_]])
        require(d0.isA[HasTypeParams[UInt]])
        require(!d0.isA[HasBlah])
      }
      ChiselStage.emitCHIRRTL(new Top)
    }
  }

  describe("(10) Select APIs") {
    it("(10.a): instancesOf") {
      val m = ChiselGeneratorAnnotation(() => {
        new AddTwoMixedModules
      }).elaborate(1).asInstanceOf[DesignAnnotation[AddTwoMixedModules]].design
      aop.Select.instancesOf[AddOne](m.toDefinition).map { (i: Instance[AddOne]) => i.toTarget } should be(
        Seq(
          "~|AddTwoMixedModules/i0:AddOne".it,
          "~|AddTwoMixedModules/i1:AddOne_1".it
        )
      )
    }
    it("(10.b): instancesIn") {
      val m = ChiselGeneratorAnnotation(() => {
        new AddTwoMixedModules
      }).elaborate(1).asInstanceOf[DesignAnnotation[AddTwoMixedModules]].design
      val insts = aop.Select.instancesIn(m.toDefinition)
      val abs = insts.map { (i: Instance[BaseModule]) => i.toAbsoluteTarget }
      val rel = insts.map { (i: Instance[BaseModule]) => i.toTarget }
      abs should be(
        Seq(
          "~|AddTwoMixedModules/i0:AddOne".it,
          "~|AddTwoMixedModules/i1:AddOne_1".it
        )
      )
      rel should be(
        Seq(
          "~|AddTwoMixedModules/i0:AddOne".it,
          "~|AddTwoMixedModules/i1:AddOne_1".it
        )
      )
    }
    it("(10.c): allInstancesOf") {
      val m = ChiselGeneratorAnnotation(() => {
        new AddFour
      }).elaborate(1).asInstanceOf[DesignAnnotation[AddFour]].design
      val insts = aop.Select.allInstancesOf[AddOne](m.toDefinition)
      val abs = insts.map { (i: Instance[AddOne]) => i.in.toAbsoluteTarget }
      val rel = insts.map { (i: Instance[AddOne]) => i.in.toTarget }
      rel should be(
        Seq(
          "~|AddFour/i0:AddTwoMixedModules/i0:AddOne>in".rt,
          "~|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>in".rt,
          "~|AddFour/i1:AddTwoMixedModules/i0:AddOne>in".rt,
          "~|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>in".rt
        )
      )
      abs should be(
        Seq(
          "~|AddFour/i0:AddTwoMixedModules/i0:AddOne>in".rt,
          "~|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>in".rt,
          "~|AddFour/i1:AddTwoMixedModules/i0:AddOne>in".rt,
          "~|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>in".rt
        )
      )
    }
    it("(10.d): definitionsOf") {
      val m = ChiselGeneratorAnnotation(() => {
        new AddTwoMixedModules
      }).elaborate(1).asInstanceOf[DesignAnnotation[AddTwoMixedModules]].design
      val targets = aop.Select.definitionsOf[AddOne](m.toDefinition).map { (i: Definition[AddOne]) => i.in.toTarget }
      targets should be(
        Seq(
          "~|AddOne>in".rt,
          "~|AddOne_1>in".rt
        )
      )
    }
    it("(10.e): definitionsIn") {
      val m = ChiselGeneratorAnnotation(() => {
        new AddTwoMixedModules
      }).elaborate(1).asInstanceOf[DesignAnnotation[AddTwoMixedModules]].design
      val targets = aop.Select.definitionsIn(m.toDefinition).map { (i: Definition[BaseModule]) => i.toTarget }
      targets should be(
        Seq(
          "~|AddOne".mt,
          "~|AddOne_1".mt
        )
      )
    }
    it("(10.f): allDefinitionsOf") {
      val m = ChiselGeneratorAnnotation(() => {
        new AddFour
      }).elaborate(1).asInstanceOf[DesignAnnotation[AddFour]].design
      val targets = aop.Select.allDefinitionsOf[AddOne](m.toDefinition).map { (i: Definition[AddOne]) => i.in.toTarget }
      targets should be(
        Seq(
          "~|AddOne>in".rt,
          "~|AddOne_1>in".rt
        )
      )
    }
    it("(10.g): Select.collectDeep should fail when combined with hierarchy package") {
      val m = ChiselGeneratorAnnotation(() => {
        new AddFour
      }).elaborate(1).asInstanceOf[DesignAnnotation[AddFour]].design
      intercept[Exception] { aop.Select.collectDeep(m) { case m: AddOne => m.toTarget } }
    }
    it("(10.h): Select.getDeep should fail when combined with hierarchy package") {
      val m = ChiselGeneratorAnnotation(() => {
        new AddFour
      }).elaborate(1).asInstanceOf[DesignAnnotation[AddFour]].design
      intercept[Exception] { aop.Select.getDeep(m) { (m: BaseModule) => Nil } }
    }
    it("(10.i): Select.instances should fail when combined with hierarchy package") {
      val m = ChiselGeneratorAnnotation(() => {
        new AddFour
      }).elaborate(1).asInstanceOf[DesignAnnotation[AddFour]].design
      intercept[Exception] { aop.Select.instances(m) }
    }
    it("(10.j): allInstancesOf.ios") {
      val m = ChiselGeneratorAnnotation(() => {
        new AddFour
      }).elaborate(1).asInstanceOf[DesignAnnotation[AddFour]].design
      val abs = aop.Select.allInstancesOf[AddOne](m.toDefinition).flatMap { (i: Instance[AddOne]) =>
        aop.Select.ios(i).map(_.toAbsoluteTarget)
      }
      val rel = aop.Select.allInstancesOf[AddOne](m.toDefinition).flatMap { (i: Instance[AddOne]) =>
        aop.Select.ios(i).map(_.toTarget)
      }
      abs should be(
        Seq(
          "~|AddFour/i0:AddTwoMixedModules/i0:AddOne>clock".rt,
          "~|AddFour/i0:AddTwoMixedModules/i0:AddOne>reset".rt,
          "~|AddFour/i0:AddTwoMixedModules/i0:AddOne>in".rt,
          "~|AddFour/i0:AddTwoMixedModules/i0:AddOne>out".rt,
          "~|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>clock".rt,
          "~|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>reset".rt,
          "~|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>in".rt,
          "~|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>out".rt,
          "~|AddFour/i1:AddTwoMixedModules/i0:AddOne>clock".rt,
          "~|AddFour/i1:AddTwoMixedModules/i0:AddOne>reset".rt,
          "~|AddFour/i1:AddTwoMixedModules/i0:AddOne>in".rt,
          "~|AddFour/i1:AddTwoMixedModules/i0:AddOne>out".rt,
          "~|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>clock".rt,
          "~|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>reset".rt,
          "~|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>in".rt,
          "~|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>out".rt
        )
      )
      rel should be(
        Seq(
          "~|AddFour/i0:AddTwoMixedModules/i0:AddOne>clock".rt,
          "~|AddFour/i0:AddTwoMixedModules/i0:AddOne>reset".rt,
          "~|AddFour/i0:AddTwoMixedModules/i0:AddOne>in".rt,
          "~|AddFour/i0:AddTwoMixedModules/i0:AddOne>out".rt,
          "~|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>clock".rt,
          "~|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>reset".rt,
          "~|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>in".rt,
          "~|AddFour/i0:AddTwoMixedModules/i1:AddOne_1>out".rt,
          "~|AddFour/i1:AddTwoMixedModules/i0:AddOne>clock".rt,
          "~|AddFour/i1:AddTwoMixedModules/i0:AddOne>reset".rt,
          "~|AddFour/i1:AddTwoMixedModules/i0:AddOne>in".rt,
          "~|AddFour/i1:AddTwoMixedModules/i0:AddOne>out".rt,
          "~|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>clock".rt,
          "~|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>reset".rt,
          "~|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>in".rt,
          "~|AddFour/i1:AddTwoMixedModules/i1:AddOne_1>out".rt
        )
      )
    }
    it("(10.k): allDefinitionsOf.ios") {
      val m = ChiselGeneratorAnnotation(() => {
        new AddFour
      }).elaborate(1).asInstanceOf[DesignAnnotation[AddFour]].design
      val abs = aop.Select.allDefinitionsOf[AddOne](m.toDefinition).flatMap { (i: Definition[AddOne]) =>
        aop.Select.ios(i).map(_.toAbsoluteTarget)
      }
      val rel = aop.Select.allDefinitionsOf[AddOne](m.toDefinition).flatMap { (i: Definition[AddOne]) =>
        aop.Select.ios(i).map(_.toTarget)
      }
      abs should be(
        Seq(
          "~|AddOne>clock".rt,
          "~|AddOne>reset".rt,
          "~|AddOne>in".rt,
          "~|AddOne>out".rt,
          "~|AddOne_1>clock".rt,
          "~|AddOne_1>reset".rt,
          "~|AddOne_1>in".rt,
          "~|AddOne_1>out".rt
        )
      )
      rel should be(
        Seq(
          "~|AddOne>clock".rt,
          "~|AddOne>reset".rt,
          "~|AddOne>in".rt,
          "~|AddOne>out".rt,
          "~|AddOne_1>clock".rt,
          "~|AddOne_1>reset".rt,
          "~|AddOne_1>in".rt,
          "~|AddOne_1>out".rt
        )
      )
    }
    it("(10.l): Select.instancesIn for typed BaseModules") {
      val m = ChiselGeneratorAnnotation(() => {
        new HasMultipleTypeParamsInside
      }).elaborate(1).asInstanceOf[DesignAnnotation[HasMultipleTypeParamsInside]].design
      val targets = aop.Select.instancesIn(m.toDefinition).map { (i: Instance[BaseModule]) => i.toTarget }
      targets should be(
        Seq(
          "~|HasMultipleTypeParamsInside/i00:HasTypeParams".it,
          "~|HasMultipleTypeParamsInside/i01:HasTypeParams".it,
          "~|HasMultipleTypeParamsInside/i10:HasTypeParams_1".it,
          "~|HasMultipleTypeParamsInside/i11:HasTypeParams_1".it
        )
      )

    }
    it("(10.m): Select.instancesOf for typed BaseModules if type is ignored") {
      val m = ChiselGeneratorAnnotation(() => {
        new HasMultipleTypeParamsInside
      }).elaborate(1).asInstanceOf[DesignAnnotation[HasMultipleTypeParamsInside]].design
      val targets =
        aop.Select.instancesOf[HasTypeParams[_]](m.toDefinition).map { (i: Instance[HasTypeParams[_]]) => i.toTarget }
      targets should be(
        Seq(
          "~|HasMultipleTypeParamsInside/i00:HasTypeParams".it,
          "~|HasMultipleTypeParamsInside/i01:HasTypeParams".it,
          "~|HasMultipleTypeParamsInside/i10:HasTypeParams_1".it,
          "~|HasMultipleTypeParamsInside/i11:HasTypeParams_1".it
        )
      )
    }
    it(
      "10.n Select.instancesOf for typed BaseModules even type is specified wrongly (should be ignored, even though we wish it weren't)"
    ) {
      val m = ChiselGeneratorAnnotation(() => {
        new HasMultipleTypeParamsInside
      }).elaborate(1).asInstanceOf[DesignAnnotation[HasMultipleTypeParamsInside]].design
      val targets = aop.Select.instancesOf[HasTypeParams[SInt]](m.toDefinition).map { (i: Instance[HasTypeParams[_]]) =>
        i.toTarget
      }
      targets should be(
        Seq(
          "~|HasMultipleTypeParamsInside/i00:HasTypeParams".it,
          "~|HasMultipleTypeParamsInside/i01:HasTypeParams".it,
          "~|HasMultipleTypeParamsInside/i10:HasTypeParams_1".it,
          "~|HasMultipleTypeParamsInside/i11:HasTypeParams_1".it
        )
      )
    }
  }
  describe("(11) .suggestName") {
    it("11.1 suggestName for Instances") {
      class Top extends Module {
        val definition = Definition(new AddOne)
        val inst0 = Instance(definition)
        val inst1 = Module(new AddOne).toInstance
        inst0.suggestName("potato")
        inst1.suggestName("potato")
      }
      val chirrtl = ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("inst potato of AddOne")
      chirrtl should include("inst potato_1 of AddOne_1")
    }
    it("11.2 suggestName at instantiation") {
      class Top extends Module {
        val k = Instance(Definition(new AddOne)).suggestName("potato")
      }
      val chirrtl = ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("inst potato of AddOne")
    }
    it("11.3 suggestName with sanitization") {
      class Top extends Module {
        val definition = Definition(new AddOne)
        val inst0 = Instance(definition)
        val inst1 = Instance(definition)
        inst0.suggestName("potato")
        inst1.suggestName("potato")
      }
      val chirrtl = ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("inst potato of AddOne")
      chirrtl should include("inst potato_1 of AddOne")
    }
    it("11.4 suggestName with multi-def collision sanitization") {
      class Top extends Module {
        val potato = Wire(UInt(8.W))
        val inst0 = Module(new AddOne()).suggestName("potato")
        val inst1 = Instance(Definition(new AddOne)).suggestName("potato")
      }
      val chirrtl = ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("wire potato : UInt<8>")
      chirrtl should include("inst potato_1 of AddOne")
      chirrtl should include("inst potato_2 of AddOne_1")
    }
  }
}
