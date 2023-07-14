// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental.hierarchy

import chisel3._
import chisel3.util.Valid
import chisel3.experimental.hierarchy._
import chisel3.experimental.BaseModule

object Examples {
  import Annotations._
  @instantiable
  class AddOne extends Module {
    @public val in = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val innerWire = Wire(UInt(32.W))
    innerWire := in + 1.U
    out := innerWire
  }
  @instantiable
  class AddOneWithAnnotation extends Module {
    @public val in = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val innerWire = Wire(UInt(32.W))
    mark(innerWire, "innerWire")
    innerWire := in + 1.U
    out := innerWire
  }
  @instantiable
  class AddOneWithAbsoluteAnnotation extends Module {
    @public val in = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val innerWire = Wire(UInt(32.W))
    amark(innerWire, "innerWire")
    innerWire := in + 1.U
    out := innerWire
  }
  @instantiable
  class AddOneParameterized(width: Int) extends Module {
    @public val in = IO(Input(UInt(width.W)))
    @public val out = IO(Output(UInt(width.W)))
    out := in + 1.U
  }
  class AddOneWithNested(width: Int) extends Module {
    @public val in = IO(Input(UInt(width.W)))
    @public val out = IO(Output(UInt(width.W)))
    val addOneDef = Seq.fill(3)(Definition(new AddOne))
    out := in + 1.U
  }
  @instantiable
  class AddOneBlackBox extends BlackBox {
    @public val io = IO(new Bundle {
      val in = Input(UInt(32.W))
      val out = Output(UInt(32.W))
    })
  }

  @instantiable
  class AddTwo extends Module {
    @public val in = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val definition = Definition(new AddOne)
    @public val i0: Instance[AddOne] = Instance(definition)
    @public val i1: Instance[AddOne] = Instance(definition)
    i0.in := in
    i1.in := i0.out
    out := i1.out
  }
  @instantiable
  class AddTwoMixedModules extends Module {
    @public val in = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    val definition = Definition(new AddOne)
    @public val i0: Instance[AddOne] = Instance(definition)
    @public val i1 = Module(new AddOne)
    i0.in := in
    i1.in := i0.out
    out := i1.out
  }
  @instantiable
  class AddTwoParameterized(width: Int, makeParameterizedOnes: Int => Seq[Instance[AddOneParameterized]])
      extends Module {
    val in = IO(Input(UInt(width.W)))
    val out = IO(Output(UInt(width.W)))
    val addOnes = makeParameterizedOnes(width)
    addOnes.head.in := in
    out := addOnes.last.out
    addOnes.zip(addOnes.tail).foreach { case (head, tail) => tail.in := head.out }
  }
  @instantiable
  class AddTwoWithNested(width: Int, makeParameterizedOnes: Int => Seq[Instance[AddOneWithNested]]) extends Module {
    val in = IO(Input(UInt(width.W)))
    val out = IO(Output(UInt(width.W)))
    val addOnes = makeParameterizedOnes(width)
  }

  @instantiable
  class AddFour extends Module {
    @public val in = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val definition = Definition(new AddTwoMixedModules)
    @public val i0 = Instance(definition)
    @public val i1 = Instance(definition)
    i0.in := in
    i1.in := i0.out
    out := i1.out
  }
  @instantiable
  class AggregatePortModule extends Module {
    @public val io = IO(new Bundle {
      val in = Input(UInt(32.W))
      val out = Output(UInt(32.W))
    })
    io.out := io.in
  }
  @instantiable
  class WireContainer {
    @public val innerWire = Wire(UInt(32.W))
  }
  @instantiable
  class AddOneWithInstantiableWire extends Module {
    @public val in = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val wireContainer = new WireContainer()
    wireContainer.innerWire := in + 1.U
    out := wireContainer.innerWire
  }
  @instantiable
  class AddOneContainer {
    @public val i0 = Module(new AddOne)
  }
  @instantiable
  class AddOneWithInstantiableModule extends Module {
    @public val in = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val moduleContainer = new AddOneContainer()
    moduleContainer.i0.in := in
    out := moduleContainer.i0.out
  }
  @instantiable
  class AddOneInstanceContainer {
    val definition = Definition(new AddOne)
    @public val i0 = Instance(definition)
  }
  @instantiable
  class AddOneWithInstantiableInstance extends Module {
    @public val in = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val instanceContainer = new AddOneInstanceContainer()
    instanceContainer.i0.in := in
    out := instanceContainer.i0.out
  }
  @instantiable
  class AddOneContainerContainer {
    @public val container = new AddOneContainer
  }
  @instantiable
  class AddOneWithInstantiableInstantiable extends Module {
    @public val in = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val containerContainer = new AddOneContainerContainer()
    containerContainer.container.i0.in := in
    out := containerContainer.container.i0.out
  }
  @instantiable
  class Viewer(val y: AddTwo, markPlease: Boolean) {
    @public val x = y
    if (markPlease) mark(x.i0.innerWire, "first")
  }
  @instantiable
  class ViewerParent(val x: AddTwo, markHere: Boolean, markThere: Boolean) extends Module {
    @public val viewer = new Viewer(x, markThere)
    if (markHere) mark(viewer.x.i0.innerWire, "second")
  }
  @instantiable
  class MultiVal() extends Module {
    @public val (x, y) = (Wire(UInt(3.W)), Wire(UInt(3.W)))
  }
  @instantiable
  class LazyVal() extends Module {
    @public val x = Wire(UInt(3.W))
    @public lazy val y = "Hi"
  }
  case class Parameters(string: String, int: Int) extends IsLookupable
  @instantiable
  class UsesParameters(p: Parameters) extends Module {
    @public val y = p
    @public val x = Wire(UInt(3.W))
  }
  @instantiable
  class HasList() extends Module {
    @public val y = List(1, 2, 3)
    @public val x = List.fill(3)(Wire(UInt(3.W)))
  }
  @instantiable
  class HasSeq() extends Module {
    @public val y = Seq(1, 2, 3)
    @public val x = Seq.fill(3)(Wire(UInt(3.W)))
  }
  @instantiable
  class HasOption() extends Module {
    @public val x: Option[UInt] = Some(Wire(UInt(3.W)))
  }
  @instantiable
  class HasEither() extends Module {
    @public val x: Either[Bool, UInt] = Right(Wire(UInt(3.W)).suggestName("x"))
    @public val y: Either[Bool, UInt] = Left(Wire(Bool()).suggestName("y"))
  }
  @instantiable
  class HasTuple2() extends Module {
    val x = Wire(UInt(3.W))
    val y = Wire(Bool())
    @public val xy = (x, y)
  }
  @instantiable
  class HasVec() extends Module {
    @public val x = VecInit(1.U, 2.U, 3.U)
  }
  @instantiable
  class HasIndexedVec() extends Module {
    val x = VecInit(1.U, 2.U, 3.U)
    @public val y = x(1)
  }
  @instantiable
  class HasSubFieldAccess extends Module {
    val in = IO(Input(Valid(UInt(8.W))))
    @public val valid = in.valid
    @public val bits = in.bits
  }
  @instantiable
  class HasPublicConstructorArgs(@public val int: Int) extends Module {
    @public val x = Wire(UInt(3.W))
  }
  @instantiable
  class InstantiatesHasVec() extends Module {
    @public val i0 = Instance(Definition(new HasVec()))
    @public val i1 = Module(new HasVec())
  }
  @instantiable
  class HasUninferredReset() extends Module {
    @public val in = IO(Input(UInt(3.W)))
    @public val out = IO(Output(UInt(3.W)))
    out := RegNext(in)
  }
  @instantiable
  abstract class HasBlah() extends Module {
    @public val blah: Int
  }

  @instantiable
  class ConcreteHasBlah() extends HasBlah {
    val blah = 10
  }
  @instantiable
  class HasTypeParams[D <: Data](d: D) extends Module {
    @public val blah = Wire(d)
  }

  @instantiable
  class HasMultipleTypeParamsInside extends Module {
    val tpDef0 = Definition(new HasTypeParams(Bool()))
    val tpDef1 = Definition(new HasTypeParams(UInt(4.W)))
    val i00 = Instance(tpDef0)
    val i01 = Instance(tpDef0)
    val i10 = Instance(tpDef1)
    val i11 = Instance(tpDef1)
  }

  @instantiable
  class HasMems() extends Module {
    @public val mem = Mem(8, UInt(32.W))
    @public val syncReadMem = SyncReadMem(8, UInt(32.W))
  }

  @instantiable
  class LeafInstantiable(val bundle: Data) {
    @public val bundle = bundle
  }

  @instantiable
  class NestedInstantiable(val in: LeafInstantiable, val out: LeafInstantiable) {
    @public val in = in
    @public val out = out
  }

  @instantiable
  class AddOneNestedInstantiableData(width: Int) extends Module {
    @public val in = IO(Input(UInt(width.W)))
    @public val out = IO(Output(UInt(width.W)))
    out := in + 1.U

    @public val leafOut = new LeafInstantiable(out)
    @public val leafIn = new LeafInstantiable(in)
    @public val nested = new NestedInstantiable(in = leafIn, out = leafOut)

  }
  @instantiable
  class HasUnsanitaryBundleField extends Module {
    class Interface extends Bundle {
      val `a-x` = UInt(8.W)
    }
    val realIn = IO(Input(new Interface))
    // It's important to have this redirection to trip an old bug
    @public val in = realIn.`a-x`
    @public val out = IO(Output(UInt(8.W)))
    out := in
  }

  class AddTwoNestedInstantiableData(width: Int) extends Module {
    val in = IO(Input(UInt(width.W)))
    val out = IO(Output(UInt(width.W)))
    val addOneDef = Definition(new AddOneNestedInstantiableData(width))
    val i0 = Instance(addOneDef)
    val i1 = Instance(addOneDef)
    i0.in := in
    i1.in := i0.out
    out := i1.out

    // both are equivalent to the above
    i1.leafIn.bundle := i0.leafOut.bundle
    i1.nested.in.bundle := i0.nested.out.bundle
  }

  class AddTwoNestedInstantiableDataSubmodule(addOneDef: Definition[AddOneNestedInstantiableData]) extends Module {
    val in = IO(Input(UInt(addOneDef.in.getWidth.W)))
    val out = IO(Output(UInt(addOneDef.out.getWidth.W)))
    val i0 = Instance(addOneDef)
    val i1 = Instance(addOneDef)
    i0.in := in
    i1.in := i0.out
    out := i1.out

    // both are equivalent to the above
    i1.leafIn.bundle := i0.leafOut.bundle
    i1.nested.in.bundle := i0.nested.out.bundle
  }

  class AddTwoNestedInstantiableDataWrapper(width: Int) extends Module {
    val in = IO(Input(UInt(width.W)))
    val out = IO(Output(UInt(width.W)))

    val original = Module(new AddOneNestedInstantiableData(width))
    val copy = Module(new AddTwoNestedInstantiableDataSubmodule(original.toDefinition))

    original.in := in
    copy.in := original.out
    out := copy.out

  }

}
