// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental.hierarchy

import chisel3._
import chisel3.util.{DecoupledIO, Valid}
import chisel3.experimental.hierarchy._
import chisel3.experimental.BaseModule
import chisel3.experimental.dataview._
import chiselTests.experimental.SimpleBundleDataView._

object Examples {
  import Annotations._
  @instantiable
  class AddOne extends Module {
    @public val in  = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val innerWire = Wire(UInt(32.W))
    innerWire := in + 1.U
    out := innerWire
  }
  @instantiable
  class AddOneWithAnnotation extends Module {
    @public val in  = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val innerWire = Wire(UInt(32.W))
    mark(innerWire, "innerWire")
    innerWire := in + 1.U
    out := innerWire
  }
  @instantiable
  class AddOneWithAbsoluteAnnotation extends Module {
    @public val in  = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val innerWire = Wire(UInt(32.W))
    amark(innerWire, "innerWire")
    innerWire := in + 1.U
    out := innerWire
  }
  @instantiable
  class AddTwo extends Module {
    @public val in  = IO(Input(UInt(32.W)))
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
    @public val in  = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    val definition = Definition(new AddOne)
    @public val i0: Instance[AddOne] = Instance(definition)
    @public val i1 = Module(new AddOne)
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
    @public val in  = IO(Input(UInt(32.W)))
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
    @public val in  = IO(Input(UInt(32.W)))
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
    @public val in  = IO(Input(UInt(32.W)))
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
    @public val in  = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val containerContainer = new AddOneContainerContainer()
    containerContainer.container.i0.in := in
    out := containerContainer.container.i0.out
  }
  @instantiable
  class Viewer(val y: AddTwo, markPlease: Boolean) {
    @public val x = y
    if(markPlease) mark(x.i0.innerWire, "first")
  }
  @instantiable
  class ViewerParent(val x: AddTwo, markHere: Boolean, markThere: Boolean) extends Module {
    @public val viewer = new Viewer(x, markThere)
    if(markHere) mark(viewer.x.i0.innerWire, "second")
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
  class HasVec() extends Module {
    @public val x = VecInit(1.U, 2.U, 3.U)
  }
  class MyBundle extends Bundle { val x = UInt(3.W) }
  @instantiable
  class HasVecOfBundle() extends Module {
    @public val x = Wire(Vec(3, new MyBundle()))
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
  class InstantiatesHasVecOfBundle() extends Module {
    @public val i0 = Instance(Definition(new HasVecOfBundle()))
    @public val i1 = Module(new HasVecOfBundle())
  }
  @instantiable
  class HasUninferredReset() extends Module {
    @public val in = IO(Input(UInt(3.W)))
    @public val out = IO(Output(UInt(3.W)))
    out := RegNext(in)
  }
  class InOutBundle extends Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  }
  @instantiable
  trait ModuleIntf extends BaseModule {
    @public val io = IO(new InOutBundle)
  }
  @instantiable
  class ModuleWithCommonIntf(suffix: String = "") extends Module with ModuleIntf {
    override def desiredName: String = super.desiredName + suffix
    @public val sum = io.in + 1.U

    io.out := sum
  }
  class BlackBoxWithCommonIntf extends BlackBox with ModuleIntf
  class SelectTop extends Module {
    val d = Definition(new AddTwo)
    val i0 = Instance(d)
    val i1 = Instance(d)
  }

  @instantiable
  class SelectParameterized[+T <: Data](gen: T) extends Module {
    val d = Definition(new AddOne)
    @public val i0 = Instance(d)
    @public val i1 = Instance(d)
  }

  class SelectTopParameterized extends Module {
    val d = Definition(new SelectParameterized[UInt](UInt(3.W)))
    val i0 = Instance(d)
    val i1 = Instance(d)
  }
  @instantiable
  class HasCMAR extends Module {
    @public val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
    @public val m = Module(new AggregatePortModule)
    @public val c = experimental.CloneModuleAsRecord(m)
  }
  @instantiable
  class MyAggDataViewModule extends RawModule {
    private val a = IO(Input(new BundleA(8)))
    private val b = IO(Output(new BundleA(8)))
    @public val in = a.viewAs[BundleB]
    @public val out = b.viewAs[BundleB]
    out := in
  }
  @instantiable
  class MyDataViewModule extends RawModule {
    val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
    val sum = in + 1.U
    out := sum + 1.U
    @public val foo = in.viewAs[UInt]
    @public val bar = sum.viewAs[UInt]
  }
  import chiselTests.experimental.FlatDecoupledDataView._
  type RegDecoupled = DecoupledIO[FizzBuzz]
  @instantiable
  class AggViewsModule extends RawModule {
    private val a = IO(Flipped(new FlatDecoupled))
    private val b = IO(new FlatDecoupled)
    @public val enq = a.viewAs[RegDecoupled]
    @public val deq = b.viewAs[RegDecoupled]
    @public val enq_valid = enq.valid // Also return a subset of the view
    deq <> enq
  }
  @instantiable
  class SimpleViews extends RawModule {
    val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
    val sum = in + 1.U
    out := sum + 1.U
    @public val foo = in.viewAs[UInt]
    @public val bar = sum.viewAs[UInt]
  }
  @instantiable
  class ViewOfViews extends RawModule {
    private val a = IO(Input(UInt(8.W)))
    private val b = IO(Output(new BundleA(8)))
    @public val in = a.viewAs[UInt].viewAs[UInt]
    @public val out = b.viewAs[BundleB].viewAs[BundleA].viewAs[BundleB]
    out.bar := in
  }
  @instantiable
  class ImplicitAndDataView extends RawModule {
    private val a = IO(Input(UInt(8.W)))
    private val b = IO(Output(UInt(8.W)))
    @public val ports = Seq(a, b)
    b := a
  }
}
