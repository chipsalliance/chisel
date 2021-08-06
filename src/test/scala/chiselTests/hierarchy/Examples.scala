package chiselTests.hierarchy
import chisel3._
import chisel3.internal.{instantiable, public}


object Examples {
  import Annotations._
  @instantiable
  class AddOne extends MultiIOModule {
    @public val in  = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val innerWire = Wire(UInt(32.W))
    innerWire := in + 1.U
    out := innerWire
  }
  @instantiable
  class AddTwo extends MultiIOModule {
    @public val in  = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    val template = Definition(new AddOne)
    @public val i0: Instance[AddOne] = Instance(template)
    @public val i1: Instance[AddOne] = Instance(template)
    i0.in := in
    i1.in := i0.out
    out := i1.out
  }
  @instantiable
  class AddTwoMixedModules extends MultiIOModule {
    @public val in  = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    val template = Definition(new AddOne)
    @public val i0: Instance[AddOne] = Instance(template)
    @public val i1 = Module(new AddOne)
    i0.in := in
    i1.in := i0.out
    out := i1.out
  }
  @instantiable
  class WireContainer extends IsInstantiable {
    @public val innerWire = Wire(UInt(32.W))
  }
  @instantiable
  class AddOneWithInstantiableWire extends MultiIOModule {
    @public val in  = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val wireContainer = new WireContainer()
    wireContainer.innerWire := in + 1.U
    out := wireContainer.innerWire
  }
  @instantiable
  class AddOneContainer extends IsInstantiable {
    @public val i0 = Module(new AddOne)
  }
  @instantiable
  class AddOneWithInstantiableModule extends MultiIOModule {
    @public val in  = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val moduleContainer = new AddOneContainer()
    moduleContainer.i0.in := in
    out := moduleContainer.i0.out
  }
  @instantiable
  class AddOneInstanceContainer extends IsInstantiable {
    val definition = Definition(new AddOne)
    @public val i0 = Instance(definition)
  }
  @instantiable
  class AddOneWithInstantiableInstance extends MultiIOModule {
    @public val in  = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val instanceContainer = new AddOneInstanceContainer()
    instanceContainer.i0.in := in
    out := instanceContainer.i0.out
  }
  @instantiable
  class AddOneContainerContainer extends IsInstantiable {
    @public val container = new AddOneContainer
  }
  @instantiable
  class AddOneWithInstantiableInstantiable extends MultiIOModule {
    @public val in  = IO(Input(UInt(32.W)))
    @public val out = IO(Output(UInt(32.W)))
    @public val containerContainer = new AddOneContainerContainer()
    containerContainer.container.i0.in := in
    out := containerContainer.container.i0.out
  }
  @instantiable
  class Viewer(val y: AddTwo, markPlease: Boolean) extends IsInstantiable {
    @public val x = y
    if(markPlease) mark(x.i0.innerWire, "first")
  }
  @instantiable
  class ViewerParent(val x: AddTwo, markHere: Boolean, markThere: Boolean) extends MultiIOModule {
    @public val viewer = new Viewer(x, markThere)
    if(markHere) mark(viewer.x.i0.innerWire, "second")
  }
  @instantiable
  class MultiVal() extends MultiIOModule {
    @public val (x, y) = (Wire(UInt(3.W)), Wire(UInt(3.W)))
  }
  @instantiable
  class LazyVal() extends MultiIOModule {
    @public val x = Wire(UInt(3.W))
    @public lazy val y = "Hi"
  }
  case class Parameters(string: String, int: Int) extends IsLookupable
  @instantiable
  class UsesParameters(p: Parameters) extends MultiIOModule {
    @public val y = p
    @public val x = Wire(UInt(3.W))
  }
  @instantiable
  class HasList() extends MultiIOModule {
    @public val y = List(1, 2, 3)
    @public val x = List.fill(3)(Wire(UInt(3.W)))
  }
  @instantiable
  class HasOption() extends MultiIOModule {
    @public val x: Option[UInt] = Some(Wire(UInt(3.W)))
  }
  @instantiable
  class HasVec() extends MultiIOModule {
    @public val x = VecInit(1.U, 2.U, 3.U)
  }
  @instantiable
  class HasIndexedVec() extends MultiIOModule {
    val x = VecInit(1.U, 2.U, 3.U)
    @public val y = x(1)
  }
  @instantiable
  class HasPublicConstructorArgs(@public val int: Int) extends MultiIOModule {
    @public val x = Wire(UInt(3.W))
  }
}