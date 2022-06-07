// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental.hierarchy

import chisel3._
import chisel3.util.Valid
import chisel3.experimental.hierarchy._
import chisel3.experimental.BaseModule
import chisel3.internal.instantiable

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
  class WireContainer extends IsWrappable {
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
  class AddOneContainer extends IsWrappable {
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
  class AddOneInstanceContainer extends IsWrappable {
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
  class AddOneContainerContainer extends IsWrappable {
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
  class Viewer(val y: AddTwo, markPlease: Boolean) extends IsWrappable {
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
    //@public val blah = Wire(d)
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
}


object DefinitiveExamples {
  @func def plusN(n: Int)(in: Int): Int = n + in

  @instantiable
  class AddOne extends Module {
    println(s"Elaborating AddOne: $this")
    @public val width: Definitive[Int] = Definitive.empty[Int]
    @public val in  = IO(Input(UInt(width.W)))
    @public val out = IO(Output(UInt(width.W)))

    override val implementation: Option[Implementation] = Some(AddOneImp)
  }
  object AddOneImp extends CustomImplementation {
    type P = AddOne
    
    def implement(d: ResolvedDefinition[P]): Unit = {
      d.out.value := d.in.value + 1.U
    }
  }
  @instantiable
  class AddTwo(d: Definition[AddOne]) extends Module {
    @public val in  = IO(Input(UInt(d.width.W)))
    @public val out = IO(Output(UInt(d.width.W)))
    @public val i0 = Instance(d)
    @public val i1 = Instance(d)

    override val implementation: Option[Implementation] = Some(AddTwoImp)
  }

  object AddTwoImp extends CustomImplementation {
    type P = AddTwo
    
    def implement(d: ResolvedDefinition[P]): Unit = {
      d.i0.in.value := d.in.value
      d.i1.in.value := d.i0.out.value
      d.out.value := d.i1.out.value
    }
  }


}
object ContextualExamples {
  @func def max()(ls: List[Int]): Int = ls.max
  case class Max() extends core.CustomCombinerFunction[Int, Int] {
    type I = Int
    type O = Int
    def apply(i: List[I]): O = i.max
  }

  @instantiable
  class Child extends Module {
    @public val index = Contextual.empty[Int]
    @public val maxIndex: Definitive[Int] = index.combine(Max())
    override val implementation = Some(ChildImp)
  }
  object ChildImp extends CustomImplementation {
    type P = Child
    def implement(d: ResolvedDefinition[Child]): Unit = {
      val w = WireInit(d.maxIndex.value.U)
    }
  }

  @instantiable
  class Parent(index0: Option[Int], index1: Option[Int]) extends Module {
    val definition = Definition(new Child)
    @public val c0 = Instance(definition)
    index0.map{ i => c0.index.value = i }
    @public val c1 = Instance(definition)
    index1.map{ i => c1.index.value = i }
  }

  @instantiable
  class DerivingParent(index0: Option[Int], index1: Option[Int]) extends Module {
    val definition = Definition(new Child)
    @public val pindex = Contextual.empty[Int]
    val c0 = Instance(definition)
    pindex.setAs(c0.index)
    val c1 = Instance(definition)
    c1.index.setAs(pindex)
  }

  object DiplomacyExample {

    @func def capitalize()(language: String): String = language.capitalize
    case class Aggregate() extends core.CustomCombinerFunction[String, List[String]] {
      type I = String
      type O = List[String]
      def apply(i: List[I]): O = i
    }

    
    @instantiable
    class SourceNode(val dnValue: Contextual[String]) extends IsWrappable {
      @public val dnOutgoing = dnValue
      @public val upIncoming = Contextual.empty[String]
      @public val edge = (upIncoming, dnValue)
    }
    @instantiable
    class SinkNode(val upValue: Contextual[String]) extends IsWrappable {
      @public val upOutgoing = upValue
      @public val dnIncoming = Contextual.empty[String]
      @public val edge = (upValue, dnIncoming)
    }
    object SinkNode {
      implicit class SinkNodeExtension(e: Instance[SinkNode]) {
        def :=(a: AdapterNode): AdapterNode = {
          e.dnIncoming.setAs(a.dnOutgoing)
          a.upIncoming.setAs(e.upOutgoing)
          a
        }
      }
    }
    @instantiable
    class AdapterNode(upFunc: ParameterFunction, dnFunc: ParameterFunction) extends IsWrappable {
      val upIncoming = Contextual.empty[String]
      val upOutgoing = Contextual.empty[String]
      val dnIncoming = Contextual.empty[String]
      val dnOutgoing = Contextual.empty[String]
      dnOutgoing.setAs(dnIncoming.modify(dnFunc))
      upOutgoing.setAs(upIncoming.modify(upFunc))

      def :=(src: Instance[SourceNode]): AdapterNode = {
        src.upIncoming.setAs(upOutgoing)
        dnIncoming.setAs(src.dnOutgoing)
        this
      }
    }
    @instantiable
    class Mouth() extends Module {
      @public val language = Contextual.empty[String]
      @public val node = new SourceNode(language)
      @public val upResult = node.edge._1.combine(Aggregate())
      @public val dnResult = node.edge._2.combine(Aggregate())
    }

    @instantiable
    class Ear() extends Module {
      @public val language = Contextual.empty[String]
      @public val node = new SinkNode(language)
      @public val upResult = node.edge._1.combine(Aggregate())
      @public val dnResult = node.edge._2.combine(Aggregate())
    }

    @instantiable
    class Top() extends Module {
      val mouthDef = Definition(new Mouth())
      @public val m0 = Instance(mouthDef)
      m0.language.value = "English"
      @public val m1 = Instance(mouthDef)
      m1.language.value = "Spanish"

      val earDef = Definition(new Ear())
      @public val e0 = Instance(earDef)
      e0.language.value = "Hebrew"
      @public val e1 = Instance(earDef)
      e1.language.value = "Portuguese"

      @public val a0 = new AdapterNode(capitalize(), capitalize())
      e0.node := a0 := m0.node

      @public val a1 = new AdapterNode(capitalize(), capitalize())
      e1.node := a1 := m1.node
    }

    object TopImp extends CustomImplementation {
      type P = Top
      def implement(d: ResolvedDefinition[Top]): Unit = {
        println(s"M0: ${d.m0.upResult.value}, ${d.m0.dnResult.value}")
        println(s"M1: ${d.m1.upResult.value}, ${d.m1.dnResult.value}")
        println(s"E0: ${d.e0.upResult.value}, ${d.e0.dnResult.value}")
        println(s"E1: ${d.e1.upResult.value}, ${d.e1.dnResult.value}")
      }
    }

  }

}