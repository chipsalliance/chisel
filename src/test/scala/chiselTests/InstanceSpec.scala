package chiselTests

import Chisel.{Bundle, Module}
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage, DesignAnnotation}
import chisel3._
import chisel3.internal.Instance
import firrtl.ir.Circuit
import firrtl.options.{Dependency, StageError}
import firrtl.stage.FirrtlCircuitAnnotation
import org.scalactic.source

import scala.reflect.ClassTag

class InstanceSpec extends ChiselPropSpec with Utils {

  /** Return a Chisel circuit for a Chisel module
    * @param gen a call-by-name Chisel module
    */
  def build[T <: RawModule](gen: => T): T = {
    val stage = new ChiselStage {
      override val targets = Seq( Dependency[chisel3.stage.phases.Checks],
        Dependency[chisel3.stage.phases.Elaborate],
        Dependency[chisel3.stage.phases.Convert]
      )
    }

    val ret = stage
      .execute(Array("--no-run-firrtl"), Seq(ChiselGeneratorAnnotation(() => gen)))

    println(ret.collectFirst { case FirrtlCircuitAnnotation(cir) => cir.serialize }.get)
    ret.collectFirst { case DesignAnnotation(a) => a } .get.asInstanceOf[T]
  }

  /** Return a FIRRTL circuit for a Chisel module
    * @param gen a call-by-name Chisel module
    */
  def buildFirrtl[T <: RawModule](gen: => T): Circuit = {
    val stage = new ChiselStage {
      override val targets = Seq( Dependency[chisel3.stage.phases.Checks],
        Dependency[chisel3.stage.phases.Elaborate],
        Dependency[chisel3.stage.phases.Convert]
      )
    }

    val ret = stage
      .execute(Array("--no-run-firrtl"), Seq(ChiselGeneratorAnnotation(() => gen)))

    ret.collectFirst { case FirrtlCircuitAnnotation(cir) => cir }.get
  }

  /** Intercept a specific ChiselException
    * @param message
    * @param f
    * @param classTag
    * @param pos
    * @tparam T
    */
  def intercept[T <: AnyRef](message: String)(f: => Any)(implicit classTag: ClassTag[T], pos: source.Position): Unit = {
    intercept[StageError] {
      val x = f
      x
    } match {
      case s: StageError => s.getCause match {
        case c: ChiselException => assert(c.getMessage.contains(message))
      }
    }
  }


  property("Calling a module's function which references its instance ports, e.g. tieoffs") {
    class Child(val width: Int) extends MultiIOModule {
      val in  = IO(Input(UInt(width.W)))
      val out = IO(Output(UInt(width.W)))
      out := in + in
      def tieoff(): Unit = in := 0.U
    }
    class Parent(child: Child) extends MultiIOModule {
      val in  = IO(Input(UInt(child.width.W)))
      val out = IO(Output(UInt(child.width.W)))
      val c: Child = Instance(child)
      c.tieoff()
      out := c.out + c.out
    }
    val child: Child = build { new Child(10) }
    buildFirrtl { new Parent(child) }
  }

  property("Calling a module's function which references its internal state should error gracefully") {
    class Child(val width: Int) extends MultiIOModule {
      val in  = IO(Input(UInt(width.W)))
      val out = IO(Output(UInt(width.W)))
      val reg = Reg(UInt(width.W))
      out := reg + reg
      def connectToReg(): Unit = reg := in
    }
    class Parent(child: Child) extends MultiIOModule {
      val in  = IO(Input(UInt(child.width.W)))
      val out = IO(Output(UInt(child.width.W)))
      val c = Instance(child)
      c.connectToReg()
      out := c.out + c.out
    }
    val child: Child = build { new Child(10) }
    intercept[StageError](
      "Connection between sink (UInt<10>(Reg in Child)) and source (UInt<10>(IO in in Child)) failed"
    ) {
      buildFirrtl { new Parent(child) }
    }
  }

  property("Passing instances to a function should work") {
    def tieoff(child: Child): Unit = {
      child.in := 0.U
    }
    class Child(val width: Int) extends MultiIOModule {
      val in  = IO(Input(UInt(width.W)))
      val out = IO(Output(UInt(width.W)))
      val reg = Reg(UInt(width.W))
      out := reg + reg
    }
    class Parent(child: Child) extends MultiIOModule {
      val in  = IO(Input(UInt(child.width.W)))
      val out = IO(Output(UInt(child.width.W)))
      val c = Instance(child)
      tieoff(c)
      out := c.out + c.out
    }
    val child: Child = build { new Child(10) }
    buildFirrtl { new Parent(child) }
  }

  property("Programmatic construction of instances should work") {
    class Child(val width: Int) extends MultiIOModule {
      val in  = IO(Input(UInt(width.W)))
      val out = IO(Output(UInt(width.W)))
      val reg = Reg(UInt(width.W))
      out := reg + reg
    }
    class Parent(child: Child, nChildren: Int) extends MultiIOModule {
      val in  = IO(Input(UInt(child.width.W)))
      val out = IO(Output(UInt(child.width.W)))
      val c = Instance(child)
      out := (0 until nChildren).foldLeft(in) {
        case (in, int) =>
          val c = Instance(child)
          c.in := in
          c.out
      }
    }
    val child: Child = build { new Child(10) }
    buildFirrtl { new Parent(child, 3) }

  }

  property("Defining and constructing a bundle in a Module should work") {
    class MyModule() extends Module {
      class MyBundle extends Bundle {
        val in = Input(UInt(3.W))
        val out = Output(UInt(3.W))
      }

      // Without a special case in the compiler plugin, this would fail because referencing `MyBundle`
      //  is via `MyModule.MyBundle`, which matches the general pattern the compiler plugin looks for
      //  to intercept references to Data members of Modules
      val io = IO(new MyBundle)
    }

    build { new MyModule() }
  }
  property("Connecting internal modules should not give null pointer exception") {
    class Leaf(val width: Int) extends MultiIOModule {
      val in  = IO(Input(UInt(width.W)))
    }

    class ChildX(val width: Int) extends MultiIOModule {
      val in  = IO(Input(UInt(width.W)))
      val out = IO(Output(UInt(width.W)))
      val leaf = Module(new Leaf(width))
      def connectToLeaf(): Unit = leaf.in := in
    }
    class Parent(child: ChildX) extends MultiIOModule {
      val in  = IO(Input(UInt(child.width.W)))
      val out = IO(Output(UInt(child.width.W)))
      val c = Instance(child)
      c.connectToLeaf()
      out := c.out + c.out
    }
    val child: ChildX = build { new ChildX(10) }
    intercept[StageError](
      "Connection between sink "
    ) {
      buildFirrtl { new Parent(child) }
    }

  }


  property("Defining stuff in a mix-in trait should still be nullified") {
    class LeafZ(val width: Int) extends MultiIOModule {
      val in  = IO(Input(UInt(width.W)))
    }
    trait InModuleBodyZ { this: MultiIOModule =>
      val leaf = Module(new LeafZ(10))
      leaf.in := 0.U
    }

    class ChildZ extends MultiIOModule with InModuleBodyZ {
      val in = IO(Input(UInt(3.W)))
      leaf.in := in
    }

    class ParentZ(child: ChildZ) extends MultiIOModule {
      val in  = IO(Input(UInt(3.W)))
      val out = IO(Output(UInt(3.W)))
      val c = Instance(child)
      c.in := in
      out := in
    }

    val child: ChildZ = build { new ChildZ() }
    val cir = buildFirrtl { new ParentZ(child) }
    val nLeafs = cir.modules.collect {
      case m: firrtl.ir.Module if m.name == "Leaf" => m
    }
    assert(nLeafs.isEmpty, "Leaf not be elaborated when building Parent")
  }

  property("Defining stuff in a mix-in trait without self type should error gracefully") {
    class Leaf(val width: Int) extends MultiIOModule {
      val in  = IO(Input(UInt(width.W)))
    }
    trait InModuleBody {
      val leaf = Module(new Leaf(10))
      leaf.in := 0.U
    }

    class Child extends MultiIOModule with InModuleBody {
      val in = IO(Input(UInt(3.W)))
      leaf.in := in
    }

    class Parent(child: Child) extends MultiIOModule {
      val in  = IO(Input(UInt(3.W)))
      val out = IO(Output(UInt(3.W)))
      val c = Instance(child)
      c.in := in
      out := in
    }

    val child: Child = build { new Child() }

    intercept[StageError]("Called Module() within an instance.") { build { new Parent(child) } }
  }


  property("Passing non-int primitives and classes should work") {
    case class Parameters(name: String)

    //byte , short , int , long , float , double , boolean and char.
    class Child(byte: Byte,
                short: Short,
                int: Int,
                long: Long,
                float: Float,
                double: Double,
                boolean: Boolean,
                char: Char,
                parameters: Parameters) extends MultiIOModule {
      val myName: String = parameters.name + byte + short + int + long + float + double + boolean + char
      val in = IO(Input(UInt(3.W)))
      val out = IO(Output(UInt(3.W)))
      out := in
    }

    class Parent(child: Child) extends MultiIOModule {
      val in  = IO(Input(UInt(3.W)))
      val out = IO(Output(UInt(3.W)))
      val c = Instance(child)
      c.in := in
      out := c.out
    }

    val child: Child = build { new Child(
      0, 1, 2, 3, 4, 5.0, true, 'a', Parameters("foo")
    ) }
    build { new Parent(child) }
  }

  property("Accessing non-data members of instance should work") {
    class Child(n: String) extends MultiIOModule {
      val myName: String = n
      val in = IO(Input(UInt(3.W)))
      val out = IO(Output(UInt(3.W)))
      out := in
    }

    class Parent(child: Child) extends MultiIOModule {
      val in  = IO(Input(UInt(3.W)))
      val out = IO(Output(UInt(3.W)))
      val c = Instance(child)
      c.in := in
      out := c.out
      assert(child.myName == c.myName)
    }

    val child: Child = build { new Child("foobar" ) }
    build { new Parent(child) }
  }

  property("Returning value from a function should work") {
    class Child(n: String) extends MultiIOModule {
      val myName: String = n
      val in = IO(Input(UInt(3.W)))
      val out = IO(Output(UInt(3.W)))
      out := in
    }

    class Parent(child: Child) extends MultiIOModule {
      val in  = IO(Input(UInt(3.W)))
      val out = IO(Output(UInt(3.W)))
      def createInstance(): Child = Instance(child).suggestName(child.myName)
      createInstance().in := in
    }

    val child: Child = build { new Child("foobar" ) }
    build { new Parent(child) }

  }
  property("Using var+null build pattern should work") {
    class Foo extends Module {
      val io = new Bundle {}
    }
    var result: Foo = null
    build { result = new Foo; result }
    //TODO: Uncommenting the following will trigger a bad error. I've tried for a while to figure out
    // how to fix it on the compiler side and haven't found a solution. two user-facing solutions are shown here
    //result.name should be ("Foo")

    // Solution 1: casting using asInstanceOf
    result.asInstanceOf[Foo].name should be ("Foo")

    // Solution 2: intermediate val
    val x = result
    x.name should be ("Foo")

  }
}

//Prints out:
/*
[info] [0.002] Elaborating design...
Elaborated Leaf!
[info] [0.063] Done elaborating.
circuit SimpleX :
  module Leaf :
    input clock : Clock
    input reset : Reset
    input in : UInt<3>
    output out : UInt<3>

    node _out_T = add(in, in) @[InstanceSpec.scala 24:13]
    node _out_T_1 = tail(_out_T, 1) @[InstanceSpec.scala 24:13]
    out <= _out_T_1 @[InstanceSpec.scala 24:7]

  module SimpleX :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<3>
    output out : UInt<3>

    inst leaf of Leaf @[InstanceSpec.scala 33:20]
    leaf.clock <= clock
    leaf.reset <= reset
    node _out_T = add(in, in) @[InstanceSpec.scala 35:13]
    node _out_T_1 = tail(_out_T, 1) @[InstanceSpec.scala 35:13]
    out <= _out_T_1 @[InstanceSpec.scala 35:7]

[info] [0.000] Elaborating design...
[info] [0.024] Done elaborating.
circuit Top :
  extmodule SimpleX :
    output out : UInt<3>
    input in : UInt<3>
    input reset : UInt<1>
    input clock : Clock

    defname = SimpleX


  extmodule SimpleX_2 :
    output out : UInt<3>
    input in : UInt<3>
    input reset : UInt<1>
    input clock : Clock

    defname = SimpleX


  module Top :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<3>
    output out : UInt<3>

    inst SIMPLE of SimpleX @[InstanceSpec.scala 50:33]
    SIMPLE.clock is invalid
    SIMPLE.reset is invalid
    SIMPLE.in is invalid
    SIMPLE.out is invalid
    inst SIMPLE2 of SimpleX_2 @[InstanceSpec.scala 51:34]
    SIMPLE2.clock is invalid
    SIMPLE2.reset is invalid
    SIMPLE2.in is invalid
    SIMPLE2.out is invalid
    SIMPLE.in <= in @[InstanceSpec.scala 53:13]
    SIMPLE2.in <= SIMPLE.out @[InstanceSpec.scala 56:14]
    out <= SIMPLE2.out @[InstanceSpec.scala 58:6]
 */

/*
class Leaf extends MultiIOModule {
  val in  = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))
  out := in + in
  println("Elaborated Leaf!")
}

class SimpleX(int: Int) extends MultiIOModule {
  val in  = IO(Input(UInt(3.W)))
  //val in  = chisel3.experimental.isInstance.check(IO(Input(UInt(3.W))))
  val out = IO(Output(UInt(3.W)))
  //val out = chisel3.experimental.isInstance.check(IO(Output(UInt(3.W))))
  val leaf = Module(new Leaf)
  val reg = Reg(UInt(3.W))
  //val leaf = chisel3.experimental.isInstance.check(Module(new Leaf))
  out := in + in
  //chisel3.experimental.isInstance.check(() => out := in + in)
  val x = int
  //val x = chisel3.experimental.isInstance.check(int)
  def tieoff() = in := 0.U
  def connectInternals() = reg := in
}

class MyPipe2(gen: Data, latency: Int = 1)(implicit compileOptions: CompileOptions) extends Module {
  /** Interface for [[Pipe]]s composed of a [[Valid]] input and [[Valid]] output
    * @define notAQueue
    */
  class MyPipeIO extends Bundle {

    /** [[Valid]] input */
    val enq = Input(Valid(UInt(3.W)))

    /** [[Valid]] output. Data will appear here `latency` cycles after being valid at `enq`. */
    val deq = Output(Valid(UInt(3.W)))
  }

  val io = IO(new MyPipeIO)
  //val io = IO(new Bundle { val x = UInt(3.W) })

  //io.deq <> Pipe(io.enq, latency)
}


class Top(simple: SimpleX) extends MultiIOModule {
  val in  = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))

  // 1) Original backing module of type Simple
  // 2) New Black Box module
  // 3) New Empty Module of type Simple

  // Jack's thoughts
  // Make sure we check if plugin has been run

  val SIMPLE: SimpleX = Instance(simple)
  val SIMPLE2: SimpleX = Module(new SimpleX(10))

  SIMPLE.tieoff()
  SIMPLE2.tieoff()
  //SIMPLE.connectInternals()

  YunsupFunc.func(SIMPLE)
  YunsupFunc.func(SIMPLE2)

  SIMPLE.in := in
  //SIMPLE.useInstance(SIMPLE.getBackingModule[SimpleX].in) := in

  SIMPLE2.in := SIMPLE.out

  out:= SIMPLE2.out
  //out := SIMPLE2.useInstance(SIMPLE.getBackingModule[SimpleX].out)
}
  //property("Explicit example test case") {
  //  //Diplomacy occurs
  //  //Chisel Construction
  //  val simple: SimpleX = build { new SimpleX(10) }
  //  val top: Top = build { new Top(simple) }
  //}


 */

