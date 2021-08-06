// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester
import chisel3.experimental.annotate
import chisel3.experimental.BaseModule
import chisel3.internal.{instantiable, public}
import _root_.firrtl.annotations._
import chisel3.stage.DesignAnnotation
import chisel3.stage.ChiselGeneratorAnnotation

@instantiable
class TopLevelDeclaration extends MultiIOModule {
  @public val in  = IO(Input(UInt(32.W)))
  @public val out = IO(Output(UInt(32.W)))
  out := in
}

@instantiable
class TopLevelDeclarationWithCompanionObject extends MultiIOModule {
  @public val in  = IO(Input(UInt(32.W)))
  @public val out = IO(Output(UInt(32.W)))
  out := in
}
object TopLevelDeclarationWithCompanionObject {
  def hello = "Hi"
}

object InstanceSpec {
  implicit class Str2RefTarget(str: String) {
    def rt = Target.deserialize(str).asInstanceOf[ReferenceTarget]
  }
  object Examples {
    @instantiable
    class AddOne(hasInner: Boolean) extends MultiIOModule {
      @public val in  = IO(Input(UInt(32.W)))
      @public val out = IO(Output(UInt(32.W)))
      @public val innerWire = if(hasInner) Some(Wire(UInt(32.W))) else None
      innerWire match {
        case Some(w) =>
          w := in + 1.U
          out := w
        case None =>
          out := in + 1.U
      }
    }
    @instantiable
    class AddTwo extends MultiIOModule {
      @public val in  = IO(Input(UInt(32.W)))
      @public val out = IO(Output(UInt(32.W)))
      val template = Definition(new AddOne(true))
      @public val i0: Instance[AddOne] = Instance(template)
      @public val i1 = Module(new AddOne(true))
      //@public val x = 10 //ERRORS!!!
      i0.in := in
      i1.in := i0.out
      out := i1.out
    }
  }
  object Annotations {
    case class MarkAnnotation(target: IsMember, tag: String) extends SingleTargetAnnotation[IsMember] {
      def duplicate(n: IsMember): Annotation = this.copy(target = n)
    }
    case class MarkChiselInstanceAnnotation[B <: BaseModule](d: Instance[B], tag: String) extends chisel3.experimental.ChiselAnnotation {
      def toFirrtl = MarkAnnotation(d.toTarget, tag)
    }
    case class MarkChiselAnnotation(d: Data, tag: String) extends chisel3.experimental.ChiselAnnotation {
      def toFirrtl = MarkAnnotation(d.toTarget, tag)
    }
    def mark(d: Data, tag: String): Unit = annotate(MarkChiselAnnotation(d, tag))
    def mark[B <: BaseModule](d: Instance[B], tag: String): Unit = annotate(MarkChiselInstanceAnnotation(d, tag))
  }
//  object HierarchyExamples {
//    //implicit def convert(m: AddOne): Instance[AddZeroInterface] = { }
//    @instantiable
//    final class AddOne extends MultiIOModule with AddInterface {
//      @public val in  = in
//      @public val out = out
//      @public val innerWire = Wire(UInt(32.W))
//      innerWire := in + 1.U
//      out := innerWire
//    }
//    @instantiable
//    class AddTwo extends MultiIOModule with AddInterface {
//      @public val in  = IO(Input(UInt(32.W)))
//      @public val out = IO(Output(UInt(32.W)))
//      @public val innerWire = Wire(UInt(32.W))
//      @public val innerModule = Module(new Interfaces.AddZero)
//      innerWire := in + 2.U
//      out := innerWire
//    }
//  }
}



class InstanceSpec extends ChiselFlatSpec with Utils {
  "Definition/Instance" should "enable instantiating the same instance multiple times" in {
    import InstanceSpec.Examples._
    import InstanceSpec.Annotations._
    class AddOneTester extends BasicTester {
      val template: Definition[AddOne] = Definition(new AddOne(true))
      val i0: Instance[AddOne] = Instance(template)
      val i1: Instance[AddOne] = Instance(template)
      i0.in := 42.U
      i1.in := i0.out
      i0.innerWire.map(x => mark(x, "Jack Was Here"))
      chisel3.assert(i1.out === 44.U)
      stop()
    }
    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
    println(output)
    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i0:AddOne>innerWire").asInstanceOf[ReferenceTarget], "Jack Was Here"))
    assertTesterPasses(new AddOneTester)
  }
  "Definition/Instance" should "enable instantiating nestingly, with modules" in {
    import InstanceSpec.Examples._
    import InstanceSpec.Annotations._
    class AddOneTester extends BasicTester {
      val template = Definition(new AddTwo)
      val i0 = Instance(template)
      val i1 = Instance(template)
      i0.in := 42.U
      i1.in := i0.out
      mark(i0.i0, "i0.i0")
      mark(i0.i1, "i0.i1")
      i0.i0.innerWire.map(x => mark(x, "Adam Was Here"))
      //i0.i1.innerWire.map(x => mark(x, "Megan Was Here"))
      chisel3.assert(i1.out === 46.U)
      stop()
    }

    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i0:AddTwo/i0:AddOne").asInstanceOf[InstanceTarget], "i0.i0"))
    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i0:AddTwo/i1:AddOne_2").asInstanceOf[InstanceTarget], "i0.i1"))
    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i0:AddTwo/i0:AddOne>innerWire").asInstanceOf[ReferenceTarget], "Adam Was Here"))
    //annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i0:AddTwo/i1:AddOne_2>innerWire").asInstanceOf[ReferenceTarget], "Megan Was Here"))
    assertTesterPasses(new AddOneTester)
  }
//  "Definition/Instance" should "implicitly convert modules to instances" in {
//    import InstanceSpec.Examples._
//    import InstanceSpec.Annotations._
//    def wireUp(i0: Instance[AddOne], i1: Instance[AddOne]): Unit = {
//      i1.in := i0.out
//    }
//    class AddOneTester extends BasicTester {
//      val template = Definition(new AddOne(false))
//      val i0 = Instance(template)
//      val i1 = Module(new AddOne(true))
//      i0.in := 42.U
//      wireUp(i0, i1)
//      i1.in := i0.out
//      chisel3.assert(i1.out === 44.U)
//      stop()
//    }
//
//    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
//    assertTesterPasses(new AddOneTester)
//  }
//  "Definition/Instance" should "work with top level declarations" in {
//    import InstanceSpec.Examples._
//    class AddZeroTester extends BasicTester {
//      val template = Definition(new TopLevelDeclaration())
//      val i0 = Instance(template)
//      val template2 = Definition(new TopLevelDeclarationWithCompanionObject())
//      val i1 = Instance(template2)
//      i0.in := 42.U
//      i1.in := i0.out
//      chisel3.assert(i1.out === 42.U)
//      stop()
//    }
//
//    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddZeroTester, args = Array("--full-stacktrace"))
//    assertTesterPasses(new AddZeroTester)
//  }
//  "Definition/Instance" should "work with public members of super classes" in {
//    @instantiable
//    class AddZero extends MultiIOModule {
//      @public val in  = IO(Input(UInt(32.W)))
//      @public val out = IO(Output(UInt(32.W)))
//      @public val clock = clock
//      out := in
//    }
//
//    @instantiable
//    class AddZeroWithCompanionObject extends MultiIOModule {
//      @public val in  = IO(Input(UInt(32.W)))
//      @public val out = IO(Output(UInt(32.W)))
//      @public val clock = clock
//      out := in
//    }
//    object AddZeroWithCompanionObject {
//      def hello = "hello"
//    }
//    class AddZeroTester extends BasicTester {
//      val i0 = Instance(Definition(new AddZero))
//      val i1 = Instance(Definition(new AddZeroWithCompanionObject()))
//      i0.in := 42.U
//      i1.in := i0.out
//      chisel3.assert(i1.out === 42.U)
//      stop()
//    }
//
//    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddZeroTester, args = Array("--full-stacktrace"))
//    assertTesterPasses(new AddZeroTester)
//  }
//  "Definition/Instance" should "access parameter case classes" in {
//    // Not sure how to do this. One option is to have marker trait IsAccessable or something, so we just define the lookupable for that
//    // Another option is to materialize the lookup somehow, but i think that requires the lookup to be implemented with a def macro 
//    // Going to try a marker trait first.
//    case class Parameters(word: String, number: Int) extends IsLookupable
//    @instantiable
//    class AddZero(val p: Parameters) extends MultiIOModule {
//      @public val in  = IO(Input(UInt(32.W)))
//      @public val out = IO(Output(UInt(32.W)))
//      @public val clock = clock
//      @public val p = p
//      out := in
//    }
//
//    class AddZeroTester extends BasicTester {
//      val i0 = Instance(Definition(new AddZero(Parameters("hi", 13))))
//      i0.in := 42.U
//      assert(i0.p.word == "hi")
//      assert(i0.p.number == 13)
//      chisel3.assert(i0.out === 42.U)
//      stop()
//    }
//
//    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddZeroTester, args = Array("--full-stacktrace"))
//    assertTesterPasses(new AddZeroTester)
//   }
//  "Definition/Instance" should "work with optional instances or modules" in {
//     // Doing typeclass derivation with chaining implicit defs.. let's see if it works!
//    @instantiable
//    class AddOne extends MultiIOModule {
//      @public val in  = IO(Input(UInt(32.W)))
//      @public val out = IO(Output(UInt(32.W)))
//      out := in + 1.U
//    }
//
//    @instantiable
//    class AddSome(first: Boolean, second: Boolean) extends MultiIOModule {
//      @public val in  = IO(Input(UInt(32.W)))
//      @public val out = IO(Output(UInt(32.W)))
//      @public val i0: Option[Instance[AddOne]] = if(first) Some(Instance(Definition(new AddOne))) else None
//      @public val i1: Option[AddOne] = if(second) Some(Module(new AddOne)) else None
//      (i0, i1) match {
//        case (None, None) =>
//          out := in
//        case (Some(i), None) =>
//          i.in := in
//          out := i.out
//        case (None, Some(i)) =>
//          i.in := in
//          out := i.out
//        case (Some(i0), Some(i1)) =>
//          i0.in := in
//          i1.in := i0.out
//          out := i1.out
//      }
//    }
//
//    class AddSomeTester extends BasicTester {
//      val i = Instance(Definition(new AddSome(true, false)))
//      i.in := 42.U
//      require(i.i0.nonEmpty)
//      chisel3.assert(i.out === 43.U)
//      stop()
//    }
//
//    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddSomeTester, args = Array("--full-stacktrace"))
//    assertTesterPasses(new AddSomeTester)
//
//  }
//  "@public" should "work on multi-vals" in {
//    // It already does automatically! just write a test
//  }
//  "@instantiable" should "work on non-modules" in {
//    // Example is counter, which is not a module, but has values which are hardware and instance specific
//    import InstanceSpec.Annotations._
//    @instantiable
//    case class Blah() extends IsInstantiable {
//      @public val w = Wire(UInt(32.W))
//      @public val x = "Hi"
//    }
//    @instantiable
//    class AddOne extends MultiIOModule {
//      @public val in  = IO(Input(UInt(32.W)))
//      @public val out = IO(Output(UInt(32.W)))
//      @public val blah = Blah()
//      blah.w := in + 1.U
//      out := blah.w
//    }
//    class AddOneTester extends BasicTester {
//      val i = Instance(Definition(new AddOne))
//      i.in := 42.U
//      chisel3.assert(i.out === 43.U)
//      require(i.blah.x == "Hi")
//      blahPrinter(i.blah)
//      mark(i.blah.w, "Adam Was Here")
//      stop()
//    }
//    def blahPrinter(b: Instance[Blah]): Unit = {
//      println(b.x)
//    }
//
//    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
//    println(output)
//    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i:AddOne>w").asInstanceOf[ReferenceTarget], "Adam Was Here"))
//    assertTesterPasses(new AddOneTester)
//
//  }
//  "@instantiable" should "work on non-modules with delayed elaborated hardware" in {
//    // Should we expose defs? issue is when IsInstantiable classes have delayed references to things
//    // We can't call them as a val because they need to be elaborated later, but we right now require @public to be on a val
//    // We could make it a lazy val?
//    import InstanceSpec.Annotations._
//    @instantiable
//    case class Blah(wire: () => UInt) extends IsInstantiable {
//      @public val x = true
//      @public lazy val w = wire()
//    }
//    @instantiable
//    class AddOne extends MultiIOModule {
//      @public val in  = IO(Input(UInt(32.W)))
//      @public val out = IO(Output(UInt(32.W)))
//      @public val blah = Blah(() => Wire(UInt(32.W)))
//      blah.w := in + 1.U
//      out := blah.w
//    }
//    class AddOneTester extends BasicTester {
//      val i = Instance(Definition(new AddOne))
//      i.in := 42.U
//      chisel3.assert(i.out === 43.U)
//      mark(i.blah.w, "Adam Was Here")
//      stop()
//    }
//
//    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
//    println(output)
//    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i:AddOne>w").asInstanceOf[ReferenceTarget], "Adam Was Here"))
//    assertTesterPasses(new AddOneTester)
//
//  }
//  "@public" should "work on vecs" in {
//    // Should we expose defs? issue is when IsInstantiable classes have delayed references to things
//    // We can't call them as a val because they need to be elaborated later, but we right now require @public to be on a val
//    // We could make it a lazy val?
//    import InstanceSpec.Annotations._
//    @instantiable
//    class AddOne extends MultiIOModule {
//      @public val in  = IO(Input(UInt(32.W)))
//      @public val out = IO(Output(UInt(32.W)))
//      @public val vec = VecInit(1.U)
//      out := in + vec(0)
//    }
//    class AddOneTester extends BasicTester {
//      val i = Instance(Definition(new AddOne))
//      i.in := 42.U
//      chisel3.assert(i.out === 43.U)
//      mark(i.vec, "Adam Was Here")
//      stop()
//    }
//
//    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
//    println(output)
//    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i:AddOne>vec").asInstanceOf[ReferenceTarget], "Adam Was Here"))
//    assertTesterPasses(new AddOneTester)
//  }
//  "@instantiable" should "work on seqs of non-modules" in {
//    // Example is counter, which is not a module, but has values which are hardware and instance specific
//    // TODO, this doesnt work with Lists, or other subtypes of Seq. Not sure why...
//    import InstanceSpec.Annotations._
//    @instantiable
//    case class Blah() extends IsInstantiable {
//      @public val w = Wire(UInt(32.W))
//      @public val x = "Hi"
//    }
//    class AddOneTester extends BasicTester {
//      val seq: Seq[Blah] = Seq(Blah())
//      blahPrinter(seq)
//      stop()
//    }
//    def blahPrinter(bs: Seq[Instance[Blah]]): Unit = {
//      bs.foreach(b => println(b.x))
//    }
//
//    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
//    //assertTesterPasses(new AddOneTester)
//
//  }
//  "@public" should "work on statically indexed vectors" in {
//    // Issue was in the toTarget logic for XMRs
//    import InstanceSpec.Annotations._
//    @instantiable
//    class AddOne extends MultiIOModule {
//      @public val in  = IO(Input(UInt(32.W)))
//      @public val out = IO(Output(UInt(32.W)))
//      val vec = VecInit(1.U)
//      @public val vec0 = VecInit(1.U)(0)
//      out := in + vec0
//      mark(vec0, "Chris Was Here")
//    }
//    class AddOneTester extends BasicTester {
//      val i = Instance(Definition(new AddOne))
//      i.in := 42.U
//      chisel3.assert(i.out === 43.U)
//      mark(i.vec0, "Adam Was Here")
//      stop()
//    }
//
//    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
//    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i:AddOne>_vec0_WIRE[0]").asInstanceOf[ReferenceTarget], "Adam Was Here"))
//    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOne|AddOne>_vec0_WIRE[0]").asInstanceOf[ReferenceTarget], "Chris Was Here"))
//    assertTesterPasses(new AddOneTester)
//  }
//  "To target" should "work outside of the builder context" in {
//    // Issue was in the toTarget logic for XMRs
//    import InstanceSpec.Annotations._
//    import InstanceSpec.Str2RefTarget
//    @instantiable
//    case class Blah() extends IsInstantiable {
//      @public val w = Wire(UInt(32.W))
//      @public val x = "Hi"
//    }
//    @instantiable
//    class AddOne extends MultiIOModule {
//      @public val in  = IO(Input(UInt(32.W)))
//      @public val out = IO(Output(UInt(32.W)))
//      @public val blah = Blah()
//      blah.w := in
//      out := blah.w + 1.U
//    }
//    class AddOneTester extends BasicTester {
//      val i = Instance(Definition(new AddOne))
//      i.in := 42.U
//      chisel3.assert(i.out === 43.U)
//      stop()
//    }
//    val annos = ChiselGeneratorAnnotation(() => new AddOneTester).elaborate
//    val addOneTester = annos.collectFirst { case d: DesignAnnotation[AddOneTester] => d.design }.get
//    val (output, _) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
//    println(output)
//    @instantiable
//    class Other extends MultiIOModule {
//      @public val x = addOneTester.i.in
//      mark(addOneTester.i.in, "addOneTester.i.in from Other")
//      mark(addOneTester.i.blah.w, "addOneTester.i.blah.w from Other")
//    }
//    class OtherTop extends MultiIOModule {
//      val i = Instance(Definition(new Other))
//      @public val x = i.x
//      mark(x, "blah")
//    }
//    val (_, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new OtherTop, args = Array("--full-stacktrace"))
//    annotations.toSeq should contain (MarkAnnotation("~AddOneTester|AddOneTester/i:AddOne>in".rt, "addOneTester.i.in"))
//    annotations.toSeq should contain (MarkAnnotation("~AddOneTester|AddOneTester/i:AddOne>w".rt, "addOneTester.i.blah.w"))
//    annotations
//  }
//  "XMR of XMR" should "get proper context" in {
//    // Issue was in the toTarget logic for XMRs
//    import InstanceSpec.Annotations._
//    @instantiable
//    class AddOne extends MultiIOModule {
//      @public val in  = IO(Input(UInt(32.W)))
//      @public val out = IO(Output(UInt(32.W)))
//      out := in + 1.U
//    }
//    @instantiable
//    class AddTwo extends MultiIOModule {
//      @public val in  = IO(Input(UInt(32.W)))
//      @public val out = IO(Output(UInt(32.W)))
//      val t = Definition(new AddOne)
//      @public val i0 = Instance(t)
//      val i1 = Instance(t)
//      @public val x = i0.in
//      i0.in := in
//      i1.in := i0.out
//      out := i1.out
//    }
//    class AddTwoTester extends BasicTester {
//      val i = Instance(Definition(new AddTwo))
//      i.in := 42.U
//      chisel3.assert(i.out === 44.U)
//      stop()
//      mark(i.x, "i.x")
//      mark(i.i0.in, "i.i0.in")
//    }
//    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddTwoTester, args = Array("--full-stacktrace"))
//    println(output)
//    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddTwoTester|AddTwoTester/i:AddTwo/i0:AddOne>in").asInstanceOf[ReferenceTarget], "i.x"))
//    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddTwoTester|AddTwoTester/i:AddTwo/i0:AddOne>in").asInstanceOf[ReferenceTarget], "i.i0.in"))
//  }
//  "Definition/Instance" should "convert to an interface?" ignore {
//    // Not sure if we should actually do this. I think its worth experimenting how far dataview can get us.
//    // Update: Talked to henry, i do think it will be necessary, but can come later.
//    // Consider the (i: Instance[Class]).as[SuperClass] syntax, and forcing people to be explicit? Or maybe we can do an similar typeclass derivation
//    //case class Parameters(s: String) extends IsLookupable
//    //trait AddInterface { self: MultiIOModule =>
//    //  val in: UInt
//    //}
//    //trait AddIO extends Bundle { }
//    //@instantiable
//    //class AddOne extends MultiIOModule with AddInterface {
//    //  @public val in = IO(Input(UInt()))
//    //}
//    //@instantiable
//    //class AddTwo extends MultiIOModule with AddInterface {
//    //  @public val in = IO(Input(UInt()))
//    //}
//    //def g(i: Instance[AddInterface]): Unit = ???
//
//    //object AddOne {
//    //  def attach(m: AddOne) = ???
//    //}
//    //val t: Definition[AddOne] = Definition(new AddOne)
//    //val i: Instance[AddOne] = Instance(t)
//    //i.as[AddInterface]
//    //i.in
//    //val m: AddOne = Module(new AddOne)
//    //f(m)
//    //def f(m: Instance[AddOne])
//  }
}
