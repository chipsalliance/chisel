// SPDX-License-Identifier: Apache-2.0

package chiselTests

import org.scalatest._

import chisel3._
import chisel3.experimental.{Analog, FixedPoint, WaivedData}
import chisel3.experimental.BundleLiterals._
import chisel3.experimental.WaivedData._
import chisel3.experimental.VecLiterals._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester
import chisel3.experimental.DataMirror
import scala.collection.immutable.SeqMap

object DirectionalBulkConnectSpec {
  class ConnectionTest[T <: Data, S <: Data](outType: S, inType: T, inDrivesOut: Boolean, op: (Data, Data) => Unit, monitorOp: Option[(Data, Data) => Unit], nTmps: Int) extends Module {
    val io = IO(new Bundle {
      val in = Flipped(inType)
      val out = Flipped(Flipped(outType)) // no clonetype, no Aligned (yet)
      val monitor = monitorOp.map(mop => {
        Output(inType)
      })
    })
    monitorOp.map(mop => {
      mop(io.monitor.get, io.in)
    })
  
    val wiresIn = Seq.fill(nTmps)(Wire(inType))
    val wiresOut = Seq.fill(nTmps)(Wire(outType))
    (Seq(io.out) ++ wiresOut ++ wiresIn).zip(wiresOut ++ wiresIn :+ io.in).foreach {
      case (l, r) => if (inDrivesOut) op(l, r) else op(r, l)
    }
  }

  def vec[T <: Data](tpe: T, n: Int = 3)(implicit c: CompileOptions) = Vec(n, tpe)
  def alignedBundle[T <: Data](fieldType: T)(implicit c: CompileOptions) = new Bundle {
    val foo = Flipped(Flipped(fieldType))
    val bar = Flipped(Flipped(fieldType))
  }
  def mixedBundle[T <: Data](fieldType: T)(implicit c: CompileOptions) = new Bundle {
    val foo = Flipped(Flipped(fieldType))
    val bar = Flipped(fieldType)
  }
  def alignedFooBundle[T <: Data](fieldType: T)(implicit c: CompileOptions) = new Bundle {
    val foo = Flipped(Flipped(fieldType))
  }
  def flippedBarBundle[T <: Data](fieldType: T)(implicit c: CompileOptions) = new Bundle {
    val bar = Flipped(fieldType)
  }

  def allElementTypes(): Seq[() => Data] = Seq(() => UInt(3.W))
  def allFieldModifiers(fieldType: () => Data): Seq[() => Data] = {
    Seq(
      fieldType,
      () => Flipped(fieldType()),
      () => Input(fieldType()),
      () => Output(fieldType())
    )
  }
  def mixedFieldModifiers(fieldType: () => Data): Seq[() => Data] = {
    allFieldModifiers(fieldType).flatMap(x => allFieldModifiers(x))
  }
  object InCompatibility {
    implicit val c = Chisel.defaultCompileOptions
    class MyCompatibilityBundle(f: () => Data) extends Bundle {
      val foo = f()
    }
  }
  class MyBundle(f: () => Data) extends Bundle {
    val baz = f()
  }
  def allBundles(fieldType: () => Data): Seq[() => Data] = {
    mixedFieldModifiers(fieldType).flatMap { f =>
      Seq(
        () => new MyBundle(f),
        () => new InCompatibility.MyCompatibilityBundle(f)
      )
    }
  }
  def allVecs(element: () => Data): Seq[() => Data] = {
    mixedFieldModifiers(element).flatMap( x => 
      Seq(() => Vec(1, x()))
    )
  }
  // 2353 types, takes a few seconds to run all of them, but it's worth it
  def allTypes(): Seq[() => Data] = {
    val elements = allElementTypes()
    val allAggs = elements.flatMap { e =>
      allVecs(e) ++ allBundles(e)
    }
    val allNestedAgg = allAggs.flatMap { e =>
      allVecs(e) ++ allBundles(e)
    }
    elements ++ allAggs ++ allNestedAgg
  }
  def getInfo(t: Data): Seq[Any] = {
    val childInfos = t match {
      case a: Aggregate => a.getElements.flatMap(getInfo)
      case other => Nil
    }
    (t, DataMirror.specifiedDirectionOf(t)) +: childInfos
  }

}

class DirectionalBulkConnectSpec extends ChiselFunSpec with Utils {
  import DirectionalBulkConnectSpec._

  def testCheck(firrtl: String, matches: Seq[String], nonMatches: Seq[String]): String = {
    val unmatched = matches.collect { 
      case m if !firrtl.contains(m) => m
    }.toList
    val badMatches = nonMatches.collect {
      case m if firrtl.contains(m) => m
    }
    assert(unmatched.isEmpty, s"Unmatched in output:\n$firrtl")
    assert(badMatches.isEmpty, s"Matched in output when shouldn't:\n$firrtl")
    firrtl
  }
  def testBuild[T <: Data, S <: Data](outType: S, inType: T)(implicit inDrivesOut: Boolean, nTmps: Int, op: (Data, Data) => Unit, monitorOp: Option[(Data, Data) => Unit]): String = {
    ChiselStage.emitChirrtl({ new ConnectionTest(outType, inType, inDrivesOut, op, monitorOp, nTmps) }, true, true)
  }
  def testException[T <: Data, S <: Data](outType: S, inType: T, messageMatches: String*)(implicit inDrivesOut: Boolean, nTmps: Int, op: (Data, Data) => Unit, monitorOp: Option[(Data, Data) => Unit]): String = {
    val x = intercept[ChiselException] {
      testBuild(outType, inType)
    }
    val message = x.getMessage()
    messageMatches.foreach { m =>
      assert(message.contains(m), "Exception has wrong error message")
    }
    message
  }
  def testDistinctTypes(tpeOut: Data, tpeIn: Data, matches: Seq[String] = Seq("io.out <= io.in"), nonMatches: Seq[String] = Nil)(implicit inDrivesOut: Boolean, nTmps: Int, op: (Data, Data) => Unit, monitorOp: Option[(Data, Data) => Unit]): String = testCheck(testBuild(tpeOut, tpeIn), matches, nonMatches)
  def test(tpeIn: Data, matches: Seq[String] = Seq("io.out <= io.in"), nonMatches: Seq[String] = Nil)(implicit inDrivesOut: Boolean, nTmps: Int, op: (Data, Data) => Unit, monitorOp: Option[(Data, Data) => Unit]): String = testDistinctTypes(tpeIn, tpeIn, matches, nonMatches)


  // (D)irectional Bulk Connect tests
  describe("(0): :<>=") {
    implicit val op: (Data, Data) => Unit = {_ :<>= _}
    implicit val monitorOp: Option[(Data, Data) => Unit] = None
    implicit val inDrivesOut = true
    implicit val nTmps = 0

    it("(0.a): Emit '<=' between identical non-Analog ground types") {
      test(Bool())
      test(UInt(16.W))
      test(SInt(16.W))
      test(Clock())
      testDistinctTypes(UInt(16.W), Bool()) // Bool inherits UInt, so this should work
      testDistinctTypes(Bool(), UInt(16.W)) // Bool inherits UInt, so this should work
    }
    it("(0.b): Emit '<=' between identical aligned aggregate types") {
      test(vec(Bool()))
      test(vec(UInt(16.W)))
      test(vec(SInt(16.W)))
      test(vec(Clock()))

      test(alignedBundle(Bool()))
      test(alignedBundle(UInt(16.W)))
      test(alignedBundle(SInt(16.W)))
      test(alignedBundle(Clock()))
    }
    it("(0.c): Emit '<=' between identical aligned aggregate types, hierarchically") {
      test(vec(vec(Bool())))
      test(vec(vec(UInt(16.W))))
      test(vec(vec(SInt(16.W))))
      test(vec(vec(Clock())))

      test(vec(alignedBundle(Bool())))
      test(vec(alignedBundle(UInt(16.W))))
      test(vec(alignedBundle(SInt(16.W))))
      test(vec(alignedBundle(Clock())))

      test(alignedBundle(vec(Bool())))
      test(alignedBundle(vec(UInt(16.W))))
      test(alignedBundle(vec(SInt(16.W))))
      test(alignedBundle(vec(Clock())))

      test(alignedBundle(alignedBundle(Bool())))
      test(alignedBundle(alignedBundle(UInt(16.W))))
      test(alignedBundle(alignedBundle(SInt(16.W))))
      test(alignedBundle(alignedBundle(Clock())))
    }
    it("(0.d): Emit '<=' between identical aggregate types with mixed flipped/aligned fields") {
      test(mixedBundle(Bool()))
      test(mixedBundle(UInt(16.W)))
      test(mixedBundle(SInt(16.W)))
      test(mixedBundle(Clock()))
    }
    it("(0.e): Emit '<=' between identical aggregate types with mixed flipped/aligned fields, hierarchically") {
      test(vec(mixedBundle(Bool())))
      test(vec(mixedBundle(UInt(16.W))))
      test(vec(mixedBundle(SInt(16.W))))
      test(vec(mixedBundle(Clock())))

      test(mixedBundle(vec(Bool())))
      test(mixedBundle(vec(UInt(16.W))))
      test(mixedBundle(vec(SInt(16.W))))
      test(mixedBundle(vec(Clock())))

      test(mixedBundle(mixedBundle(Bool())))
      test(mixedBundle(mixedBundle(UInt(16.W))))
      test(mixedBundle(mixedBundle(SInt(16.W))))
      test(mixedBundle(mixedBundle(Clock())))
    }
    it("(0.f): Throw exception between differing ground types") {
      testException(UInt(3.W), SInt(3.W), "have different types")
      testException(UInt(3.W), Clock(), "have different types")
      testException(SInt(3.W), Clock(), "have different types")
    }
    it("(0.g): Emit 'attach' between Analog types or Aggregates with Analog types") {
      test(Analog(3.W), Seq("attach (io.out, io.in)"))
      test(mixedBundle(Analog(3.W)), Seq("attach (io.out.foo, io.in.foo)", "attach (io.out.bar, io.in.bar"))
      test(vec(Analog(3.W), 2), Seq("attach (io.out[0], io.in[0])", "attach (io.out[1], io.in[1]"))
    }
    it("(0.h): Error on missing subfield/subindex from either right-hand-side or left-hand-side") {
      // Missing flip bar
      testException(mixedBundle(Bool()), alignedFooBundle(Bool()), "dangling consumer field")
      testException(alignedFooBundle(Bool()), mixedBundle(Bool()), "unassigned producer field")

      // Missing foo
      testException(mixedBundle(Bool()), flippedBarBundle(Bool()), "unassigned consumer field")
      testException(flippedBarBundle(Bool()), mixedBundle(Bool()), "dangling producer field")

      // Vec sizes don't match
      testException(vec(alignedFooBundle(Bool())),    vec(alignedFooBundle(Bool()), 4), "dangling producer field")
      testException(vec(alignedFooBundle(Bool()), 4), vec(alignedFooBundle(Bool())), "unassigned consumer field")
      testException(vec(flippedBarBundle(Bool())),    vec(flippedBarBundle(Bool()), 4), "dangling producer field")
      testException(vec(flippedBarBundle(Bool()), 4), vec(flippedBarBundle(Bool())), "unassigned consumer field")

      // Correct dangling/unassigned consumer/producer if vec has a bundle who has a flip field
      testException(vec(alignedFooBundle(Bool())), vec(mixedBundle(Bool()), 4), "dangling producer field")
      testException(vec(mixedBundle(Bool()), 4), vec(alignedFooBundle(Bool())), "unassigned consumer field")
    }
    it("(0.i): Error if different root-relative flippedness on leaf fields between right-hand-side or left-hand-side") {
      testException(mixedBundle(Bool()), alignedBundle(Bool()), "inversely oriented fields")
      testException(alignedBundle(Bool()), mixedBundle(Bool()), "inversely oriented fields")
    }
    ignore("(0.j): Emit defaultable assignments on type with default, instead of erroring with missing fields") {
      //testDistinctTypes(
      //  alignedBundle(Bool().withConnectableDefault(true.B)),
      //  alignedFooBundle(Bool()),
      //  Seq("io.out.foo <= io.in.foo", "io.out.bar <= UInt<1>(\"h1\")")
      //)
      //testDistinctTypes(
      //  alignedBundle(Bool()).withConnectableDefault{(_.bar -> true.B)},
      //  alignedFooBundle(Bool()),
      //  Seq("io.out.foo <= io.in.foo", "io.out.bar <= UInt<1>(\"h1\")")
      //)
    }
    it("(0.k): When connecting FROM DontCare, emit for aligned aggregate fields and error for flipped aggregate fields") {
      implicit val op: (Data, Data) => Unit = {(x, y) => x :<>= DontCare}
      test(UInt(3.W), Seq("io.out is invalid"))
      test(SInt(3.W), Seq("io.out is invalid"))
      test(Clock(), Seq("io.out is invalid"))
      test(Analog(3.W), Seq("io.out is invalid"))
      test(vec(Bool()), Seq("io.out[0] is invalid", "io.out[1] is invalid", "io.out[2] is invalid"))
      test(alignedBundle(Bool()), Seq("io.out.foo is invalid", "io.out.bar is invalid"))
      testException(mixedBundle(Bool()), mixedBundle(Bool()), "DontCare cannot be a connection sink")
    }
    it("(0.l): Compile without 'sink cannot be driven errors' for mixed compatibility Bundles") {
      allTypes().foreach { t => test(t(), Seq("<=")) }
    }
  }
  describe("(1): :<= ") {
    implicit val op: (Data, Data) => Unit = {_ :<= _}
    implicit val monitorOp: Option[(Data, Data) => Unit] = None
    implicit val inDrivesOut = true
    implicit val nTmps = 0

    it("(1.a): Emit '<=' between identical non-Analog ground types") {
      test(Bool())
      test(UInt(16.W))
      test(SInt(16.W))
      test(Clock())
      testDistinctTypes(UInt(16.W), Bool()) // Bool inherits UInt, so this should work
      testDistinctTypes(Bool(), UInt(16.W)) // Bool inherits UInt, so this should work
    }
    it("(1.b): Emit multiple '<=' between identical aligned aggregate types") {
      val vecMatches = Seq(
        "io.out[0] <= io.in[0]",
        "io.out[1] <= io.in[1]",
        "io.out[2] <= io.in[2]",
      )
      test(vec(Bool()), vecMatches)
      test(vec(UInt(16.W)), vecMatches)
      test(vec(SInt(16.W)), vecMatches)
      test(vec(Clock()), vecMatches)

      val bundleMatches = Seq(
        "io.out.bar <= io.in.bar",
        "io.out.foo <= io.in.foo",
      )
      test(alignedBundle(Bool()), bundleMatches)
      test(alignedBundle(UInt(16.W)), bundleMatches)
      test(alignedBundle(SInt(16.W)), bundleMatches)
      test(alignedBundle(Clock()), bundleMatches)
    }
    it("(1.c): Emit multiple '<=' between identical aligned aggregate types, hierarchically") {
      val vecVecMatches = Seq(
        "io.out[0][0] <= io.in[0][0]",
        "io.out[0][1] <= io.in[0][1]",
        "io.out[0][2] <= io.in[0][2]",
        "io.out[1][0] <= io.in[1][0]",
        "io.out[1][1] <= io.in[1][1]",
        "io.out[1][2] <= io.in[1][2]",
        "io.out[2][0] <= io.in[2][0]",
        "io.out[2][1] <= io.in[2][1]",
        "io.out[2][2] <= io.in[2][2]",
      )
      test(vec(vec(Bool())), vecVecMatches)
      test(vec(vec(UInt(16.W))), vecVecMatches)
      test(vec(vec(SInt(16.W))), vecVecMatches)
      test(vec(vec(Clock())), vecVecMatches)

      val vecBundleMatches = Seq(
        "io.out[0].bar <= io.in[0].bar",
        "io.out[0].foo <= io.in[0].foo",
        "io.out[1].bar <= io.in[1].bar",
        "io.out[1].foo <= io.in[1].foo",
        "io.out[2].bar <= io.in[2].bar",
        "io.out[2].foo <= io.in[2].foo",
      )
      test(vec(alignedBundle(Bool())), vecBundleMatches)
      test(vec(alignedBundle(UInt(16.W))), vecBundleMatches)
      test(vec(alignedBundle(SInt(16.W))), vecBundleMatches)
      test(vec(alignedBundle(Clock())), vecBundleMatches)

      val bundleVecMatches = Seq(
        "io.out.bar[0] <= io.in.bar[0]",
        "io.out.bar[1] <= io.in.bar[1]",
        "io.out.bar[2] <= io.in.bar[2]",
        "io.out.foo[0] <= io.in.foo[0]",
        "io.out.foo[1] <= io.in.foo[1]",
        "io.out.foo[2] <= io.in.foo[2]",
      )
      test(alignedBundle(vec(Bool())), bundleVecMatches)
      test(alignedBundle(vec(UInt(16.W))), bundleVecMatches)
      test(alignedBundle(vec(SInt(16.W))), bundleVecMatches)
      test(alignedBundle(vec(Clock())), bundleVecMatches)

      val bundleBundleMatches = Seq(
        "io.out.bar.bar <= io.in.bar.bar",
        "io.out.bar.foo <= io.in.bar.foo",
        "io.out.foo.bar <= io.in.foo.bar",
        "io.out.foo.foo <= io.in.foo.foo",
      )
      test(alignedBundle(alignedBundle(Bool())), bundleBundleMatches)
      test(alignedBundle(alignedBundle(UInt(16.W))), bundleBundleMatches)
      test(alignedBundle(alignedBundle(SInt(16.W))), bundleBundleMatches)
      test(alignedBundle(alignedBundle(Clock())), bundleBundleMatches)
    }
    it("(1.d): Emit '<=' between identical aggregate types with mixed flipped/aligned fields") {
      val bundleMatches = Seq("io.out.foo <= io.in.foo")
      val nonBundleMatches = Seq("io.in.bar <= io.out.bar")
      test(mixedBundle(Bool()), bundleMatches, nonBundleMatches)
      test(mixedBundle(UInt(16.W)), bundleMatches, nonBundleMatches)
      test(mixedBundle(SInt(16.W)), bundleMatches, nonBundleMatches)
      test(mixedBundle(Clock()), bundleMatches, nonBundleMatches)
    }
    it("(1.e): Emit '<=' between identical aggregate types with mixed flipped/aligned fields, hierarchically") {
      val vecBundleMatches = Seq(
        "io.out[0].foo <= io.in[0].foo",
        "io.out[1].foo <= io.in[1].foo",
        "io.out[2].foo <= io.in[2].foo",
      )
      val nonVecBundleMatches = Seq(
        "io.in[0].bar <= io.out[0].bar",
        "io.in[1].bar <= io.out[1].bar",
        "io.in[2].bar <= io.out[2].bar",
      )
      test(vec(mixedBundle(Bool())), vecBundleMatches, nonVecBundleMatches)
      test(vec(mixedBundle(UInt(16.W))), vecBundleMatches, nonVecBundleMatches)
      test(vec(mixedBundle(SInt(16.W))), vecBundleMatches, nonVecBundleMatches)
      test(vec(mixedBundle(Clock())), vecBundleMatches, nonVecBundleMatches)

      val bundleVecMatches = Seq(
        "io.out.foo[0] <= io.in.foo[0]",
        "io.out.foo[1] <= io.in.foo[1]",
        "io.out.foo[2] <= io.in.foo[2]",
      )
      val nonBundleVecMatches = Seq(
        "io.in.bar[0] <= io.out.bar[0]",
        "io.in.bar[1] <= io.out.bar[1]",
        "io.in.bar[2] <= io.out.bar[2]",
      )
      test(mixedBundle(vec(Bool())), bundleVecMatches, nonBundleVecMatches)
      test(mixedBundle(vec(UInt(16.W))), bundleVecMatches, nonBundleVecMatches)
      test(mixedBundle(vec(SInt(16.W))), bundleVecMatches, nonBundleVecMatches)
      test(mixedBundle(vec(Clock())), bundleVecMatches, nonBundleVecMatches)

      val bundleBundleMatches = Seq(
        "io.out.bar.bar <= io.in.bar.bar",
        "io.out.foo.foo <= io.in.foo.foo",
      )
      val nonBundleBundleMatches = Seq(
        "io.in.bar.foo <= io.out.bar.foo",
        "io.in.foo.bar <= io.out.foo.bar",
      )
      test(mixedBundle(mixedBundle(Bool())), bundleBundleMatches, nonBundleBundleMatches)
      test(mixedBundle(mixedBundle(UInt(16.W))), bundleBundleMatches, nonBundleBundleMatches)
      test(mixedBundle(mixedBundle(SInt(16.W))), bundleBundleMatches, nonBundleBundleMatches)
      test(mixedBundle(mixedBundle(Clock())), bundleBundleMatches, nonBundleBundleMatches)
    }
    it("(1.f): Throw exception between differing ground types") {
      testException(UInt(3.W), SInt(3.W), "have different types")
      testException(UInt(3.W), Clock(), "have different types")
      testException(SInt(3.W), Clock(), "have different types")
    }
    it("(1.g): Emit 'attach' between Analog types or Aggregates with Analog types") {
      test(Analog(3.W), Seq("attach (io.out, io.in)"))
      test(mixedBundle(Analog(3.W)), Seq("attach (io.out.foo, io.in.foo)", "attach (io.out.bar, io.in.bar"))
      test(vec(Analog(3.W), 2), Seq("attach (io.out[0], io.in[0])", "attach (io.out[1], io.in[1]"))
    }
    it("(1.h): Error on unassigned subfield/subindex from either side, but do not throw exception for dangling fields") {
      // Missing flip bar
      testException(mixedBundle(Bool()), alignedFooBundle(Bool()), "unmatched consumer field")
      testException(alignedFooBundle(Bool()), mixedBundle(Bool()), "unmatched producer field")

      // Missing foo
      testException(mixedBundle(Bool()), flippedBarBundle(Bool()), "unassigned consumer field")
      testException(flippedBarBundle(Bool()), mixedBundle(Bool()), "dangling producer field")

      // Vec sizes don't match
      testException(vec(alignedFooBundle(Bool())), vec(alignedFooBundle(Bool()), 4), "dangling producer field")
      testException(vec(alignedFooBundle(Bool()), 4), vec(alignedFooBundle(Bool())), "unassigned consumer field")
      testException(vec(flippedBarBundle(Bool())), vec(flippedBarBundle(Bool()), 4), "dangling producer field")
      testException(vec(flippedBarBundle(Bool()), 4), vec(flippedBarBundle(Bool())), "unassigned consumer field")

      // Correct dangling/unassigned consumer/producer if vec has a bundle who has a flip field
      testException(vec(alignedFooBundle(Bool())), vec(mixedBundle(Bool()), 4), "dangling producer field")
      testException(vec(mixedBundle(Bool()), 4), vec(alignedFooBundle(Bool())), "unassigned consumer field")
    }
    it("(1.i): Error if different root-relative flippedness on fields between right-hand-side or left-hand-side") {
      testException(mixedBundle(Bool()), alignedBundle(Bool()), "inversely oriented fields")
      testException(alignedBundle(Bool()), mixedBundle(Bool()), "inversely oriented fields")
    }
    ignore("(1.j): Emit defaultable assignments on type with default, instead of erroring with missing fields") {
      //testDistinctTypes(
      //  alignedBundle(Bool().withConnectableDefault(true.B)),
      //  alignedFooBundle(Bool()),
      //  Seq("io.out.foo <= io.in.foo", "io.out.bar <= UInt<1>(\"h1\")")
      //)
      //testDistinctTypes(
      //  alignedBundle(Bool()).withConnectableDefault{ (_.bar -> true.B)},
      //  alignedFooBundle(Bool()),
      //  Seq("io.out.foo <= io.in.foo", "io.out.bar <= UInt<1>(\"h1\")")
      //)
    }
    it("(1.k): When connecting FROM DontCare, emit for aligned aggregate fields and skip for flipped aggregate fields") {
      implicit val op: (Data, Data) => Unit = {(x, y) => x :<= DontCare}
      test(UInt(3.W), Seq("io.out is invalid"))
      test(SInt(3.W), Seq("io.out is invalid"))
      test(Clock(), Seq("io.out is invalid"))
      test(Analog(3.W), Seq("io.out is invalid"))
      test(vec(Bool()), Seq("io.out[0] is invalid", "io.out[1] is invalid", "io.out[2] is invalid"))
      test(alignedBundle(Bool()), Seq("io.out.foo is invalid", "io.out.bar is invalid"))
      test(mixedBundle(Bool()), Seq("io.out.foo is invalid"))
    }
    it("(1.l): Compile without 'sink cannot be driven errors' for mixed compatibility Bundles") {
      allTypes().foreach { t => test(t(), Nil) }
    }
  }
  describe("(2): :>= ") {
    implicit val op: (Data, Data) => Unit = {_ :>= _}
    implicit val monitorOp: Option[(Data, Data) => Unit] = None
    implicit val inDrivesOut = true
    implicit val nTmps = 0

    it("(2.a): Emit 'skip' between identical non-Analog ground types") {
      val skip = Seq("skip")
      test(Bool(), skip)
      test(UInt(16.W), skip)
      test(SInt(16.W), skip)
      test(Clock(), skip)
      testDistinctTypes(UInt(16.W), Bool(), skip) // Bool inherits UInt, so this should work
      testDistinctTypes(Bool(), UInt(16.W), skip) // Bool inherits UInt, so this should work
    }
    it("(2.b): Emit 'skip' between identical aligned aggregate types") {
      val skip = Seq("skip")
      test(vec(Bool()), skip)
      test(vec(UInt(16.W)), skip)
      test(vec(SInt(16.W)), skip)
      test(vec(Clock()), skip)

      test(alignedBundle(Bool()), skip)
      test(alignedBundle(UInt(16.W)), skip)
      test(alignedBundle(SInt(16.W)), skip)
      test(alignedBundle(Clock()), skip)
    }
    it("(2.c): Emit 'skip' between identical aligned aggregate types, hierarchically") {
      val skip = Seq("skip")

      test(vec(vec(Bool())), skip)
      test(vec(vec(UInt(16.W))), skip)
      test(vec(vec(SInt(16.W))), skip)
      test(vec(vec(Clock())), skip)

      test(vec(alignedBundle(Bool())), skip)
      test(vec(alignedBundle(UInt(16.W))), skip)
      test(vec(alignedBundle(SInt(16.W))), skip)
      test(vec(alignedBundle(Clock())), skip)

      test(alignedBundle(vec(Bool())), skip)
      test(alignedBundle(vec(UInt(16.W))), skip)
      test(alignedBundle(vec(SInt(16.W))), skip)
      test(alignedBundle(vec(Clock())), skip)

      test(alignedBundle(alignedBundle(Bool())), skip)
      test(alignedBundle(alignedBundle(UInt(16.W))), skip)
      test(alignedBundle(alignedBundle(SInt(16.W))), skip)
      test(alignedBundle(alignedBundle(Clock())), skip)
    }
    it("(2.d): Emit '<=' between identical aggregate types with mixed flipped/aligned fields") {
      val bundleMatches = Seq("io.in.bar <= io.out.bar")
      val nonBundleMatches = Seq("io.out.foo <= io.in.foo")
      test(mixedBundle(Bool()), bundleMatches, nonBundleMatches)
      test(mixedBundle(UInt(16.W)), bundleMatches, nonBundleMatches)
      test(mixedBundle(SInt(16.W)), bundleMatches, nonBundleMatches)
      test(mixedBundle(Clock()), bundleMatches, nonBundleMatches)
    }
    it("(2.e): Emit '<=' between identical aggregate types with mixed flipped/aligned fields, hierarchically") {
      val vecBundleMatches = Seq(
        "io.in[0].bar <= io.out[0].bar",
        "io.in[1].bar <= io.out[1].bar",
        "io.in[2].bar <= io.out[2].bar",
      )
      val nonVecBundleMatches = Seq(
        "io.out[0].foo <= io.in[0].foo",
        "io.out[1].foo <= io.in[1].foo",
        "io.out[2].foo <= io.in[2].foo",
      )
      test(vec(mixedBundle(Bool())), vecBundleMatches, nonVecBundleMatches)
      test(vec(mixedBundle(UInt(16.W))), vecBundleMatches, nonVecBundleMatches)
      test(vec(mixedBundle(SInt(16.W))), vecBundleMatches, nonVecBundleMatches)
      test(vec(mixedBundle(Clock())), vecBundleMatches, nonVecBundleMatches)

      val bundleVecMatches = Seq(
        "io.in.bar[0] <= io.out.bar[0]",
        "io.in.bar[1] <= io.out.bar[1]",
        "io.in.bar[2] <= io.out.bar[2]",
      )
      val nonBundleVecMatches = Seq(
        "io.out.foo[0] <= io.in.foo[0]",
        "io.out.foo[1] <= io.in.foo[1]",
        "io.out.foo[2] <= io.in.foo[2]",
      )
      test(mixedBundle(vec(Bool())), bundleVecMatches, nonBundleVecMatches)
      test(mixedBundle(vec(UInt(16.W))), bundleVecMatches, nonBundleVecMatches)
      test(mixedBundle(vec(SInt(16.W))), bundleVecMatches, nonBundleVecMatches)
      test(mixedBundle(vec(Clock())), bundleVecMatches, nonBundleVecMatches)

      val bundleBundleMatches = Seq(
        "io.in.bar.foo <= io.out.bar.foo",
        "io.in.foo.bar <= io.out.foo.bar",
      )
      val nonBundleBundleMatches = Seq(
        "io.out.bar.bar <= io.in.bar.bar",
        "io.out.foo.foo <= io.in.foo.foo",
      )
      test(mixedBundle(mixedBundle(Bool())), bundleBundleMatches, nonBundleBundleMatches)
      test(mixedBundle(mixedBundle(UInt(16.W))), bundleBundleMatches, nonBundleBundleMatches)
      test(mixedBundle(mixedBundle(SInt(16.W))), bundleBundleMatches, nonBundleBundleMatches)
      test(mixedBundle(mixedBundle(Clock())), bundleBundleMatches, nonBundleBundleMatches)
    }
    it("(2.f): Throw exception between differing ground types") {
      testDistinctTypes(UInt(3.W), SInt(3.W), Seq("skip"), Seq("<="))
      testDistinctTypes(UInt(3.W), Clock(), Seq("skip"), Seq("<="))
      testDistinctTypes(SInt(3.W), Clock(), Seq("skip"), Seq("<="))
    }
    it("(2.g): Emit 'attach' between Analog types or Aggregates with Analog types") {
      test(Analog(3.W), Seq("attach (io.out, io.in)"))
      test(mixedBundle(Analog(3.W)), Seq("attach (io.out.foo, io.in.foo)", "attach (io.out.bar, io.in.bar"))
      test(vec(Analog(3.W), 2), Seq("attach (io.out[0], io.in[0])", "attach (io.out[1], io.in[1]"))
    }
    it("(2.h): Error on unassigned subfield/subindex from either side, and throw exception for dangling fields") {
      // Missing flip bar
      testException(mixedBundle(Bool()), alignedFooBundle(Bool()), "dangling consumer field")
      testException(alignedFooBundle(Bool()), mixedBundle(Bool()), "unassigned producer field")

      // Missing foo
      testException(mixedBundle(Bool()), flippedBarBundle(Bool()), "unmatched consumer field")
      testException(flippedBarBundle(Bool()), mixedBundle(Bool()), "unmatched producer field")

      // Vec sizes don't match
      testException(vec(alignedFooBundle(Bool())), vec(alignedFooBundle(Bool()), 4), "unmatched producer field")
      testException(vec(alignedFooBundle(Bool()), 4), vec(alignedFooBundle(Bool())), "unmatched consumer field")
      testException(vec(flippedBarBundle(Bool())), vec(flippedBarBundle(Bool()), 4), "unmatched producer field")
      testException(vec(flippedBarBundle(Bool()), 4), vec(flippedBarBundle(Bool())), "unmatched consumer field")

      // Correct dangling/unassigned consumer/producer if vec has a bundle who has a flip field
      testException(vec(flippedBarBundle(Bool())), vec(mixedBundle(Bool()), 4), "unmatched producer field")
      testException(vec(mixedBundle(Bool()), 4), vec(flippedBarBundle(Bool())), "unmatched consumer field")
    }
    it("(2.i): Error if different root-relative flippedness on fields between right-hand-side or left-hand-side") {
      testException(mixedBundle(Bool()), alignedBundle(Bool()), "inversely oriented fields")
      testException(alignedBundle(Bool()), mixedBundle(Bool()), "inversely oriented fields")
    }
    it("(2.k): When connecting TO DontCare, error for aligned aggregate fields and error for flipped aggregate fields") {
      implicit val op: (Data, Data) => Unit = {(x, y) => DontCare :>= y}
      test(UInt(3.W), Seq("skip"))
      test(SInt(3.W), Seq("skip"))
      test(Clock(), Seq("skip"))
      test(Analog(3.W), Seq("skip"))
      test(vec(Bool()), Seq("skip"))
      test(alignedBundle(Bool()), Seq("skip"))
      test(mixedBundle(Bool()), Seq("io.in.bar is invalid"))
    }
    it("(2.l): Compile without 'sink cannot be driven errors' for mixed compatibility Bundles") {
      allTypes().foreach { t => test(t(), Nil) }
    }
  }
  describe("(3): :#= ") {
    implicit val op: (Data, Data) => Unit = {_ :<>= _}
    implicit val monitorOp: Option[(Data, Data) => Unit] = Some({_ :#= _})
    implicit val inDrivesOut = true
    implicit val nTmps = 0

    it("(3.a): Emit '<=' between identical non-Analog ground types") {
      test(Bool())
      test(UInt(16.W))
      test(SInt(16.W))
      test(Clock())
      testDistinctTypes(UInt(16.W), Bool()) // Bool inherits UInt, so this should work
      testDistinctTypes(Bool(), UInt(16.W)) // Bool inherits UInt, so this should work
    }
    it("(3.b): Emit multiple '<=' between identical aligned aggregate types") {
      val vecMatches = Seq(
        "io.monitor[0] <= io.in[0]",
        "io.monitor[1] <= io.in[1]",
        "io.monitor[2] <= io.in[2]",
      )
      test(vec(Bool()), vecMatches)
      test(vec(UInt(16.W)), vecMatches)
      test(vec(SInt(16.W)), vecMatches)
      test(vec(Clock()), vecMatches)

      val bundleMatches = Seq(
        "io.monitor.bar <= io.in.bar",
        "io.monitor.foo <= io.in.foo",
      )
      test(alignedBundle(Bool()), bundleMatches)
      test(alignedBundle(UInt(16.W)), bundleMatches)
      test(alignedBundle(SInt(16.W)), bundleMatches)
      test(alignedBundle(Clock()), bundleMatches)
    }
    it("(3.c): Emit multiple '<=' between identical aligned aggregate types, hierarchically") {
      val vecVecMatches = Seq(
        "io.monitor[0][0] <= io.in[0][0]",
        "io.monitor[0][1] <= io.in[0][1]",
        "io.monitor[0][2] <= io.in[0][2]",
        "io.monitor[1][0] <= io.in[1][0]",
        "io.monitor[1][1] <= io.in[1][1]",
        "io.monitor[1][2] <= io.in[1][2]",
        "io.monitor[2][0] <= io.in[2][0]",
        "io.monitor[2][1] <= io.in[2][1]",
        "io.monitor[2][2] <= io.in[2][2]",
      )
      test(vec(vec(Bool())), vecVecMatches)
      test(vec(vec(UInt(16.W))), vecVecMatches)
      test(vec(vec(SInt(16.W))), vecVecMatches)
      test(vec(vec(Clock())), vecVecMatches)

      val vecBundleMatches = Seq(
        "io.monitor[0].bar <= io.in[0].bar",
        "io.monitor[0].foo <= io.in[0].foo",
        "io.monitor[1].bar <= io.in[1].bar",
        "io.monitor[1].foo <= io.in[1].foo",
        "io.monitor[2].bar <= io.in[2].bar",
        "io.monitor[2].foo <= io.in[2].foo",
      )
      test(vec(alignedBundle(Bool())), vecBundleMatches)
      test(vec(alignedBundle(UInt(16.W))), vecBundleMatches)
      test(vec(alignedBundle(SInt(16.W))), vecBundleMatches)
      test(vec(alignedBundle(Clock())), vecBundleMatches)

      val bundleVecMatches = Seq(
        "io.monitor.bar[0] <= io.in.bar[0]",
        "io.monitor.bar[1] <= io.in.bar[1]",
        "io.monitor.bar[2] <= io.in.bar[2]",
        "io.monitor.foo[0] <= io.in.foo[0]",
        "io.monitor.foo[1] <= io.in.foo[1]",
        "io.monitor.foo[2] <= io.in.foo[2]",
      )
      test(alignedBundle(vec(Bool())), bundleVecMatches)
      test(alignedBundle(vec(UInt(16.W))), bundleVecMatches)
      test(alignedBundle(vec(SInt(16.W))), bundleVecMatches)
      test(alignedBundle(vec(Clock())), bundleVecMatches)

      val bundleBundleMatches = Seq(
        "io.monitor.bar.bar <= io.in.bar.bar",
        "io.monitor.bar.foo <= io.in.bar.foo",
        "io.monitor.foo.bar <= io.in.foo.bar",
        "io.monitor.foo.foo <= io.in.foo.foo",
      )
      test(alignedBundle(alignedBundle(Bool())), bundleBundleMatches)
      test(alignedBundle(alignedBundle(UInt(16.W))), bundleBundleMatches)
      test(alignedBundle(alignedBundle(SInt(16.W))), bundleBundleMatches)
      test(alignedBundle(alignedBundle(Clock())), bundleBundleMatches)
    }
    it("(3.d): Emit '<=' between identical aggregate types with mixed flipped/aligned fields") {
      val bundleMatches = Seq("io.monitor.foo <= io.in.foo", "io.monitor.bar <= io.in.bar")
      test(mixedBundle(Bool()), bundleMatches)
      test(mixedBundle(UInt(16.W)), bundleMatches)
      test(mixedBundle(SInt(16.W)), bundleMatches)
      test(mixedBundle(Clock()), bundleMatches)
    }
    it("(3.e): Emit '<=' between identical aggregate types with mixed flipped/aligned fields, hierarchically") {
      val vecBundleMatches = Seq(
        "io.monitor[0].foo <= io.in[0].foo",
        "io.monitor[1].foo <= io.in[1].foo",
        "io.monitor[2].foo <= io.in[2].foo",
        "io.monitor[0].bar <= io.in[0].bar",
        "io.monitor[1].bar <= io.in[1].bar",
        "io.monitor[2].bar <= io.in[2].bar",
      )
      test(vec(mixedBundle(Bool())), vecBundleMatches)
      test(vec(mixedBundle(UInt(16.W))), vecBundleMatches)
      test(vec(mixedBundle(SInt(16.W))), vecBundleMatches)
      test(vec(mixedBundle(Clock())), vecBundleMatches)

      val bundleVecMatches = Seq(
        "io.monitor.foo[0] <= io.in.foo[0]",
        "io.monitor.foo[1] <= io.in.foo[1]",
        "io.monitor.foo[2] <= io.in.foo[2]",
        "io.monitor.bar[0] <= io.in.bar[0]",
        "io.monitor.bar[1] <= io.in.bar[1]",
        "io.monitor.bar[2] <= io.in.bar[2]",
      )
      test(mixedBundle(vec(Bool())), bundleVecMatches)
      test(mixedBundle(vec(UInt(16.W))), bundleVecMatches)
      test(mixedBundle(vec(SInt(16.W))), bundleVecMatches)
      test(mixedBundle(vec(Clock())), bundleVecMatches)

      val bundleBundleMatches = Seq(
        "io.monitor.bar.bar <= io.in.bar.bar",
        "io.monitor.foo.foo <= io.in.foo.foo",
        "io.monitor.bar.foo <= io.in.bar.foo",
        "io.monitor.foo.bar <= io.in.foo.bar",
      )
      test(mixedBundle(mixedBundle(Bool())), bundleBundleMatches)
      test(mixedBundle(mixedBundle(UInt(16.W))), bundleBundleMatches)
      test(mixedBundle(mixedBundle(SInt(16.W))), bundleBundleMatches)
      test(mixedBundle(mixedBundle(Clock())), bundleBundleMatches)
    }
    it("(3.f): Throw exception between differing ground types") {
      testException(UInt(3.W), SInt(3.W), "have different types")
      testException(UInt(3.W), Clock(), "have different types")
      testException(SInt(3.W), Clock(), "have different types")
    }
    it("(3.g): Emit 'attach' between Analog types or Aggregates with Analog types") {
      implicit val op: (Data, Data) => Unit = {_ :#= _}
      implicit val monitorOp: Option[(Data, Data) => Unit] = None
      test(Analog(3.W), Seq("attach (io.out, io.in)"))
      test(mixedBundle(Analog(3.W)), Seq("attach (io.out.foo, io.in.foo)", "attach (io.out.bar, io.in.bar"))
      test(vec(Analog(3.W), 2), Seq("attach (io.out[0], io.in[0])", "attach (io.out[1], io.in[1]"))
    }
    it("(3.h): Error on unassigned or dangling subfield/subindex from either side") {
      // Missing flip bar
      implicit val op: (Data, Data) => Unit = {_ :#= _}
      implicit val monitorOp: Option[(Data, Data) => Unit] = None
      testException(mixedBundle(Bool()), alignedFooBundle(Bool()), "unmatched consumer field")
      testException(alignedFooBundle(Bool()), mixedBundle(Bool()), "unmatched producer field")

      // Missing foo
      testException(mixedBundle(Bool()), flippedBarBundle(Bool()), "cannot be written from module")
      testException(flippedBarBundle(Bool()), mixedBundle(Bool()), "cannot be written from module")

      // Vec sizes don't match
      testException(vec(alignedFooBundle(Bool())), vec(alignedFooBundle(Bool()), 4), "dangling producer field")
      testException(vec(alignedFooBundle(Bool()), 4), vec(alignedFooBundle(Bool())), "unassigned consumer field")
      testException(vec(flippedBarBundle(Bool())), vec(flippedBarBundle(Bool()), 4), "cannot be written from module")
      testException(vec(flippedBarBundle(Bool()), 4), vec(flippedBarBundle(Bool())), "cannot be written from module")

      // Correct dangling/unassigned consumer/producer if vec has a bundle who has a flip field
      testException(vec(alignedFooBundle(Bool())), vec(mixedBundle(Bool()), 4), "unmatched producer field", "dangling producer field")
      testException(vec(mixedBundle(Bool()), 4), vec(alignedFooBundle(Bool())), "unmatched consumer field", "unassigned consumer field")
    }
    it("(3.i): Always assign to consumer regardless of orientation") {
      implicit val op: (Data, Data) => Unit = {_ :#= _}
      implicit val monitorOp: Option[(Data, Data) => Unit] = None
      testException(mixedBundle(Bool()), alignedBundle(Bool()), "cannot be written from module")
      testDistinctTypes(alignedBundle(Bool()), mixedBundle(Bool()), Seq(
        "io.out.foo <= io.in.foo",
        "io.out.bar <= io.in.bar"
      ))
    }
    it("(3.k): When connecting FROM DontCare, emit for aligned aggregate fields and emit for flipped aggregate fields") {
      implicit val op: (Data, Data) => Unit = {(x, y) => x :#= DontCare}
      implicit val monitorOp: Option[(Data, Data) => Unit] = None
      test(UInt(3.W), Seq("io.out is invalid"))
      test(SInt(3.W), Seq("io.out is invalid"))
      test(Clock(), Seq("io.out is invalid"))
      test(Analog(3.W), Seq("io.out is invalid"))
      test(vec(Bool()), Seq("io.out[0] is invalid", "io.out[1] is invalid", "io.out[2] is invalid"))
      test(alignedBundle(Bool()), Seq("io.out.foo is invalid", "io.out.bar is invalid"))
      test(mixedBundle(Bool()), Seq("io.out.foo is invalid", "io.out.bar is invalid"))
    }
    it("(3.l): Compile without 'sink cannot be driven errors' for mixed compatibility Bundles") {
      allTypes().foreach { t => test(t(), Seq("<=")) }
    }
  }

  describe("(?): Use Cases") {
    it("(?.a.a) Initalize wires with default values and :<>= to connect wires of mixed directions") {
      class MixedBundle extends Bundle {
        val foo = UInt(3.W)
        val bar = Flipped(UInt(3.W))
      }
      class MyModule extends Module {
        val lit = new MixedBundle().Lit(
          _.foo            -> 0.U,
          _.bar            -> 1.U,
        )
        val w0 = Wire(new MixedBundle)
        w0 :#= lit
        val w1 = Wire(new MixedBundle)
        w1 :#= lit
        w1 :<>= w0
      }
      val out = ChiselStage.emitChirrtl({ new MyModule() }, true, true)
      testCheck(out, Seq(
        """wire w0 : { foo : UInt<3>, flip bar : UInt<3>}""",
        """w0.bar <= UInt<1>("h1")""",
        """w0.foo <= UInt<1>("h0")""",
        """wire w1 : { foo : UInt<3>, flip bar : UInt<3>}""",
        """w1.bar <= UInt<1>("h1")""",
        """w1.foo <= UInt<1>("h0")""",
        """w1 <= w0""",
      ), Nil)
    }
    it("(?.a.b) Initialize wires with different optional fields with :#= and using :<>= to connect wires of mixed directions, waiving extra field for being unassigned or dangling") {
      class MixedBundle extends Bundle {
        val foo = UInt(3.W)
        val bar = Flipped(UInt(3.W))
      }
      class Parent(hasOptional: Boolean) extends Bundle {
        val necessary = new MixedBundle
        val optional = if(hasOptional) Some(new MixedBundle) else None
      }
      class MyModule extends Module {
        val lit = new Parent(true).Lit(
          _.necessary.foo     -> 0.U,
          _.necessary.bar     -> 1.U,
          _.optional.get.foo  -> 2.U,
          _.optional.get.bar  -> 3.U
        )
        val hasOptional = Wire(new Parent(true))
        hasOptional :#= lit
        val lacksOptional = Wire(new Parent(false))
        lacksOptional :#= lit.waive(_.optional.get)
        hasOptional.waive(_.optional.get) :<>= lacksOptional
      }
      val out = ChiselStage.emitChirrtl({ new MyModule() }, true, true)
      testCheck(out, Seq(
        """lacksOptional.necessary.bar <= hasOptional.necessary.bar""",
        """hasOptional.necessary.foo <= lacksOptional.necessary.foo""",
      ), Nil)
    }
    it("(?.b) Waiving ok-to-dangle field connecting a wider bus to a narrower bus") {
      class ReadyValid extends Bundle {
        val valid = Bool()
        val ready = Flipped(Bool())
      }
      class Decoupled extends ReadyValid {
        val data = UInt(32.W)
      }
      class MyModule extends Module {
        val in  = IO(Flipped(new Decoupled()))
        val out = IO(new ReadyValid())
        out :<>= in.waiveAs[ReadyValid](_.data)
      }
      val out = ChiselStage.emitChirrtl({ new MyModule() }, true, true)
      testCheck(out, Seq(
        """out.valid <= in.valid""",
        """in.ready <= out.ready""",
      ), Nil)
    }
    it("(?.c) Waiving ok-to-unassign field connecting a narrower bus to a wider bus, with defaults for unassigned fields set via last connect semantics") {
      class ReadyValid extends Bundle {
        val valid = Bool()
        val ready = Flipped(Bool())
      }
      class Decoupled extends ReadyValid {
        val data = UInt(32.W)
      }
      class MyModule extends Module {
        val in  = IO(Flipped(new ReadyValid()))
        val out = IO(new Decoupled())
        out :<= (new Decoupled()).Lit(_.data -> 0.U)
        out.waiveAs[ReadyValid](_.data) :<>= in
      }
      val out = ChiselStage.emitChirrtl({ new MyModule() }, true, true)
      testCheck(out, Seq(
        """out.data <= UInt<1>("h0")""",
        """in.ready <= out.ready""",
        """out.valid <= in.valid""",
      ), Nil)
    }
    it("(?.d) Waiving ok-to-unassign field connecting a narrower bus to a wider bus will error if no default specified") {
      class ReadyValid extends Bundle {
        val valid = Bool()
        val ready = Flipped(Bool())
      }
      class Decoupled extends ReadyValid {
        val data = UInt(32.W)
      }
      class MyModule extends Module {
        val in  = IO(Flipped(new ReadyValid()))
        val out = IO(new Decoupled())
        (out: ReadyValid) :<>= in
      }
      intercept[Exception] {
        ChiselStage.emitVerilog({ new MyModule() }, true, true)
      }
    }
    it("(?.d) A structurally identical but fully aligned monitor version of a bundle can easily be connected to") {
      class Decoupled extends Bundle {
        val valid = Bool()
        val ready = Flipped(Bool())
        val data = UInt(32.W)
      }
      class MyModule extends Module {
        val in  = IO(Flipped(new Decoupled()))
        val out = IO(new Decoupled())
        val monitor = IO(Output(new Decoupled()))
        out :<>= in
        monitor :#= in
      }
      val out = ChiselStage.emitChirrtl({ new MyModule() }, true, true)
      testCheck(out, Seq(
        """monitor.data <= in.data""",
        """monitor.ready <= in.ready""",
        """monitor.valid <= in.valid""",
      ), Nil)
    }
    it("(?.e) A structurally different and fully aligned monitor version of a bundle can easily be connected to, provided missing fields are ok-to-dangle") {
      class ReadyValid extends Bundle {
        val valid = Bool()
        val ready = Flipped(Bool())
      }
      class Decoupled extends ReadyValid {
        val data = UInt(32.W)
      }
      class MyModule extends Module {
        val in  = IO(Flipped(new Decoupled()))
        val out = IO(new Decoupled())
        val monitor = IO(Output(new ReadyValid()))
        out :<>= in
        monitor :#= in.waiveAs[ReadyValid](_.data)
      }
      val out = ChiselStage.emitChirrtl({ new MyModule() }, true, true)
      testCheck(out, Seq(
        """monitor.ready <= in.ready""",
        """monitor.valid <= in.valid""",
      ), Nil)
    }
    it("(?.f) Discarding echo bits is ok if waived ok-to-dangle (waived dangles)") {
      class Decoupled extends Bundle {
        val valid = Bool()
        val ready = Flipped(Bool())
        val data = UInt(32.W)
      }
      class DecoupledEcho extends Decoupled {
        val echo = Flipped(UInt(3.W))
      }
      class MyModule extends Module {
        val in  = IO(Flipped(new Decoupled()))
        val out = IO(new DecoupledEcho())
        out.waiveAs[Decoupled](_.echo) :<>= in
      }
      val out = ChiselStage.emitChirrtl({ new MyModule() }, true, true)
      testCheck(out, Seq(
        """in.ready <= out.ready""",
        """out.valid <= in.valid""",
        """out.data <= in.data""",
      ), Nil)
    }
    it("(?.g) Discarding echo bits is an error if not waived (dangles default to errors)") {
      class Decoupled extends Bundle {
        val valid = Bool()
        val ready = Flipped(Bool())
        val data = UInt(32.W)
      }
      class DecoupledEcho extends Decoupled {
        val echo = Flipped(UInt(3.W))
      }
      class MyModule extends Module {
        val in  = IO(Flipped(new Decoupled()))
        val out = IO(new DecoupledEcho())
        (out: Decoupled) :<>= in
      }
      intercept[Exception] { ChiselStage.emitChirrtl({ new MyModule() }, true, true) }
    }
    it("(?.h) Partial connect on records") {
      class BoolRecord(fields: String*) extends Record {
        val elements = SeqMap(fields.map(f => f -> Bool()):_*)
        override def cloneType = new BoolRecord(fields:_*).asInstanceOf[this.type]
      }
      class MyModule extends Module {
        val in  = IO(Flipped(new BoolRecord("a", "b")))
        val out = IO(new BoolRecord("b", "c"))
        out :<!>= in
      }
      println(ChiselStage.emitChirrtl({ new MyModule() }, true, true))
    }
    ignore("(?.h) (Example required) -     ok for non-waived dangling bits in a :<=, because :<= is less strict for dangles") { }
    ignore("(?.i) (Example required) - not ok for non-waived unassigned bits in a :<= are ok, because :<= is strict for unassigns") { }
    ignore("(?.j) (Example required) -     ok for non-waived dangling bits in a :>=, because :>= is less strict for dangles") { }
    ignore("(?.k) (Example required) - not ok for non-waived unassigned bits in a :>= are ok, because :>= is strict for unassigns") { }

    /*
     1) c :<= p
      a) [ERROR(waivable)]        unassigned aligned c field
      b) [ERROR(mismatchedflips)] aligned c field and flipped p field
      c) [ERROR(mismatchedflips)] flipped c field and aligned p field
      d) [ERROR(waivable)]        dangling aligned p fields (no c field)
      e) [Ok(prolly)]             dangling (and ignored due to :<= operator) flipped c fields (no p field)
      f) [Ok(prolly)]             unassigned (and ignored due to :<= operator) flipped p fields (no c field)
     2) c :#= w (composition breaks from 1.b and 1.c)
       c :<= w
       w :>= c
     3) m :#= p (composition breaks from 1.b and 1.c)
       m :<= p
       p :>= m
     4) c :<>= p (composition ok)
       c :<= p
       c :>= p
     */ 
  }
  describe("TODO: Unit tests") {
    ignore("(?.?) x :<>= DontCare") { }
    ignore("(?.?) x :<= DontCare") { }
    ignore("(?.?) x :>= DontCare") { }
    ignore("(?.?) x :#= DontCare") { }
  }
  describe("(9) Using WaivedData examples") {
    import scala.collection.immutable.SeqMap
    class Decoupled(val hasData: Boolean) extends Bundle {
      val valid = Bool()
      val ready = Flipped(Bool())
      val data = if(hasData) Some(UInt(32.W)) else None
    }
    class BundleMap(fields: SeqMap[String, () => Data]) extends Record {
      val elements = fields.map { case (name, gen) => name -> gen()}
      override def cloneType = new BundleMap(fields).asInstanceOf[this.type]
    }
    object BundleMap {
      import Connectable._
      def waive[T <: Data](d: T): WaivedData[T] = {
        val bundleMapElements = DataMirror.collectDeep(d) { case b: BundleMap => b.getElements }
        println(bundleMapElements)
        WaivedData(d, bundleMapElements.flatten.toSet)
      }
    }
    class DecoupledGen[T <: Data](val gen: () => T) extends Bundle {
      val valid = Bool()
      val ready = Flipped(Bool())
      val data = gen()
    }
    it("(9.c) Using waive does work for nested field") {
      class NestedDecoupled(val hasData: Boolean) extends Bundle {
        val foo = new Decoupled(hasData)
      }
      class MyModule extends Module {
        val in  = IO(Flipped(new NestedDecoupled(true)))
        val out = IO(new NestedDecoupled(false))
        out :<>= in.waiveAll { case d: Decoupled if d.data.nonEmpty => d.data.get }
      }
      println(ChiselStage.emitChirrtl({ new MyModule() }, true, true))
    }
    it("(9.e) Inline waiver things") {
      class MyModule extends Module {
        val in  = IO(Flipped(new Decoupled(true)))
        val out = IO(new Decoupled(false))
        out :<>= in.waive(_.data.get)
      }
      println(ChiselStage.emitChirrtl({ new MyModule() }, true, true))
    }
    it("(9.f) BundleMap example can use programmatic waiving") {
      class MyModule extends Module {
        def ab = new BundleMap(SeqMap(
          "a" -> (() => UInt(2.W)),
          "b" -> (() => UInt(2.W))
        ))
        def bc = new BundleMap(SeqMap(
          "b" -> (() => UInt(2.W)),
          "c" -> (() => UInt(2.W))
        ))
        val in  = IO(Flipped(new DecoupledGen(() => ab)))
        val out = IO(new DecoupledGen(() => bc))
        //Programmatic
        BundleMap.waive(out) :<>= BundleMap.waive(in)
      }
      println(ChiselStage.emitChirrtl({ new MyModule() }, true, true))
    }
    it("(9.g) Somehow return the unused fields?") {
      class MyModule extends Module {
        def ab = new BundleMap(SeqMap(
          "a" -> (() => UInt(2.W)),
          "b" -> (() => UInt(2.W))
        ))
        def bc = new BundleMap(SeqMap(
          "b" -> (() => UInt(2.W)),
          "c" -> (() => UInt(2.W))
        ))
        val in  = IO(Flipped(new DecoupledGen(() => ab)))
        val out = IO(new DecoupledGen(() => bc))
        out :<= (chiselTypeOf(out).Lit(_.data.elements("b") -> 1.U, _.data.elements("c") -> 1.U))
        //Programmatic
        val (waivedOut: WaivedData[DecoupledGen[BundleMap]], waivedIn: WaivedData[DecoupledGen[BundleMap]]) = WaivedData.waiveUnmatched(out, in)
        waivedOut :<>= waivedIn
      }
      println(ChiselStage.emitChirrtl({ new MyModule() }, true, true))
    }
    it("(9.h) Create WaivedData to connect to") {
      class MyModule extends Module {
        def ab = new BundleMap(SeqMap(
          "a" -> (() => UInt(2.W)),
          "b" -> (() => UInt(2.W))
        ))
        def bc = new BundleMap(SeqMap(
          "b" -> (() => UInt(2.W)),
          "c" -> (() => UInt(2.W))
        ))
        val in  = IO(Flipped(new DecoupledGen(() => ab)))
        val out = IO(new DecoupledGen(() => bc))
        out :<= (chiselTypeOf(out).Lit(_.data.elements("b") -> 1.U, _.data.elements("c") -> 1.U))
        //Programmatic
        BundleMap.waive(out) :<>= BundleMap.waive(in)
      }
      println(ChiselStage.emitChirrtl({ new MyModule() }, true, true))
    }
    it("(9.i) (Good or bad?) Mismatched aggregate containing backpressure must be waived for :<=") {
      // My concern with this use-case is if you have an unmatched aggregate field, but it only contains fields that your operator would ignore anyways, should you error?
      //  - For the simplicity of reasoning about the operator semantics, I think the answer is yes because erroring is now a local decision (does not depend on the child field type)
      //  - I just want to make sure, so this example kind of demonstrates that I think the behavior is sensible
      //  - In addition, with the 'waive' feature, it's very straightforward to make the operator do what you want it to do, in this case, and the explicitness is good.
      class OnlyBackPressure extends Bundle {
        val ready = Flipped(UInt(3.W))
      }
      class MyModule extends Module {
        // Have to nest in bundle because it calls the connecting-to-seq version
        val in3  = IO(Flipped(new Bundle { val v = Vec(3, new OnlyBackPressure) } ))
        val out3 = IO(new Bundle { val v = Vec(3, new OnlyBackPressure) } )
        val in2  = IO(Flipped(new Bundle { val v = Vec(2, new OnlyBackPressure) } ))
        val out2 = IO(new Bundle { val v = Vec(2, new OnlyBackPressure) } )
        // Should do nothing, but also doesn't error, which is good
        out3 :<= in3
        // Should error, unless waived
        out3.waive(_.v(2)) :>= in2
        // Should error, unless waived
        out2 :<= in3.waive(_.v(2))
      }
      println(ChiselStage.emitChirrtl({ new MyModule() }, true, true))
    }
  }
  //property("(D.a) SInt :<>= SInt should succeed") {
  //  checkTest(buildTest(SInt(16.W), SInt(16.W), true, {_ :<>= _}, 0), "io.out <= io.in")
  //}
  //property("(D.b) UInt :<>= UInt should succeed") {
  //  ChiselStage.elaborate { new CrossDirectionalBulkConnects(UInt(16.W), UInt(16.W)) }
  //}
  //property("(D.c) SInt :<>= UInt should fail") {
  //  intercept[ChiselException] { ChiselStage.elaborate { new CrossDirectionalBulkConnects(UInt(16.W), SInt(16.W)) } }
  //}
  //property("(D.d) Decoupled :<>= Decoupled should succeed") {
  //  class Decoupled extends Bundle {
  //    val bits = UInt(3.W)
  //    val valid = Bool()
  //    val ready = Flipped(Bool())
  //  }
  //  val out = ChiselStage.emitChirrtl { new CrossDirectionalBulkConnects(new Decoupled, new Decoupled) }
  //  assert(out.contains("io.out <= io.in"))
  //}
  //property("(D.d) Aggregates with same-named fields should succeed") {
  //  class Foo extends Bundle {
  //    val foo = Bool()
  //    val bar = Flipped(Bool())
  //  }
  //  class FooLike extends Bundle {
  //    val foo = Bool()
  //    val bar = Flipped(Bool())
  //  }
  //  val out = ChiselStage.emitChirrtl { new CrossDirectionalBulkConnects(new Foo, new FooLike) }
  //  assert(out.contains("io.out <= io.in"))
  //}
  //property("(D.d) Decoupled[Foo] :<>= Decoupled[Foo-Like] should succeed") {
  //  class Foo extends Bundle {
  //    val foo = Bool()
  //    val bar = Flipped(Bool())
  //  }
  //  class FooLike extends Bundle {
  //    val foo = Bool()
  //    val bar = Flipped(Bool())
  //  }
  //  class Decoupled[T <: Data](gen: => T) extends Bundle {
  //    val bits = gen
  //    val valid = Bool()
  //    val ready = Flipped(Bool())
  //  }
  //  val out = ChiselStage.emitChirrtl {
  //    new CrossDirectionalBulkConnects(new Decoupled(new Foo()), new Decoupled(new FooLike()))
  //  }
  //  assert(out.contains("io.out <= io.in"))
  //}
  //property("(D.e) different relative flips, but same absolute flippage is an error") {
  //  class X(yflip: Boolean, zflip: Boolean) extends Bundle {
  //    val y = if (yflip) Flipped(new Y(zflip)) else new Y(zflip)
  //  }
  //  class Y(flip: Boolean) extends Bundle {
  //    val z = if (flip) Flipped(Bool()) else Bool()
  //  }
  //  intercept[ChiselException] {
  //    ChiselStage.emitVerilog { new CrossDirectionalBulkConnects(new X(true, false), new X(false, true)) }
  //  }
  //}
  //property("(D.f) :<>= is not commutative.") {
  //  intercept[ChiselException] {
  //    ChiselStage.elaborate { new NotCommutativeCrossDirectionalBulkConnects(UInt(16.W), UInt(16.W)) }
  //  }
  //}
  //property("(D.g) UInt :<>= UInt should succeed with intermediate Wires") {
  //  ChiselStage.elaborate { new CrossDirectionalBulkConnectsWithWires(UInt(16.W), UInt(16.W), 1) }
  //}
  //property("(D.h) Decoupled :<>= Decoupled should succeed with intermediate Wires") {
  //  class Decoupled extends Bundle {
  //    val bits = UInt(3.W)
  //    val valid = Bool()
  //    val ready = Flipped(Bool())
  //  }
  //  val out = ChiselStage.emitChirrtl { new CrossDirectionalBulkConnectsWithWires(new Decoupled, new Decoupled, 2) }
  //  assert(out.contains("io.out <= wiresOut_0"))
  //  assert(out.contains("wiresOut_1 <= wiresIn_0"))
  //  assert(out.contains("wiresIn_1 <= io.in"))
  //}
  //property("(D.i) Aggregates :<>= with missing fields should not succeed, no matter the direction.") {
  //  class Foo extends Bundle {
  //    val foo = Bool()
  //  }
  //  class FooBar extends Bundle {
  //    val foo = Bool()
  //    val bar = Bool()
  //  }
  //  intercept[ChiselException] {
  //    ChiselStage.emitChirrtl { new CrossDirectionalBulkConnects(new Foo(), new FooBar()) }
  //  }
  //  intercept[ChiselException] {
  //    ChiselStage.emitChirrtl { new CrossDirectionalBulkConnects(new FooBar(), new Foo()) }
  //  }
  //}
  //property("(D.j) Cannot :<>= to something that is not writable.") {

  //  intercept[ChiselException] {
  //    ChiselStage.emitChirrtl { new NotWritableCrossDirectionalBulkConnects(UInt(16.W)) }
  //  }
  //}
  //property("(D.k) Can :<>= to Vecs of the same length") {
  //  val out = (new ChiselStage).emitChirrtl { new CrossDirectionalBulkConnects(Vec(3, UInt(16.W)), Vec(3, UInt(16.W))) }
  //  assert(out.contains("io.out <= io.in"))
  //}
  //property("(D.l) :<>= between Vecs of different length should not succeed, no matter the direction") {
  //  intercept[ChiselException] {
  //    ChiselStage.emitChirrtl { new CrossDirectionalBulkConnects(Vec(2, UInt(16.W)), Vec(3, UInt(16.W))) }
  //  }
  //  intercept[ChiselException] {
  //    ChiselStage.emitChirrtl { new CrossDirectionalBulkConnects(Vec(3, UInt(16.W)), Vec(2, UInt(16.W))) }
  //  }

  //}
  //property(
  //  "(D.m) :<>= is NOT equivalent to Chisel.:= in that  `A Module with missing bundle fields when compiled with the Chisel compatibility package` *should* `throw an exception` "
  //) {
  //  // This is copied from CompatibilitySpec but the := is replaced with :<>=
  //  class SmallBundle extends Bundle {
  //    val f1 = UInt(4.W)
  //    val f2 = UInt(5.W)
  //  }
  //  class BigBundle extends SmallBundle {
  //    val f3 = UInt(6.W)
  //  }

  //  class ConnectFieldMismatchModule extends Module {
  //    val io = IO(new Bundle {
  //      val in = Input((new SmallBundle))
  //      val out = Output((new BigBundle))
  //    })
  //    (io.out: Data) :<>= (io.in: Data)
  //  }
  //  intercept[ChiselException] {
  //    ChiselStage.elaborate { new ConnectFieldMismatchModule() }
  //  }
  //}

  //property(
  //  "(D.n) :#= is the same as chisel3.:=, in that fields must match and all consumer fields are written to, regardless of flippedness"
  //) {
  //  // This is copied from CompatibilitySpec but the := is replaced with :<>=
  //  class Decoupled extends Bundle {
  //    val bits = UInt(3.W)
  //    val valid = Bool()
  //    val ready = Flipped(Bool())
  //  }
  //  val out = ChiselStage.emitChirrtl { new CrossDirectionalMonoConnectsWithWires(new Decoupled, new Decoupled, 1) }
  //  assert(out.contains("wiresIn_0.bits <= io.in.bits"))
  //  assert(out.contains("wiresIn_0.valid <= io.in.valid"))
  //  assert(out.contains("wiresIn_0.ready <= io.in.ready"))
  //}

  //property("(D.o) :<>= works with DataView to connect a bundle that is a subtype") {
  //  import chisel3.experimental.dataview._

  //  class SmallBundle extends Bundle {
  //    val f1 = UInt(4.W)
  //    val f2 = UInt(5.W)
  //  }
  //  class BigBundle extends SmallBundle {
  //    val f3 = UInt(6.W)
  //  }

  //  class ConnectSupertype extends Module {
  //    val io = IO(new Bundle {
  //      val in = Input((new SmallBundle))
  //      val out = Output((new BigBundle))

  //      val foo = Input((new BigBundle))
  //      val bar = Output(new SmallBundle)
  //    })
  //    io.out := DontCare
  //    io.out.viewAsSupertype(Output(new SmallBundle)) :<>= io.in

  //    io.bar := DontCare
  //    io.bar :<>= io.foo.viewAsSupertype(Input((new SmallBundle)))
  //  }
  //  val out = (new ChiselStage).emitChirrtl(gen = new ConnectSupertype(), args = Array("--full-stacktrace"))
  //  assert(out.contains("io.out.f1 <= io.in.f1"))
  //  assert(out.contains("io.out.f2 <= io.in.f2"))
  //  assert(!out.contains("io.out.f3 <= io.in.f3"))
  //  assert(!out.contains("io.out <= io.in"))
  //  assert(!out.contains("io.out <- io.in"))

  //  assert(out.contains("io.bar.f1 <= io.foo.f1"))
  //  assert(out.contains("io.bar.f2 <= io.foo.f2"))
  //  assert(!out.contains("io.bar.f3 <= io.foo.f3"))
  //  assert(!out.contains("io.bar <= io.foo"))
  //  assert(!out.contains("io.bar <- io.foo"))

  //}
  //property("(D.p) :<>= works with DataView to connect a two Bundles with a common trait") {
  //  import chisel3.experimental.dataview._

  //  class SmallBundle extends Bundle {
  //    val common = Output(UInt(4.W))
  //    val commonFlipped = Input(UInt(4.W))
  //  }
  //  class BigA extends SmallBundle {
  //    val a = Input(UInt(6.W))
  //  }
  //  class BigB extends SmallBundle {
  //    val b = Output(UInt(6.W))
  //  }

  //  class ConnectCommonTrait extends Module {
  //    val io = IO(new Bundle {
  //      val in = Flipped(new BigA)
  //      val out = (new BigB)
  //    })
  //    io.in := DontCare
  //    io.out := DontCare
  //    io.out.viewAsSupertype(new SmallBundle) :<>= io.in.viewAsSupertype(Flipped(new SmallBundle))
  //  }
  //  val out = ChiselStage.emitChirrtl { new ConnectCommonTrait() }
  //  assert(!out.contains("io.out <= io.in"))
  //  assert(!out.contains("io.out <- io.in"))
  //  assert(out.contains("io.out.common <= io.in.common"))
  //  assert(out.contains("io.in.commonFlipped <= io.out.commonFlipped"))
  //  assert(!out.contains("io.out.b <= io.in.b"))
  //  assert(!out.contains("io.in.a  <= io.out.a"))
  //}

  //property("(D.q) :<>= works between Vec and Seq, as well as Vec and Vec") {
  //  class ConnectVecSeqAndVecVec extends Module {
  //    val a = IO(Vec(3, UInt(3.W)))
  //    val b = IO(Vec(3, UInt(3.W)))
  //    a :<>= Seq(0.U, 1.U, 2.U)
  //    b :<>= VecInit(0.U, 1.U, 2.U)
  //  }
  //  val out = ChiselStage.emitChirrtl { new ConnectVecSeqAndVecVec() }
  //  assert(out.contains("""a[0] <= UInt<1>("h0")"""))
  //  assert(out.contains("""a[1] <= UInt<1>("h1")"""))
  //  assert(out.contains("""a[2] <= UInt<2>("h2")"""))
  //  assert(out.contains("""b[0] <= _WIRE[0]"""))
  //  assert(out.contains("""b[1] <= _WIRE[1]"""))
  //  assert(out.contains("""b[2] <= _WIRE[2]"""))
  //}

  //property("(D.r) :<>= works for different missing Defaulting subfields") {
  //  trait Info extends Bundle {
  //    val info = UInt(32.W)
  //  }
  //  class InfoECC extends Info {
  //    val ecc = Defaulting(false.B)
  //  }
  //  class InfoControl extends Info {
  //    val control = Defaulting(false.B)
  //  }
  //  val firrtl = ChiselStage.emitChirrtl {
  //    new CrossDirectionalBulkConnectsWithWires(new InfoECC(): Info, new InfoControl(): Info, 1)
  //  }
  //  println(firrtl)
  //  assert(firrtl.contains("wiresOut_0.control <= UInt<1>(\"h0\")"))
  //  assert(firrtl.contains("wiresOut_0.info <= wiresIn_0.info"))
  //}
  //property("(D.s) :<>= works for different missing Defaulting subindexes") {
  //  def vecType(size: Int) = Vec(size, Defaulting(UInt(3.W), 0.U))
  //  val firrtl = ChiselStage.emitChirrtl { new CrossDirectionalBulkConnectsWithWires(vecType(2), vecType(3), 1) }
  //  println(firrtl)
  //  assert(firrtl.contains("wiresOut_0[2] <= UInt<1>(\"h0\")"))
  //  assert(firrtl.contains("wiresOut_0[1] <= wiresIn_0[1]"))
  //  assert(firrtl.contains("wiresOut_0[0] <= wiresIn_0[0]"))

  //}
}
