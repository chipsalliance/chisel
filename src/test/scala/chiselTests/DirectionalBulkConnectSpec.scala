// SPDX-License-Identifier: Apache-2.0

package chiselTests

import org.scalatest._

import chisel3._
import chisel3.experimental.{Analog, FixedPoint}
import chisel3.experimental.BundleLiterals._
import chisel3.experimental.Defaulting._
import chisel3.experimental.VecLiterals._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester

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

  def vec[T <: Data](tpe: T, n: Int = 3) = Vec(n, tpe)
  def alignedBundle[T <: Data](fieldType: T) = new Bundle {
    val foo = Flipped(Flipped(fieldType))
    val bar = Flipped(Flipped(fieldType))
  }
  def mixedBundle[T <: Data](fieldType: T) = new Bundle {
    val foo = Flipped(Flipped(fieldType))
    val bar = Flipped(fieldType)
  }
  def alignedFooBundle[T <: Data](fieldType: T) = new Bundle {
    val foo = Flipped(Flipped(fieldType))
  }
  def flippedBarBundle[T <: Data](fieldType: T) = new Bundle {
    val bar = Flipped(fieldType)
  }

}


//class CrossDirectionalMonoConnectsWithWires(inType: Data, outType: Data, nTmps: Int) extends Module {
//  val io = IO(new Bundle {
//    val in = Flipped(inType)
//    val out = Output(outType) // no clonetype, no Aligned (yet)
//  })
//  require(nTmps > 0)
//  val wiresIn = Seq.fill(nTmps)(Wire(inType))
//  val wiresOut = Seq.fill(nTmps)(Wire(outType))
//  (Seq(io.out) ++ wiresOut ++ wiresIn).zip(wiresOut ++ wiresIn :+ io.in).foreach {
//    case (l, r) =>
//      l :#= r
//  }
//}

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
      testException(vec(flippedBarBundle(Bool())),    vec(flippedBarBundle(Bool()), 4), "unassigned producer field")
      testException(vec(flippedBarBundle(Bool()), 4), vec(flippedBarBundle(Bool())), "dangling consumer field")

      // Correct dangling/unassigned consumer/producer if vec has a bundle who has a flip field
      testException(vec(alignedFooBundle(Bool())), vec(mixedBundle(Bool()), 4), "unassigned producer field")
      testException(vec(mixedBundle(Bool()), 4), vec(alignedFooBundle(Bool())), "dangling consumer field", "unassigned consumer field")
    }
    it("(0.i): Error if different root-relative flippedness on leaf fields between right-hand-side or left-hand-side") {
      testException(mixedBundle(Bool()), alignedBundle(Bool()), "inversely oriented fields")
      testException(alignedBundle(Bool()), mixedBundle(Bool()), "inversely oriented fields")
    }
    it("(0.j): Emit defaultable assignments on type with default, instead of erroring with missing fields") {
      testDistinctTypes(
        alignedBundle(Bool().withConnectableDefault(true.B)),
        alignedFooBundle(Bool()),
        Seq("io.out.foo <= io.in.foo", "io.out.bar <= UInt<1>(\"h1\")")
      )
      testDistinctTypes(
        alignedBundle(Bool()).withConnectableDefault{(_.bar -> true.B)},
        alignedFooBundle(Bool()),
        Seq("io.out.foo <= io.in.foo", "io.out.bar <= UInt<1>(\"h1\")")
      )
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
      testDistinctTypes(mixedBundle(Bool()), alignedFooBundle(Bool()), Seq("io.out.foo <= io.in.foo"))
      testDistinctTypes(alignedFooBundle(Bool()), mixedBundle(Bool()), Seq("io.out.foo <= io.in.foo"))

      // Missing foo
      testException(mixedBundle(Bool()), flippedBarBundle(Bool()), "unassigned consumer field")
      testDistinctTypes(flippedBarBundle(Bool()), mixedBundle(Bool()), Seq("skip"), Seq("<=")) // No connection should be emitted

      // Vec sizes don't match
      testDistinctTypes(vec(alignedFooBundle(Bool())), vec(alignedFooBundle(Bool()), 4), Seq(
        "io.out[0].foo <= io.in[0].foo",
        "io.out[1].foo <= io.in[1].foo",
        "io.out[2].foo <= io.in[2].foo"
      ))
      testException(vec(alignedFooBundle(Bool()), 4), vec(alignedFooBundle(Bool())), "unassigned consumer field")
      testDistinctTypes(vec(flippedBarBundle(Bool())), vec(flippedBarBundle(Bool()), 4), Seq("skip"))
      testDistinctTypes(vec(flippedBarBundle(Bool()), 4), vec(flippedBarBundle(Bool())), Seq("skip"))

      // Correct dangling/unassigned consumer/producer if vec has a bundle who has a flip field
      testDistinctTypes(vec(alignedFooBundle(Bool())), vec(mixedBundle(Bool()), 4), Seq(
        "io.out[0].foo <= io.in[0].foo",
        "io.out[1].foo <= io.in[1].foo",
        "io.out[2].foo <= io.in[2].foo"
      ))
      testException(vec(mixedBundle(Bool()), 4), vec(alignedFooBundle(Bool())), "unassigned consumer field")
    }
    it("(1.i): Use consumer orientation if different root-relative flippedness on leaf fields between right-hand-side or left-hand-side") {
      testDistinctTypes(mixedBundle(Bool()), alignedBundle(Bool()), Seq("io.out.foo <= io.in.foo"), Seq("io.in.bar <= io.out.bar"))
      testDistinctTypes(alignedBundle(Bool()), mixedBundle(Bool()), Seq(
        "io.out.foo <= io.in.foo",
        "io.out.bar <= io.in.bar"
      ))
    }
    it("(1.j): Emit defaultable assignments on type with default, instead of erroring with missing fields") {
      testDistinctTypes(
        alignedBundle(Bool().withConnectableDefault(true.B)),
        alignedFooBundle(Bool()),
        Seq("io.out.foo <= io.in.foo", "io.out.bar <= UInt<1>(\"h1\")")
      )
      testDistinctTypes(
        alignedBundle(Bool()).withConnectableDefault{ (_.bar -> true.B)},
        alignedFooBundle(Bool()),
        Seq("io.out.foo <= io.in.foo", "io.out.bar <= UInt<1>(\"h1\")")
      )
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
    it("(2.h): Error on unassigned subfield/subindex from either side, but do not throw exception for dangling fields") {
      // Missing flip bar
      testDistinctTypes(mixedBundle(Bool()), alignedFooBundle(Bool()), Seq("skip"), Seq("<="))
      testException(alignedFooBundle(Bool()), mixedBundle(Bool()), "unassigned producer field")

      // Missing foo
      testDistinctTypes(mixedBundle(Bool()), flippedBarBundle(Bool()), Seq("io.in.bar <= io.out.bar"))
      testDistinctTypes(flippedBarBundle(Bool()), mixedBundle(Bool()), Seq("io.in.bar <= io.out.bar")) // No connection should be emitted

      // Vec sizes don't match
      testDistinctTypes(vec(alignedFooBundle(Bool())), vec(alignedFooBundle(Bool()), 4), Seq("skip"), Seq("<="))
      testDistinctTypes(vec(alignedFooBundle(Bool()), 4), vec(alignedFooBundle(Bool())), Seq("skip"), Seq("<="))
      testException(vec(flippedBarBundle(Bool())), vec(flippedBarBundle(Bool()), 4), "unassigned producer field")
      testDistinctTypes(vec(flippedBarBundle(Bool()), 4), vec(flippedBarBundle(Bool())), Seq(
        "io.in[0].bar <= io.out[0].bar",
        "io.in[1].bar <= io.out[1].bar",
        "io.in[2].bar <= io.out[2].bar"
      ))

      // Correct dangling/unassigned consumer/producer if vec has a bundle who has a flip field
      testException(vec(flippedBarBundle(Bool())), vec(mixedBundle(Bool()), 4), "unassigned producer field")
      testDistinctTypes(vec(mixedBundle(Bool()), 4), vec(flippedBarBundle(Bool())), Seq(
        "io.in[0].bar <= io.out[0].bar",
        "io.in[1].bar <= io.out[1].bar",
        "io.in[2].bar <= io.out[2].bar"
      ))
    }
    it("(2.i): Use producer orientation if different root-relative flippedness on leaf fields between right-hand-side or left-hand-side") {
      testDistinctTypes(mixedBundle(Bool()), alignedBundle(Bool()), Seq("skip"), Seq("<="))
      testDistinctTypes(alignedBundle(Bool()), mixedBundle(Bool()), Seq("io.in.bar <= io.out.bar"), Seq("io.out.foo <= io.in.foo"))
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
      testException(mixedBundle(Bool()), alignedFooBundle(Bool()), "dangling consumer field")
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
      testException(vec(alignedFooBundle(Bool())), vec(mixedBundle(Bool()), 4), "unmatched producer field")
      testException(vec(mixedBundle(Bool()), 4), vec(alignedFooBundle(Bool())), "unassigned consumer field")
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
