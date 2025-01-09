// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental

import chisel3._
import chisel3.experimental.BaseModule
import chisel3.experimental.dataview._
import chisel3.experimental.conversions._
import chisel3.experimental.{annotate, ChiselAnnotation}
import chiselTests.ChiselFlatSpec

object DataViewTargetSpec {
  import firrtl.annotations._
  private case class DummyAnno(target: ReferenceTarget, id: Int) extends SingleTargetAnnotation[ReferenceTarget] {
    override def duplicate(n: ReferenceTarget) = this.copy(target = n)
  }
  private def mark(d: Data, id: Int) = annotate(new ChiselAnnotation {
    override def toFirrtl: Annotation = DummyAnno(d.toTarget, id)
  })
  private def markAbs(d: Data, id: Int) = annotate(new ChiselAnnotation {
    override def toFirrtl: Annotation = DummyAnno(d.toAbsoluteTarget, id)
  })
  private def markRel(d: Data, root: Option[BaseModule], id: Int) = annotate(new ChiselAnnotation {
    override def toFirrtl: Annotation = DummyAnno(d.toRelativeTarget(root), id)
  })
}

class DataViewTargetSpec extends ChiselFlatSpec {
  import DataViewTargetSpec._
  private val checks: Seq[Data => String] = Seq(
    _.toTarget.toString,
    _.toAbsoluteTarget.toString,
    _.instanceName,
    _.pathName,
    _.parentPathName,
    _.parentModName
  )

  // Check helpers
  private def checkAll(impl: Data, refs: String*): Unit = {
    refs.size should be(checks.size)
    for ((check, value) <- checks.zip(refs)) {
      check(impl) should be(value)
    }
  }
  private def checkSameAs(impl: Data, refs: Data*): Unit =
    for (ref <- refs) {
      checkAll(impl, checks.map(_(ref)): _*)
    }

  behavior.of("DataView Naming")

  it should "support views of Elements" in {
    class MyChild extends Module {
      val out = IO(Output(UInt(8.W)))
      val insideView = out.viewAs[UInt]
      out := 0.U
    }
    class MyParent extends Module {
      val out = IO(Output(UInt(8.W)))
      val inst = Module(new MyChild)
      out := inst.out
    }
    val m = elaborateAndGetModule(new MyParent)
    val outsideView = m.inst.out.viewAs[UInt]
    checkSameAs(m.inst.out, m.inst.insideView, outsideView)
  }

  it should "support 1:1 mappings of Aggregates and their children" in {
    class MyBundle extends Bundle {
      val foo = UInt(8.W)
      val bars = Vec(2, UInt(8.W))
    }
    implicit val dv =
      DataView[MyBundle, Vec[UInt]](_ => Vec(3, UInt(8.W)), _.foo -> _(0), _.bars(0) -> _(1), _.bars(1) -> _(2))
    class MyChild extends Module {
      val out = IO(Output(new MyBundle))
      val outView = out.viewAs[Vec[UInt]] // Note different type
      val outFooView = out.foo.viewAs[UInt]
      val outBarsView = out.bars.viewAs[Vec[UInt]]
      val outBars0View = out.bars(0).viewAs[UInt]
      out := 0.U.asTypeOf(new MyBundle)
    }
    class MyParent extends Module {
      val out = IO(Output(new MyBundle))
      val inst = Module(new MyChild)
      out := inst.out
    }
    val m = elaborateAndGetModule(new MyParent)
    val outView = m.inst.out.viewAs[Vec[UInt]] // Note different type
    val outFooView = m.inst.out.foo.viewAs[UInt]
    val outBarsView = m.inst.out.bars.viewAs[Vec[UInt]]
    val outBars0View = m.inst.out.bars(0).viewAs[UInt]

    checkSameAs(m.inst.out, m.inst.outView, outView)
    checkSameAs(m.inst.out.foo, m.inst.outFooView, m.inst.outView(0), outFooView, outView(0))
    checkSameAs(m.inst.out.bars, m.inst.outBarsView, outBarsView)
    checkSameAs(
      m.inst.out.bars(0),
      m.inst.outBars0View,
      outBars0View,
      m.inst.outView(1),
      outView(1),
      m.inst.outBarsView(0),
      outBarsView(0)
    )
  }

  // Ideally this would work 1:1 but that requires changing the binding
  it should "support annotation renaming of Aggregate children of Aggregate views" in {
    class MyBundle extends Bundle {
      val foo = Vec(2, UInt(8.W))
    }
    class MyChild extends Module {
      val out = IO(Output(new MyBundle))
      val outView = out.viewAs[MyBundle]
      mark(out.foo, 0)
      mark(outView.foo, 1)
      markAbs(out.foo, 2)
      markAbs(outView, 3)
      out := 0.U.asTypeOf(new MyBundle)
    }
    class MyParent extends Module {
      val out = IO(Output(new MyBundle))
      val inst = Module(new MyChild)
      out := inst.out
    }
    val (_, annos) = getFirrtlAndAnnos(new MyParent)
    val pairs = annos.collect { case DummyAnno(t, idx) => (idx, t.toString) }.sortBy(_._1)
    val expected = Seq(
      0 -> "~MyParent|MyChild>out.foo",
      1 -> "~MyParent|MyChild>out.foo",
      2 -> "~MyParent|MyParent/inst:MyChild>out.foo",
      3 -> "~MyParent|MyParent/inst:MyChild>out"
    )
    pairs should equal(expected)
  }

  it should "support annotating views that cannot be mapped to a single ReferenceTarget" in {
    class MyBundle extends Bundle {
      val a, b = Input(UInt(8.W))
      val c, d = Output(UInt(8.W))
    }
    // Note that each use of a Tuple as Data causes an implicit conversion creating a View
    class MyChild extends Module {
      val io = IO(new MyBundle)
      (io.c, io.d) := (io.a, io.b)
      // The type annotations create the views via the implicit conversion
      val view1: Data = (io.a, io.b)
      val view2: Data = (io.c, io.d)
      mark(view1, 0)
      mark(view2, 1)
      markAbs(view1, 2)
      markAbs(view2, 3)
      mark((io.b, io.d), 4) // Mix it up for fun
    }
    class MyParent extends Module {
      val io = IO(new MyBundle)
      val inst = Module(new MyChild)
      io <> inst.io
    }
    val (_, annos) = getFirrtlAndAnnos(new MyParent)
    val pairs = annos.collect { case DummyAnno(t, idx) => (idx, t.toString) }.sorted
    val expected = Seq(
      0 -> "~MyParent|MyChild>io.a",
      0 -> "~MyParent|MyChild>io.b",
      1 -> "~MyParent|MyChild>io.c",
      1 -> "~MyParent|MyChild>io.d",
      2 -> "~MyParent|MyParent/inst:MyChild>io.a",
      2 -> "~MyParent|MyParent/inst:MyChild>io.b",
      3 -> "~MyParent|MyParent/inst:MyChild>io.c",
      3 -> "~MyParent|MyParent/inst:MyChild>io.d",
      4 -> "~MyParent|MyChild>io.b",
      4 -> "~MyParent|MyChild>io.d"
    )
    pairs should equal(expected)
  }

  it should "support views with toRelativeTarget" in {
    class MyBundle extends Bundle {
      val foo = UInt(8.W)
      val bars = Vec(2, UInt(8.W))
    }
    implicit val dv =
      DataView[MyBundle, Vec[UInt]](_ => Vec(3, UInt(8.W)), _.foo -> _(0), _.bars(0) -> _(1), _.bars(1) -> _(2))
    class MyChild extends Module {
      val out = IO(Output(new MyBundle))
      val outView = out.viewAs[Vec[UInt]]
      out := 0.U.asTypeOf(new MyBundle)
    }
    class MyParent extends Module {
      val inst = Module(new MyChild)
      atModuleBodyEnd {
        markRel(inst.outView, Some(this), 0)
        markRel(inst.outView, Some(inst), 1)
        markRel(inst.outView, None, 2)
      }
    }
    val (_, annos) = getFirrtlAndAnnos(new MyParent)
    val pairs = annos.collect { case DummyAnno(t, idx) => (idx, t.toString) }.sorted
    val expected = Seq(
      0 -> "~MyParent|MyParent/inst:MyChild>out",
      1 -> "~MyParent|MyChild>out",
      2 -> "~MyParent|MyParent/inst:MyChild>out"
    )
    pairs should equal(expected)
  }

  // TODO check these properties when using @instance API (especially preservation of totality)
}
