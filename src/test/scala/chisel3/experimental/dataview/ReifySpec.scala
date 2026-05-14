// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.dataview

import circt.stage.ChiselStage
import chisel3._
import chisel3.probe.Probe
import chisel3.properties.Property
import chisel3.experimental.Analog
import chisel3.experimental.dataview._

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers._
import chisel3.reflect.DataMirror
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}

object ReifySpec {

  // Helpers to unpack the tuple returned by reify.
  def _reify(elt:              Element): Element = reify(elt)._1
  def _reifyIdentityView(data: Data):    Option[Data] = reifyIdentityView(data).map(_._1)

  // Views can be single target of the same type, but still not identity.
  type ReversedVec[T <: Data] = Vec[T]
  implicit def reversedVecView[T <: Data]: DataView[Vec[T], ReversedVec[T]] =
    DataView.mapping[Vec[T], ReversedVec[T]](v => v.cloneType, { case (a, b) => a.reverse.zip(b) })

  class AllElementsBundle extends Bundle {
    val u = UInt(8.W)
    val s = SInt(8.W)
    val b = Bool()
    val r = Reset()
    val d = AsyncReset()
    val c = Clock()
    val a = Analog(8.W)
    val p = Probe(UInt(8.W))
    val prop = Property[String]()
  }
  implicit val allElementsView: DataView[Seq[Data], AllElementsBundle] =
    DataView.mapping[Seq[Data], AllElementsBundle](
      _ => new AllElementsBundle,
      { case (a, b) => a.zip(b.getElements) }
    )

  class SimpleBundle extends Bundle {
    val value = UInt(8.W)
  }
  class NestedBundle extends Bundle {
    val child = new SimpleBundle
  }
  class TargetBundle extends Bundle {
    val fizz = new NestedBundle
    val buzz = new SimpleBundle
    val vec = Vec(2, UInt(8.W))
  }
  class ViewChildBundle extends Bundle {
    val a = Vec(2, UInt(8.W))
    val b = Vec(2, UInt(8.W))
  }
  class ViewBundle extends Bundle {
    val foo = UInt(8.W) // used to ensure ViewBundle isn't 1-1
    val bar = new ViewChildBundle
  }
  // The key thing about this mapping is that ViewBundle.bar maps 1-1 (but not identity) to TargetBundle
  // and that ViewBundle.bar.b identity maps to TargetBundle.vec.
  implicit val myView: DataView[(UInt, TargetBundle), ViewBundle] =
    DataView.mapping[(UInt, TargetBundle), ViewBundle](
      _ => new ViewBundle,
      { case ((u, t), v) =>
        Seq(u -> v.foo, t.fizz.child.value -> v.bar.a(0), t.buzz.value -> v.bar.a(1), t.vec -> v.bar.b)
      }
    )
}

import ReifySpec._

class ReifySpec extends AnyFunSpec {

  describe("dataview.reify") {

    it("should reify single targets and identity for all non-view Elements") {
      ChiselStage.elaborate(new Module {
        val wires = (new AllElementsBundle).getElements.map(Wire(_))

        // .getElements returns Data so we have to match that these are Elements.
        wires.foreach { case elt: Element =>
          _reify(elt) should be(elt)
          _reifyIdentityView(elt) should be(Some(elt))
          reifySingleTarget(elt) should be(Some(elt))
        }
      })
    }

    it("should reify single targets and identity for all non-view Elements even as children of an Aggregate") {
      ChiselStage.elaborate(new Module {
        val bundle = IO(new AllElementsBundle)

        // .getElements returns Data so we have to match that these are Elements.
        bundle.getElements.foreach { case elt: Element =>
          _reify(elt) should be(elt)
          _reifyIdentityView(elt) should be(Some(elt))
          reifySingleTarget(elt) should be(Some(elt))
        }
      })
    }

    it("should reify single targets and identity for all Elements in an Aggregate identity-view") {
      ChiselStage.elaborate(new Module {
        val bundle = IO(new AllElementsBundle)
        val view = bundle.viewAs[AllElementsBundle]

        _reifyIdentityView(view) should be(Some(bundle))
        reifySingleTarget(view) should be(Some(bundle))

        // .getElements returns Data so we have to match that these are Elements.
        view.getElements.zip(bundle.getElements).foreach { case (v: Element, t: Element) =>
          _reify(v) should be(t)
          _reifyIdentityView(v) should be(Some(t))
          reifySingleTarget(v) should be(Some(t))
        }
      })
    }

    it("should reify single targets and identity for all Elements in an Aggregate non-identity and non-1-1 view") {
      ChiselStage.elaborate(new Module {
        val wires = (new AllElementsBundle).getElements.map(Wire(_))
        val view = wires.viewAs[AllElementsBundle]

        _reifyIdentityView(view) should be(None)
        reifySingleTarget(view) should be(None)

        // .getElements returns Data so we have to match that these are Elements.
        view.getElements.zip(wires).foreach { case (v: Element, t: Element) =>
          _reify(v) should be(t)
          _reifyIdentityView(v) should be(Some(t))
          reifySingleTarget(v) should be(Some(t))
        }
      })
    }

    it("should distinguish identity views from single-target views (even if the single target is the same type!") {
      ChiselStage.elaborate(new Module {
        val vec = IO(Vec(2, UInt(8.W)))
        val view = vec.viewAs[ReversedVec[UInt]]

        DataMirror.checkTypeEquivalence(vec, view) should be(true)

        _reifyIdentityView(view) should be(None)
        reifySingleTarget(view) should be(Some(vec))

        // But child elements remain identity views, yet note the reverse.
        _reify(view(0)) should be(vec(1))
        _reifyIdentityView(view(0)) should be(Some(vec(1)))
        reifySingleTarget(view(0)) should be(Some(vec(1)))
        _reify(view(1)) should be(vec(0))
        _reifyIdentityView(view(1)) should be(Some(vec(0)))
        reifySingleTarget(view(1)) should be(Some(vec(0)))
      })
    }

    it("should correctly reify single-target views despite complex hierarchy") {
      ChiselStage.elaborate(new Module {
        val in0 = IO(Input(UInt(8.W)))
        val in1 = IO(Input(new TargetBundle))
        val view = (in0, in1).viewAs[ViewBundle]

        reifySingleTarget(view) should be(None)
        _reifyIdentityView(view.bar) should be(None)
        reifySingleTarget(view.bar) should be(Some(in1))
        // Note how view.bar has a single target, but view.bar.a does not.
        _reifyIdentityView(view.bar.a) should be(None)
        reifySingleTarget(view.bar.a) should be(None)
        // Note that view.bar does not have an identity view, but view.bar.b does.
        _reifyIdentityView(view.bar.b) should be(Some(in1.vec))
        reifySingleTarget(view.bar.b) should be(Some(in1.vec))
      })
    }
  }

  // One would like to not duplicate these checks, but it's tricky because T and Instance[T] are different.
  // We could box T with .toInstance but we do want to check the 2 main user code paths.
  describe("dataview.reify + D/I") {

    it("should reify single targets and identity for all non-view Elements") {
      @instantiable
      class MyModule extends RawModule {
        @public val ios = (new AllElementsBundle).getElements.map(IO(_))
      }
      ChiselStage.elaborate(new RawModule {

        val child = Instantiate(new MyModule)

        // .getElements returns Data so we have to match that these are Elements.
        child.ios.foreach { case elt: Element =>
          _reify(elt) should be(elt)
          _reifyIdentityView(elt) should be(Some(elt))
          reifySingleTarget(elt) should be(Some(elt))
        }
      })
    }

    it("should reify single targets and identity for all non-view Elements even as children of an Aggregate") {
      @instantiable
      class MyModule extends RawModule {
        @public val bundle = IO(new AllElementsBundle)
      }
      ChiselStage.elaborate(new Module {
        val child = Instantiate(new MyModule)

        // .getElements returns Data so we have to match that these are Elements.
        child.bundle.getElements.foreach { case elt: Element =>
          _reify(elt) should be(elt)
          _reifyIdentityView(elt) should be(Some(elt))
          reifySingleTarget(elt) should be(Some(elt))
        }
      })
    }

    it("should reify single targets and identity for all Elements in an Aggregate identity-view") {
      @instantiable
      class MyModule extends RawModule {
        @public val bundle = IO(new AllElementsBundle)
        @public val view = bundle.viewAs[AllElementsBundle]
      }
      ChiselStage.elaborate(new Module {
        val child = Instantiate(new MyModule)

        _reifyIdentityView(child.view) should be(Some(child.bundle))
        reifySingleTarget(child.view) should be(Some(child.bundle))

        // .getElements returns Data so we have to match that these are Elements.
        child.view.getElements.zip(child.bundle.getElements).foreach { case (v: Element, t: Element) =>
          _reify(v) should be(t)
          _reifyIdentityView(v) should be(Some(t))
          reifySingleTarget(v) should be(Some(t))
        }
      })
    }

    it("should reify single targets and identity for all Elements in an Aggregate non-identity and non-1-1 view") {
      @instantiable
      class MyModule extends RawModule {
        @public val wires = (new AllElementsBundle).getElements.map(Wire(_))
        @public val view = wires.viewAs[AllElementsBundle]
      }
      ChiselStage.elaborate(new Module {
        val child = Instantiate(new MyModule)

        _reifyIdentityView(child.view) should be(None)
        reifySingleTarget(child.view) should be(None)

        // .getElements returns Data so we have to match that these are Elements.
        child.view.getElements.zip(child.wires).foreach { case (v: Element, t: Element) =>
          _reify(v) should be(t)
          _reifyIdentityView(v) should be(Some(t))
          reifySingleTarget(v) should be(Some(t))
        }
      })
    }

    it("should distinguish identity views from single-target views (even if the single target is the same type!") {
      @instantiable
      class MyModule extends RawModule {
        @public val vec = IO(Vec(2, UInt(8.W)))
        @public val view = vec.viewAs[ReversedVec[UInt]]
      }
      ChiselStage.elaborate(new Module {
        val child = Instantiate(new MyModule)
        val vec = child.vec
        val view = child.view

        DataMirror.checkTypeEquivalence(vec, view) should be(true)

        _reifyIdentityView(view) should be(None)
        reifySingleTarget(view) should be(Some(vec))

        // But child elements remain identity views, yet note the reverse.
        _reify(view(0)) should be(vec(1))
        _reifyIdentityView(view(0)) should be(Some(vec(1)))
        reifySingleTarget(view(0)) should be(Some(vec(1)))
        _reify(view(1)) should be(vec(0))
        _reifyIdentityView(view(1)) should be(Some(vec(0)))
        reifySingleTarget(view(1)) should be(Some(vec(0)))
      })
    }

    it("should correctly reify single-target views despite complex hierarchy") {
      @instantiable
      class MyModule extends RawModule {
        @public val in0 = IO(Input(UInt(8.W)))
        @public val in1 = IO(Input(new TargetBundle))
        @public val view = (in0, in1).viewAs[ViewBundle]
      }
      ChiselStage.elaborate(new Module {
        val child = Instantiate(new MyModule)
        val in0 = child.in0
        val in1 = child.in1
        val view = child.view

        reifySingleTarget(view) should be(None)
        _reifyIdentityView(view.bar) should be(None)
        reifySingleTarget(view.bar) should be(Some(in1))
        // Note how view.bar has a single target, but view.bar.a does not.
        _reifyIdentityView(view.bar.a) should be(None)
        reifySingleTarget(view.bar.a) should be(None)
        // Note that view.bar does not have an identity view, but view.bar.b does.
        _reifyIdentityView(view.bar.b) should be(Some(in1.vec))
        reifySingleTarget(view.bar.b) should be(Some(in1.vec))
      })
    }
  }

}
