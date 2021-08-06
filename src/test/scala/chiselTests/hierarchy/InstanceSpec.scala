package chiselTests
package hierarchy

import chisel3._
import chisel3.testers.BasicTester
import chisel3.experimental.annotate
import chisel3.experimental.BaseModule
import chisel3.internal.{instantiable, public}
import _root_.firrtl.annotations._
import chisel3.stage.DesignAnnotation
import chisel3.stage.ChiselGeneratorAnnotation
import javax.print.attribute.HashPrintRequestAttributeSet


class InstanceSpec extends ChiselFunSpec with Utils {
  import Annotations._
  import Examples._
  describe("0: Names of instances") {
    it("0.0: name of an instance should be correct") {
      class Top extends MultiIOModule {
        val definition = Definition(new AddOne)
        val i0 = Instance(definition)
      }
      check(new Top(), "i0", "AddOne")
    }
  }
  describe("1: Annotations on instances in same chisel compilation") {
    it("1.0: should work on a single instance, annotating the instance") {
      class Top extends MultiIOModule {
        val definition: Definition[AddOne] = Definition(new AddOne)
        val i0: Instance[AddOne] = Instance(definition)
        mark(i0, "i0")
      }
      check(new Top(), "~Top|Top/i0:AddOne".it, "i0")
    }
    it("1.1: should work on a single instance, annotating an inner wire") {
      class Top extends MultiIOModule {
        val definition: Definition[AddOne] = Definition(new AddOne)
        val i0: Instance[AddOne] = Instance(definition)
        mark(i0.innerWire, "i0.innerWire")
      }
      check(new Top(), "~Top|Top/i0:AddOne>innerWire".rt, "i0.innerWire")
    }
    it("1.2: should work on a two nested instances, annotating the instance") {
      class Top extends MultiIOModule {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        val i0: Instance[AddTwo] = Instance(definition)
        mark(i0.i0, "i0.i0")
      }
      check(new Top(), "~Top|Top/i0:AddTwo/i0:AddOne".it, "i0.i0")
    }
    it("1.3: should work on a two nested instances, annotating the inner wire") {
      class Top extends MultiIOModule {
        val definition: Definition[AddTwo] = Definition(new AddTwo)
        val i0: Instance[AddTwo] = Instance(definition)
        mark(i0.i0.innerWire, "i0.i0.innerWire")
      }
      check(new Top(), "~Top|Top/i0:AddTwo/i0:AddOne>innerWire".rt, "i0.i0.innerWire")
    }
    it("1.4: should work on a nested module in an instance, annotating the module") {
      class Top extends MultiIOModule {
        val definition: Definition[AddTwoMixedModules] = Definition(new AddTwoMixedModules)
        val i0: Instance[AddTwoMixedModules] = Instance(definition)
        mark(i0.i1, "i0.i1")
      }
      check(new Top(), "~Top|Top/i0:AddTwoMixedModules/i1:AddOne_2".it, "i0.i1")
    }
    it("1.5: should work on an instantiable container, annotating a wire") {
      class Top extends MultiIOModule {
        val definition: Definition[AddOneWithInstantiableWire] = Definition(new AddOneWithInstantiableWire)
        val i0: Instance[AddOneWithInstantiableWire] = Instance(definition)
        mark(i0.wireContainer.innerWire, "i0.innerWire")
      }
      check(new Top(), "~Top|Top/i0:AddOneWithInstantiableWire>innerWire".rt, "i0.innerWire")
    }
    it("1.6: should work on an instantiable container, annotating a module") {
      class Top extends MultiIOModule {
        val definition = Definition(new AddOneWithInstantiableModule)
        val i0 = Instance(definition)
        mark(i0.moduleContainer.i0, "i0.i0")
      }
      check(new Top(), "~Top|Top/i0:AddOneWithInstantiableModule/i0:AddOne".it, "i0.i0")
    }
    it("1.7: should work on an instantiable container, annotating an instance") {
      class Top extends MultiIOModule {
        val definition = Definition(new AddOneWithInstantiableInstance)
        val i0 = Instance(definition)
        mark(i0.instanceContainer.i0, "i0.i0")
      }
      check(new Top(), "~Top|Top/i0:AddOneWithInstantiableInstance/i0:AddOne".it, "i0.i0")
    }
    it("1.8: should work on an instantiable container, annotating an instantiable container's module") {
      class Top extends MultiIOModule {
        val definition = Definition(new AddOneWithInstantiableInstantiable)
        val i0 = Instance(definition)
        mark(i0.containerContainer.container.i0, "i0.i0")
      }
      check(new Top(), "~Top|Top/i0:AddOneWithInstantiableInstantiable/i0:AddOne".it, "i0.i0")
    }
    it("1.9: should work on public member which references public member of another instance") {
      class Top extends MultiIOModule {
        val definition = Definition(new AddOneWithInstantiableInstantiable)
        val i0 = Instance(definition)
        mark(i0.containerContainer.container.i0, "i0.i0")
      }
      check(new Top(), "~Top|Top/i0:AddOneWithInstantiableInstantiable/i0:AddOne".it, "i0.i0")
    }
  }
  describe("2: Annotations on designs not in the same chisel compilation") {
    it("2.0: should work on an innerWire, marked in a different compilation") {
      val first = elaborate(new AddTwo)
      class Top(x: AddTwo) extends MultiIOModule {
        val parent = Instance(Definition(new ViewerParent(x, false, true)))
      }
      check(new Top(first), "~AddTwo|AddTwo/i0:AddOne>innerWire".rt, "first")
    }
    it("2.1: should work on an innerWire, marked in a different compilation, in instanced instantiable") {
      val first = elaborate(new AddTwo)
      class Top(x: AddTwo) extends MultiIOModule {
        val parent = Instance(Definition(new ViewerParent(x, true, false)))
      }
      check(new Top(first), "~AddTwo|AddTwo/i0:AddOne>innerWire".rt, "second")
    }
    it("2.2: should work on an innerWire, marked in a different compilation, in instanced module") {
      val first = elaborate(new AddTwo)
      class Top(x: AddTwo) extends MultiIOModule {
        val parent = Instance(Definition(new ViewerParent(x, false, false)))
        mark(parent.viewer.x.i0.innerWire, "third")
      }
      check(new Top(first), "~AddTwo|AddTwo/i0:AddOne>innerWire".rt, "third")
    }
  }
  describe("3: @public") {
    it("3.0: should work on multi-vals") {
      class Top() extends MultiIOModule {
        val mv = Instance(Definition(new MultiVal()))
        mark(mv.x, "mv.x")
      }
      check(new Top(), "~Top|Top/mv:MultiVal>x".rt, "mv.x")
    }
    it("3.1: should work on lazy vals") {
      class Top() extends MultiIOModule {
        val lv = Instance(Definition(new LazyVal()))
        mark(lv.x, lv.y)
      }
      check(new Top(), "~Top|Top/lv:LazyVal>x".rt, "Hi")
    }
    it("3.2: should work on islookupables") {
      class Top() extends MultiIOModule {
        val p = Parameters("hi", 0)
        val up = Instance(Definition(new UsesParameters(p)))
        mark(up.x, up.y.string + up.y.int)
      }
      check(new Top(), "~Top|Top/up:UsesParameters>x".rt, "hi0")
    }
    it("3.3: should work on lists") {
      class Top() extends MultiIOModule {
        val i = Instance(Definition(new HasList()))
        mark(i.x(1), i.y(1).toString)
      }
      check(new Top(), "~Top|Top/i:HasList>x_1".rt, "2")
    }
    it("3.4: should work on options") {
      class Top() extends MultiIOModule {
        val i = Instance(Definition(new HasOption()))
        i.x.map(x => mark(x, "x"))
      }
      check(new Top(), "~Top|Top/i:HasOption>x".rt, "x")
    }
    it("3.5: should work on vecs") {
      class Top() extends MultiIOModule {
        val i = Instance(Definition(new HasVec()))
        mark(i.x, "blah")
      }
      check(new Top(), "~Top|Top/i:HasVec>x".rt, "blah")
    }
    it("3.6: should work on statically indexed vectors external to module") {
      class Top() extends MultiIOModule {
        val i = Instance(Definition(new HasVec()))
        mark(i.x(1), "blah")
      }
      check(new Top(), "~Top|Top/i:HasVec>x[1]".rt, "blah")
    }
    it("3.7: should work on statically indexed vectors internal to module") {
      class Top() extends MultiIOModule {
        val i = Instance(Definition(new HasIndexedVec()))
        mark(i.y, "blah")
      }
      check(new Top(), "~Top|Top/i:HasIndexedVec>x[1]".rt, "blah")
    }
    ignore("3.8: should work on vals in constructor arguments") {
      class Top() extends MultiIOModule {
        val i = Instance(Definition(new HasPublicConstructorArgs(10)))
        //mark(i.x, i.int.toString)
      }
      check(new Top(), "~Top|Top/i:HasPublicConstructorArgs>x".rt, "10")
    }
  }
  describe("4: Conversions") {
    it("4.0: should work on modules") {
      class Top() extends MultiIOModule {
        val i = Module(new AddOne())
        f(i)
      }
      def f(i: Instance[AddOne]): Unit = mark(i.innerWire, "blah")
      check(new Top(), "~Top|AddOne>innerWire".rt, "blah")
    }
    it("4.1: should work on isinstantiables") {
      class Top() extends MultiIOModule {
        val i = Module(new AddTwo())
        val v = new Viewer(i, false)
        mark(f(v), "blah")
      }
      def f(i: Instance[Viewer]): Data = i.x.i0.innerWire
      check(new Top(), "~Top|AddTwo/i0:AddOne>innerWire".rt, "blah")
    }
    it("4.2: should work on seqs of modules") {
      class Top() extends MultiIOModule {
        val is = Seq(Module(new AddTwo()), Module(new AddTwo()))
        mark(f(is), "blah")
      }
      def f(i: Seq[Instance[AddTwo]]): Data = i.head.i0.innerWire
      check(new Top(), "~Top|AddTwo/i0:AddOne>innerWire".rt, "blah")
    }
    it("4.3: should work on seqs of isInstantiables") {
      class Top() extends MultiIOModule {
        val i = Module(new AddTwo())
        val vs = Seq(new Viewer(i, false), new Viewer(i, false))
        mark(f(vs), "blah")
      }
      def f(i: Seq[Instance[Viewer]]): Data = i.head.x.i0.innerWire
      check(new Top(), "~Top|AddTwo/i0:AddOne>innerWire".rt, "blah")
    }
  }
}