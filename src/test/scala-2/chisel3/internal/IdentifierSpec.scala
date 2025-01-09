// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import chisel3.{assert => _, _}
import chisel3.experimental.BaseModule
import chisel3.naming.{fixTraitIdentifier, HasCustomIdentifier, IdentifierProposer}
import chiselTests.ChiselFunSpec

class ConcreteClass(i: Int) extends Module
trait NormalTrait extends Module
abstract class AbstractClass extends Module
abstract class AbstArgsClass(i: Int) extends Module
abstract class NestedAbtrClz(b: Int) extends AbstArgsClass(b)
class NestedCrctClz(b: Int) extends ConcreteClass(b)
class NestedTratClz(b: Int) extends NormalTrait
@fixTraitIdentifier
trait FixedNormalTrait extends Module

class FooModule(i: Int) extends Module with HasCustomIdentifier {
  protected val customDefinitionIdentifierProposal = s"A_Different_Name_for_FooModule"
}

class Foo(i: Int) extends HasCustomIdentifier {
  protected val customDefinitionIdentifierProposal = s"A_Different_Name_for_Foo"
}
class BuiltGivenFoo(f: Foo) extends Module
class BuiltGivenInstance[A <: Module](i: A) extends Module

class HasByNameArg(g: => Module) extends Module {
  val x = Module(g)
}

class IdentifierSpec extends ChiselFunSpec {
  it("(1): definitionIdentifier works on classes, abstract classes, but not traits") {
    class TopA extends Module {
      assert(Module(new ConcreteClass(0)).definitionIdentifier == "ConcreteClass_0")
      assert(Module(new ConcreteClass(0)).definitionIdentifier == "ConcreteClass_0$1")
      assert(Module(new ConcreteClass(1) {}).definitionIdentifier == "ConcreteClass_1")
      assert(Module(new NormalTrait {}).definitionIdentifier == "_1_Anon") // Traits don't work, Scala compiler bug?!?
      assert(Module(new AbstractClass {}).definitionIdentifier == "AbstractClass")
      assert(Module(new AbstArgsClass(2) {}).definitionIdentifier == "AbstArgsClass_2")
      assert(Module(new NestedAbtrClz(3) {}).definitionIdentifier == "NestedAbtrClz_3")
      assert(Module(new NestedCrctClz(4)).definitionIdentifier == "NestedCrctClz_4")
      assert(Module(new NestedCrctClz(5) {}).definitionIdentifier == "NestedCrctClz_5")
      assert(Module(new NestedTratClz(6)).definitionIdentifier == "NestedTratClz_6")
      assert(Module(new NestedTratClz(7) {}).definitionIdentifier == "NestedTratClz_7")
      assert(Module(new BuiltGivenFoo(new Foo(1))).definitionIdentifier == "BuiltGivenFoo_A_Different_Name_for_Foo")
      val cc = Module(new FooModule(1))
      assert(
        Module(new BuiltGivenInstance(cc)).definitionIdentifier == "BuiltGivenInstance_A_Different_Name_for_FooModule"
      )
    }
    circt.stage.ChiselStage.emitCHIRRTL(new TopA, Array("--full-stacktrace"))
  }
  it("(2): using @fixTraitIdentifier works on to fix traits") {
    class TopA extends Module {
      val a = Module(new FixedNormalTrait {})
      assert(a.definitionIdentifier == "FixedNormalTrait")
    }
    circt.stage.ChiselStage.emitCHIRRTL(new TopA, Array("--full-stacktrace"))
  }
  it("(3): By-name arguments aren't called") {
    class TopA extends Module {
      val a = Module(new HasByNameArg(new FixedNormalTrait {}))
      assert(a.definitionIdentifier == "HasByNameArg")
    }
    circt.stage.ChiselStage.emitCHIRRTL(new TopA, Array("--full-stacktrace"))
  }
}
