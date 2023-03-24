// SPDX-License-Identifier: Apache-2.0

/* TODO(azidar):
 - (future PR) instance identifiers
 - test non-case class arguments
 - investigate definitionIdentifiers for more cornercases
 */

package chiselTests
package naming

import chisel3.{assert => _, _}
import chisel3.experimental.BaseModule
import chisel3.naming.{fixTraitIdentifier, IdentifierProposer}

class ConcreteClass(i: Int) extends Module
trait NormalTrait extends Module
abstract class AbstractClass extends Module
abstract class AbstArgsClass(i: Int) extends Module
abstract class NestedAbtrClz(b: Int) extends AbstArgsClass(b)
class NestedCrctClz(b: Int) extends ConcreteClass(b)
class NestedTratClz(b: Int) extends NormalTrait
@fixTraitIdentifier
trait FixedNormalTrait extends Module

class IdentifierSpec extends ChiselFunSpec with Utils {
  it("(1): definitionIdentifier works on classes, abstract classes, but not traits") {
    class TopA extends Module {
      assert(Module(new ConcreteClass(0)).definitionIdentifier == "ConcreteClass_0")
      assert(Module(new ConcreteClass(0) {}).definitionIdentifier == "ConcreteClass_0_1")
      assert(Module(new NormalTrait {}).definitionIdentifier == "_1_Anon") // Traits don't work, Scala compiler bug?!?
      assert(Module(new AbstractClass {}).definitionIdentifier == "AbstractClass")
      assert(Module(new AbstArgsClass(1) {}).definitionIdentifier == "AbstArgsClass_1")
      assert(Module(new NestedAbtrClz(1) {}).definitionIdentifier == "NestedAbtrClz_1")
      assert(Module(new NestedCrctClz(1)).definitionIdentifier == "NestedCrctClz_1")
      assert(Module(new NestedCrctClz(1) {}).definitionIdentifier == "NestedCrctClz_1_1")
      assert(Module(new NestedTratClz(1)).definitionIdentifier == "NestedTratClz_1")
      assert(Module(new NestedTratClz(1) {}).definitionIdentifier == "NestedTratClz_1_1")
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
}
