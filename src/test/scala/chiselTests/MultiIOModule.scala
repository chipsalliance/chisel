// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.MultiIOModule
import chisel3.testers.BasicTester

class MultiIOPlusOne extends MultiIOModule {
  val in  = IO(Input(UInt(32.W)))
  val out = IO(Output(UInt(32.W)))

  out := in + 1.asUInt
}

class MultiIOTester extends BasicTester {
  val plusModule = Module(new MultiIOPlusOne)
  plusModule.in := 42.U
  assert(plusModule.out === 43.U)
  stop()
}

// Demonstrate multiple IOs with inheritance where the IO is assigned to internally
trait LiteralOutputTrait extends MultiIOModule {
  val myLiteralIO = IO(Output(UInt(32.W)))
  myLiteralIO := 2.U
}

// Demonstrate multiple IOs with inheritance where the IO is not assigned
// (and must be assigned by what extends this trait).
trait MultiIOTrait extends MultiIOModule {
  val myTraitIO = IO(Output(UInt(32.W)))
}

// Composition of the two above traits, example of IO composition directly using multiple top-level
// IOs rather than indirectly by constraining the type of the single .io field.
class ComposedMultiIOModule extends MultiIOModule 
    with LiteralOutputTrait with MultiIOTrait {
  val topModuleIO = IO(Input(UInt(32.W)))
  myTraitIO := topModuleIO
}

class ComposedMultiIOTester extends BasicTester {
  val composedModule = Module(new ComposedMultiIOModule)
  composedModule.topModuleIO := 42.U
  assert(composedModule.myTraitIO === 42.U)
  assert(composedModule.myLiteralIO === 2.U)
  stop()
}

class MultiIOSpec extends ChiselFlatSpec {
  "Multiple IOs in MultiIOModule" should "work" in {
    assertTesterPasses({ new MultiIOTester })
  }
  "Composed MultiIO Modules" should "work" in {
    assertTesterPasses({ new ComposedMultiIOTester })
  }
}
