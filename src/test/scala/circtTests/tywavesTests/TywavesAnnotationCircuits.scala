package circtTests.tywavesTests

import chisel3._
import chisel3.experimental.{Analog, IntrinsicModule}
import chisel3.experimental.hierarchy.{instantiable, Definition, Instance}
import chisel3.properties.Class
import chisel3.util.MixedVec

object TywavesAnnotationCircuits {

  trait BindingChoice { def apply[T <: Data](data: T): T }
  case object PortBinding extends BindingChoice {
    override def apply[T <: Data](data: T): T = IO((data))
    override def toString: String = "IO"
  }
  case object WireBinding extends BindingChoice {
    override def apply[T <: Data](data: T): T = Wire(data)
    override def toString: String = "Wire"
  }
  case object RegBinding extends BindingChoice {
    override def apply[T <: Data](data: T): T = Reg(data)
    override def toString: String = "Reg"
  }

  abstract class TywavesTestModule(bindingChoice: BindingChoice) extends RawModule {
    def body: Unit
    bindingChoice match {
      case RegBinding =>
        val clock = IO(Input(Clock()))
        val reset = IO(Input(Reset()))
        withClockAndReset(clock, reset) { body }
      case _ => body
    }
  }

  // Empty circuit
  class TopCircuit extends RawModule

  // Circuit with submodule
  class MyModule extends RawModule
  class TopCircuitSubModule extends RawModule { val mod = Module(new MyModule) }

  // Circuit with submodule and submodule with submodule
  class TopCircuitMultiModule extends RawModule {
    // 4 times MyModule in total
    val mod1 = Module(new MyModule)
    val mod2 = Module(new MyModule)
    val mods = Seq.fill(2)(Module(new MyModule))
  }

  // Circuit with submodule and submodule with submodule
  class TopCircuitWithParams extends RawModule {
    class MyModule(val width: Int) extends RawModule
    val mod1: MyModule = Module(new MyModule(8))
    val mod2: MyModule = Module(new MyModule(16))
    val mod3: MyModule = Module(new MyModule(32))
  }

  // Circuit with submodule and submodule with submodule
  class TopCircuitBlackBox extends RawModule {
    class MyBlackBox extends BlackBox(Map("PARAM1" -> "TRUE", "PARAM2" -> "DEFAULT")) { val io = IO(new Bundle {}) }
    val myBlackBox1:  MyBlackBox = Module(new MyBlackBox)
    val myBlackBox2:  MyBlackBox = Module(new MyBlackBox)
    val myBlackBoxes: Seq[MyBlackBox] = Seq.fill(2)(Module(new MyBlackBox))
  }

  // Circuit using intrinsics
  class ExampleIntrinsicModule(str: String) extends IntrinsicModule("OtherIntrinsic", Map("STRING" -> str)) {}
  class TopCircuitIntrinsic extends RawModule {
    val myIntrinsicModule1: ExampleIntrinsicModule = Module(new ExampleIntrinsicModule("Hello"))
    val myIntrinsicModule2: ExampleIntrinsicModule = Module(new ExampleIntrinsicModule("World"))
    val myIntrinsicModules: Seq[ExampleIntrinsicModule] = Seq.fill(2)(Module(new ExampleIntrinsicModule("Hello")))
  }

  // An abstract description of a CSR, represented as a Class.
  @instantiable
  class CSRDescription extends Class
  class CSRModule(csrDescDef: Definition[CSRDescription]) extends RawModule {
    val csrDescription = Instance(csrDescDef)
  }
  class TopCircuitClasses extends RawModule {
    val csrDescDef = Definition(new CSRDescription)
    val csrModule1: CSRModule = Module(new CSRModule(csrDescDef))
    val csrModule2: CSRModule = Module(new CSRModule(csrDescDef))
    val csrModules: Seq[CSRModule] = Seq.fill(2)(Module(new CSRModule(csrDescDef)))
  }

  // Test Clock, Reset and AsyncReset
  class TopCircuitClockReset extends RawModule {
    val clock: Clock = IO(Input(Clock()))
    // Types od reset https://www.chisel-lang.org/docs/explanations/reset
    val syncReset:  Bool = IO(Input(Bool()))
    val reset:      Reset = IO(Input(Reset()))
    val asyncReset: AsyncReset = IO(Input(AsyncReset()))
  }

  // Test implicit Clock and Reset
  class TopCircuitImplicitClockReset extends Module

  // TODO: add the other ground types
  class TopCircuitGroundTypes(bindingChoice: BindingChoice) extends TywavesTestModule(bindingChoice) {
    override def body: Unit = {
      val uint: UInt = bindingChoice(UInt(8.W))
      val sint: SInt = bindingChoice(SInt(8.W))
      val bool: Bool = bindingChoice(Bool())
      if (bindingChoice != RegBinding) { val analog: Analog = bindingChoice(Analog(1.W)) }
      //        val fixedPoint: FixedPoint TODO: does fixed point still exist?
      //        val interval: Interval = bindingChoice(Interval())) TODO: does interval still exist?
      val bits: UInt = bindingChoice(Bits(8.W))
    }
  }

  // Test Bundles
  class MyEmptyBundle extends Bundle
  class MyBundle extends Bundle {
    val a: UInt = UInt(8.W)
    val b: SInt = SInt(8.W)
    val c: Bool = Bool()
  }
  class TopCircuitBundles(bindingChoice: BindingChoice) extends TywavesTestModule(bindingChoice) {
    override def body: Unit = {
      val a: Bundle = bindingChoice(new Bundle {})
      val b: MyEmptyBundle = bindingChoice(new MyEmptyBundle)
      val c: MyBundle = bindingChoice(new MyBundle)
    }
  }

  class MyNestedBundle extends Bundle {
    val a: Bool = Bool()
    val b: MyBundle = new MyBundle
    val c: MyBundle = Flipped(new MyBundle)
  }
  class TopCircuitBundlesNested(bindingChoice: BindingChoice) extends TywavesTestModule(bindingChoice) {
    override def body: Unit = { val a: MyNestedBundle = bindingChoice(new MyNestedBundle) }
  }

  // Test Vecs
  class TopCircuitVecs(bindingChoice: BindingChoice) extends TywavesTestModule(bindingChoice) {
    override def body: Unit = {
      val a: Vec[SInt] = bindingChoice(Vec(5, SInt(23.W)))
      val b: Vec[Vec[SInt]] = bindingChoice(Vec(5, Vec(3, SInt(23.W))))
      val c = bindingChoice(Vec(5, new Bundle { val x: UInt = UInt(8.W) }))
      // TODO: check if this should have a better representation, now its type is represented as other Records
      val d = bindingChoice(MixedVec(UInt(3.W), SInt(10.W)))
    }
  }

  // Test Bundles with Vecs
  class TopCircuitBundleWithVec(bindingChoice: BindingChoice) extends TywavesTestModule(bindingChoice) {
    override def body: Unit = { val a = bindingChoice(new Bundle { val vec = Vec(5, UInt(8.W)) }) }
  }

  // Test signals in submodules
  class TopCircuitTypeInSubmodule(bindingChoice: BindingChoice) extends RawModule {
    val mod = Module(new TopCircuitGroundTypes(bindingChoice))
  }
}
