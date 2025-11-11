// SPDX-License-Identifier: Apache-2.0

package chiselTests.aop

import chisel3._
import chisel3.aop.Select
import chisel3.aop.Select.{PredicatedConnect, When, WhenNot}
import chisel3.experimental.{ExtModule, IntrinsicModule}
import chisel3.stage.{ChiselGeneratorAnnotation, DesignAnnotation}
import chisel3.util.{Cat, MuxLookup}
import circt.stage.ChiselStage
import firrtl.AnnotationSeq
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.reflect.runtime.universe.TypeTag

class SelectTester(results: Seq[Int]) extends Module {
  val values = VecInit(results.map(_.U))
  val counter = RegInit(0.U(results.length.W))
  val added = counter + 1.U
  counter := added
  val overflow = counter >= values.length.U
  val nreset = reset.asBool === false.B
  val selected = values(counter)
  val zero = 0.U + 0.U
  var p: printf.Printf = null
  when(overflow) {
    counter := zero
    stop()
  }.otherwise {
    when(nreset) {
      assert(counter === values(counter))
      p = printf("values(%d) = %d\n", counter, selected)
    }

  }
}

class SelectSpec extends AnyFlatSpec with Matchers {

  "Test" should "pass if selecting correct registers" in {
    val dut = ChiselGeneratorAnnotation(() => {
      new SelectTester(Seq(0, 1, 2))
    }).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    Select.registers(dut) should be(Seq(dut.counter))
  }

  "Test" should "pass if selecting correct wires" in {
    val dut = ChiselGeneratorAnnotation(() => {
      new SelectTester(Seq(0, 1, 2))
    }).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    Select.wires(dut) should be(Seq(dut.values))

  }

  "Test" should "pass if selecting correct printfs" in {
    val dut = ChiselGeneratorAnnotation(() => {
      new SelectTester(Seq(0, 1, 2))
    }).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    Seq(Select.printfs(dut).last.toString) should be(
      Seq(
        Select
          .Printf(
            dut.p,
            Seq(
              When(Select.ops("eq")(dut).last.asInstanceOf[Bool]),
              When(dut.nreset),
              WhenNot(dut.overflow)
            ),
            dut.p.pable,
            dut.clock
          )
          .toString
      )
    )
  }

  "Test" should "pass if selecting correct connections" in {
    val dut = ChiselGeneratorAnnotation(() => {
      new SelectTester(Seq(0, 1, 2))
    }).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    Select.connectionsTo(dut)(dut.counter) should be(
      Seq(
        PredicatedConnect(Nil, dut.counter, dut.added, false),
        PredicatedConnect(Seq(When(dut.overflow)), dut.counter, dut.zero, false)
      )
    )

  }

  "Test" should "pass if selecting attach" in {
    import chisel3.experimental.{attach, Analog}
    class AttachTest extends RawModule {
      val a, b, c = IO(Analog(8.W))
      attach(a, b, c)
    }
    val dut = ChiselGeneratorAnnotation(() => new AttachTest)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[AttachTest]]
      .design
    Select.attachedTo(dut)(dut.a) should be(Set(dut.a, dut.b, dut.c))

  }

  "Test" should "pass if selecting ops by kind" in {
    val dut = ChiselGeneratorAnnotation(() => {
      new SelectTester(Seq(0, 1, 2))
    }).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    Select.ops("tail")(dut) should be(Seq(dut.added, dut.zero))
  }

  "Test" should "pass if selecting ops" in {
    val dut = ChiselGeneratorAnnotation(() => {
      new SelectTester(Seq(0, 1, 2))
    }).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    Select.ops(dut).collect { case ("tail", d) => d } should be(Seq(dut.added, dut.zero))
  }

  "Test" should "pass if selecting correct stops" in {
    val dut = ChiselGeneratorAnnotation(() => {
      new SelectTester(Seq(0, 1, 2))
    }).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    Select.stops(dut) should be(
      Seq(
        Select.Stop(
          Seq(
            When(Select.ops("eq")(dut)(1).asInstanceOf[Bool]),
            When(dut.overflow)
          ),
          0,
          dut.clock
        )
      )
    )
  }

  "Blackboxes" should "be supported in Select.instances" in {
    class BB extends ExtModule {}
    class Top extends RawModule {
      val bb = Module(new BB)
    }
    val top = ChiselGeneratorAnnotation(() => {
      new Top()
    }).elaborate(1).asInstanceOf[DesignAnnotation[Top]].design
    val bbs = Select.collectDeep(top) { case b: BB => b }
    assert(bbs.size == 1)
  }

  "collectDeep" should "should look in when regions" in {
    class BB extends ExtModule {}
    class Top extends RawModule {
      when(true.B) {
        val bb = Module(new BB)
      }
    }
    val top = ChiselGeneratorAnnotation(() => {
      new Top()
    }).elaborate(1).asInstanceOf[DesignAnnotation[Top]].design
    val bbs = Select.collectDeep(top) { case b: BB => b }
    assert(bbs.size == 1)
  }

  "collectDeep" should "should look in layer regions" in {
    object TestLayer extends layer.Layer(layer.LayerConfig.Extract())
    class BB extends ExtModule {}
    class Top extends RawModule {
      layer.block(TestLayer) {
        val bb = Module(new BB)
      }
    }
    val top = ChiselGeneratorAnnotation(() => {
      new Top()
    }).elaborate(1).asInstanceOf[DesignAnnotation[Top]].design
    val bbs = Select.collectDeep(top) { case b: BB => b }
    assert(bbs.size == 1)
  }

  "CloneModuleAsRecord" should "NOT show up in Select aspects" in {
    import chisel3.experimental.CloneModuleAsRecord
    class Child extends RawModule {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      out := in
    }
    class Top extends Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      val inst0 = Module(new Child)
      val inst1 = CloneModuleAsRecord(inst0)
      inst0.in := in
      inst1("in") := inst0.out
      out := inst1("out")
    }
    val top = ChiselGeneratorAnnotation(() => {
      new Top()
    }).elaborate.collectFirst { case DesignAnnotation(design: Top, _) => design }.get
    Select.collectDeep(top) { case x => x } should equal(Seq(top, top.inst0))
    Select.getDeep(top)(x => Seq(x)) should equal(Seq(top, top.inst0))
    Select.instances(top) should equal(Seq(top.inst0))
  }

  "Using Definition/Instance with Select APIs" should "throw an error" in {
    import chisel3.experimental.CloneModuleAsRecord
    import chisel3.experimental.hierarchy._
    @instantiable
    class Child extends RawModule {
      @public val in = IO(Input(UInt(8.W)))
      @public val out = IO(Output(UInt(8.W)))
      out := in
    }
    class Top extends Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      val definition = Definition(new Child)
      val inst0 = Instance(definition)
      val inst1 = Instance(definition)
      inst0.in := in
      inst1.in := inst0.out
      out := inst1.out
    }
    val top = ChiselGeneratorAnnotation(() => {
      new Top()
    }).elaborate.collectFirst { case DesignAnnotation(design: Top, _) => design }.get
    intercept[Exception] { Select.collectDeep(top) { case x => x } }
    intercept[Exception] { Select.getDeep(top)(x => Seq(x)) }
    intercept[Exception] { Select.instances(top) }
  }

  "Select currentInstancesIn and allCurrentInstancesIn" should "support module, extmodule, and D/I" in {
    import chisel3.experimental.hierarchy._

    class MyLeafExtModule extends ExtModule {}

    class MyLeafModule extends RawModule {}

    class MyIntermediateModule extends RawModule {
      Module(new MyLeafModule)
      Module(new MyLeafModule)
    }

    @instantiable
    class MyLeafInstance extends RawModule {}

    @instantiable
    class MyIntermediateInstance extends RawModule {
      val definition = Definition(new MyLeafInstance)
      Instance(definition)
      Instance(definition)
    }

    class MyWrapperModule extends RawModule {
      implicit val mg = new chisel3.internal.MacroGenerated {}

      val definition = Definition(new MyIntermediateInstance)

      Module(new MyIntermediateModule)

      Instance(definition)

      // Check instances thus far.
      val myInstances0 = Select.unsafe.currentInstancesIn(this).map(_._lookup { m => m.name })

      myInstances0 should be(Seq("MyIntermediateModule", "MyIntermediateInstance"))

      Module(new MyLeafExtModule)

      // Check that the new instance is also returned.
      val myInstances1 = Select.unsafe.currentInstancesIn(this).map(_._lookup { m => m.name })

      myInstances1 should be(Seq("MyIntermediateModule", "MyIntermediateInstance", "MyLeafExtModule"))
    }

    class MyTopModule extends RawModule {
      implicit val mg = new chisel3.internal.MacroGenerated {}

      Module(new MyWrapperModule)

      // Check the recursive version goes all the way down.
      val allInstances = Select.unsafe.allCurrentInstancesIn(this).map(_._lookup { m => m.name })

      allInstances should be(
        Seq(
          "MyWrapperModule",
          "MyIntermediateModule",
          "MyIntermediateInstance",
          "MyLeafExtModule",
          "MyLeafModule",
          "MyLeafModule_1",
          "MyLeafInstance",
          "MyLeafInstance"
        )
      )
    }

    ChiselStage.emitCHIRRTL(new MyTopModule)
  }

  "Looking up instances" should "not affect the result of currentInstancesIn" in {
    import chisel3.experimental.hierarchy._

    @instantiable
    class Grandchild extends Module

    @instantiable
    class Child extends Module {
      @public val g1 = Instantiate(new Grandchild)
    }

    class Parent extends Module {
      val c1 = Instantiate(new Child)

      Select.unsafe.currentInstancesIn(this) should be(Seq(c1))

      c1.g1 // This does more than you think :)

      Select.unsafe.currentInstancesIn(this) should be(Seq(c1))

      // We can't check the exact contents because what this returns is not the same object as c1.g1
      // TODO is it a problem that it's not the same object?
      Select.unsafe.allCurrentInstancesIn(this).size should be(2)

      Select.unsafe.currentInstancesIn(this) should be(Seq(c1))
    }

    ChiselStage.elaborate(new Parent)
  }

  "Select.unreliableDedupHash" should "work for simple modules" in {
    val gen = () => new SelectTester(Seq(0, 1, 2))
    val dut1 = ChiselGeneratorAnnotation(gen).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    val dut2 = ChiselGeneratorAnnotation(gen).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    println(circt.stage.ChiselStage.emitCHIRRTL(gen()))
    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 1: Basic module with registers
  "Select.unreliableDedupHash" should "produce identical hashes for modules with identical registers" in {
    class SimpleRegModule extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      val reg = RegNext(io.in)
      io.out := reg
    }

    val dut1 = ChiselGeneratorAnnotation(() => new SimpleRegModule)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[SimpleRegModule]]
      .design
    val dut2 = ChiselGeneratorAnnotation(() => new SimpleRegModule)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[SimpleRegModule]]
      .design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 2: Module with wires and arithmetic operations
  "Select.unreliableDedupHash" should "produce identical hashes for modules with identical wires and operations" in {
    class ArithmeticModule extends Module {
      val io = IO(new Bundle {
        val a = Input(UInt(8.W))
        val b = Input(UInt(8.W))
        val sum = Output(UInt(8.W))
        val diff = Output(UInt(8.W))
      })
      val wire1 = Wire(UInt(8.W))
      val wire2 = Wire(UInt(8.W))
      wire1 := io.a + io.b
      wire2 := io.a - io.b
      io.sum := wire1
      io.diff := wire2
    }

    val dut1 = ChiselGeneratorAnnotation(() => new ArithmeticModule)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ArithmeticModule]]
      .design
    val dut2 = ChiselGeneratorAnnotation(() => new ArithmeticModule)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ArithmeticModule]]
      .design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 3: Module with memory
  "Select.unreliableDedupHash" should "produce identical hashes for modules with identical memories" in {
    class MemoryModule extends Module {
      val io = IO(new Bundle {
        val addr = Input(UInt(4.W))
        val dataIn = Input(UInt(32.W))
        val dataOut = Output(UInt(32.W))
        val write = Input(Bool())
      })
      val mem = SyncReadMem(16, UInt(32.W))
      when(io.write) {
        mem.write(io.addr, io.dataIn)
      }
      io.dataOut := mem.read(io.addr)
    }

    val dut1 =
      ChiselGeneratorAnnotation(() => new MemoryModule).elaborate(1).asInstanceOf[DesignAnnotation[MemoryModule]].design
    val dut2 =
      ChiselGeneratorAnnotation(() => new MemoryModule).elaborate(1).asInstanceOf[DesignAnnotation[MemoryModule]].design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 4: Module with when/otherwise blocks
  "Select.unreliableDedupHash" should "produce identical hashes for modules with identical conditional logic" in {
    class ConditionalModule extends Module {
      val io = IO(new Bundle {
        val sel = Input(Bool())
        val a = Input(UInt(8.W))
        val b = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      when(io.sel) {
        io.out := io.a
      }.otherwise {
        io.out := io.b
      }
    }

    val dut1 = ChiselGeneratorAnnotation(() => new ConditionalModule)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ConditionalModule]]
      .design
    val dut2 = ChiselGeneratorAnnotation(() => new ConditionalModule)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ConditionalModule]]
      .design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 5: Module with multiple instances of the same submodule
  "Select.unreliableDedupHash" should "produce identical hashes for modules with multiple identical submodule instances" in {
    class SubModule extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      io.out := io.in + 1.U
    }

    class TopModule extends Module {
      val io = IO(new Bundle {
        val in1 = Input(UInt(8.W))
        val in2 = Input(UInt(8.W))
        val out1 = Output(UInt(8.W))
        val out2 = Output(UInt(8.W))
      })
      val sub1 = Module(new SubModule)
      val sub2 = Module(new SubModule)
      sub1.io.in := io.in1
      sub2.io.in := io.in2
      io.out1 := sub1.io.out
      io.out2 := sub2.io.out
    }

    val dut1 =
      ChiselGeneratorAnnotation(() => new TopModule).elaborate(1).asInstanceOf[DesignAnnotation[TopModule]].design
    val dut2 =
      ChiselGeneratorAnnotation(() => new TopModule).elaborate(1).asInstanceOf[DesignAnnotation[TopModule]].design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 6: Module with different data types and bundles
  "Select.unreliableDedupHash" should "produce identical hashes for modules with identical bundle structures" in {
    class CustomBundle extends Bundle {
      val data = UInt(16.W)
      val valid = Bool()
    }

    class BundleModule extends Module {
      val io = IO(new Bundle {
        val in = Input(new CustomBundle)
        val out = Output(new CustomBundle)
      })
      val reg = Reg(new CustomBundle)
      reg := io.in
      io.out := reg
    }

    val dut1 =
      ChiselGeneratorAnnotation(() => new BundleModule).elaborate(1).asInstanceOf[DesignAnnotation[BundleModule]].design
    val dut2 =
      ChiselGeneratorAnnotation(() => new BundleModule).elaborate(1).asInstanceOf[DesignAnnotation[BundleModule]].design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 7: Module with Vec operations
  "Select.unreliableDedupHash" should "produce identical hashes for modules with identical Vec operations" in {
    class VecModule extends Module {
      val io = IO(new Bundle {
        val sel = Input(UInt(2.W))
        val data = Input(Vec(4, UInt(8.W)))
        val out = Output(UInt(8.W))
      })
      val vecReg = Reg(Vec(4, UInt(8.W)))
      vecReg := io.data
      io.out := vecReg(io.sel)
    }

    val dut1 =
      ChiselGeneratorAnnotation(() => new VecModule).elaborate(1).asInstanceOf[DesignAnnotation[VecModule]].design
    val dut2 =
      ChiselGeneratorAnnotation(() => new VecModule).elaborate(1).asInstanceOf[DesignAnnotation[VecModule]].design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 8: Module with bit manipulation operations
  "Select.unreliableDedupHash" should "produce identical hashes for modules with identical bit operations" in {
    class BitOpsModule extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(16.W))
        val out = Output(UInt(16.W))
      })
      val shifted = io.in << 2
      val masked = shifted & 0xff.U
      val concatenated = Cat(masked(7, 4), masked(3, 0))
      io.out := concatenated
    }

    val dut1 =
      ChiselGeneratorAnnotation(() => new BitOpsModule).elaborate(1).asInstanceOf[DesignAnnotation[BitOpsModule]].design
    val dut2 =
      ChiselGeneratorAnnotation(() => new BitOpsModule).elaborate(1).asInstanceOf[DesignAnnotation[BitOpsModule]].design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 9: Module with RegInit and reset logic
  "Select.unreliableDedupHash" should "produce identical hashes for modules with identical reset logic" in {
    class ResetModule extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
        val enable = Input(Bool())
      })
      val counter = RegInit(0.U(8.W))
      when(io.enable) {
        counter := counter + 1.U
      }
      io.out := counter + io.in
    }

    val dut1 =
      ChiselGeneratorAnnotation(() => new ResetModule).elaborate(1).asInstanceOf[DesignAnnotation[ResetModule]].design
    val dut2 =
      ChiselGeneratorAnnotation(() => new ResetModule).elaborate(1).asInstanceOf[DesignAnnotation[ResetModule]].design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 10: Module with Mux operations
  "Select.unreliableDedupHash" should "produce identical hashes for modules with identical Mux operations" in {
    class MuxModule extends Module {
      val io = IO(new Bundle {
        val sel = Input(UInt(2.W))
        val in0 = Input(UInt(8.W))
        val in1 = Input(UInt(8.W))
        val in2 = Input(UInt(8.W))
        val in3 = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      io.out := MuxLookup(io.sel, 0.U)(
        Seq(
          0.U -> io.in0,
          1.U -> io.in1,
          2.U -> io.in2,
          3.U -> io.in3
        )
      )
    }

    val dut1 =
      ChiselGeneratorAnnotation(() => new MuxModule).elaborate(1).asInstanceOf[DesignAnnotation[MuxModule]].design
    val dut2 =
      ChiselGeneratorAnnotation(() => new MuxModule).elaborate(1).asInstanceOf[DesignAnnotation[MuxModule]].design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 11: Module with different but equivalent implementations should have different hashes
  "Select.unreliableDedupHash" should "produce different hashes for modules with different implementations" in {
    class ModuleA extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      io.out := io.in + 1.U
    }

    class ModuleB extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      io.out := io.in + 2.U // Different constant
    }

    val dutA = ChiselGeneratorAnnotation(() => new ModuleA).elaborate(1).asInstanceOf[DesignAnnotation[ModuleA]].design
    val dutB = ChiselGeneratorAnnotation(() => new ModuleB).elaborate(1).asInstanceOf[DesignAnnotation[ModuleB]].design

    Select.unreliableDedupHash(dutA.toDefinition) should not be (Select.unreliableDedupHash(dutB.toDefinition))
  }

  // Test 12: Module with attach operations (analog signals)
  "Select.unreliableDedupHash" should "produce identical hashes for modules with identical attach operations" in {
    import chisel3.experimental.{attach, Analog}

    class AnalogModule extends RawModule {
      val a = IO(Analog(8.W))
      val b = IO(Analog(8.W))
      val c = IO(Analog(8.W))
      attach(a, b, c)
    }

    val dut1 =
      ChiselGeneratorAnnotation(() => new AnalogModule).elaborate(1).asInstanceOf[DesignAnnotation[AnalogModule]].design
    val dut2 =
      ChiselGeneratorAnnotation(() => new AnalogModule).elaborate(1).asInstanceOf[DesignAnnotation[AnalogModule]].design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 13: Module with complex nested when blocks
  "Select.unreliableDedupHash" should "produce identical hashes for modules with identical nested when blocks" in {
    class NestedWhenModule extends Module {
      val io = IO(new Bundle {
        val sel1 = Input(Bool())
        val sel2 = Input(Bool())
        val a = Input(UInt(8.W))
        val b = Input(UInt(8.W))
        val c = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })

      when(io.sel1) {
        when(io.sel2) {
          io.out := io.a
        }.otherwise {
          io.out := io.b
        }
      }.otherwise {
        io.out := io.c
      }
    }

    val dut1 = ChiselGeneratorAnnotation(() => new NestedWhenModule)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[NestedWhenModule]]
      .design
    val dut2 = ChiselGeneratorAnnotation(() => new NestedWhenModule)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[NestedWhenModule]]
      .design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 14: Module with multiple different submodule types
  "Select.unreliableDedupHash" should "produce identical hashes for modules with multiple different submodule types" in {
    class AdderModule extends Module {
      val io = IO(new Bundle {
        val a = Input(UInt(8.W))
        val b = Input(UInt(8.W))
        val sum = Output(UInt(8.W))
      })
      io.sum := io.a + io.b
    }

    class MultiplierModule extends Module {
      val io = IO(new Bundle {
        val a = Input(UInt(8.W))
        val b = Input(UInt(8.W))
        val product = Output(UInt(16.W))
      })
      io.product := io.a * io.b
    }

    class CompositeModule extends Module {
      val io = IO(new Bundle {
        val a = Input(UInt(8.W))
        val b = Input(UInt(8.W))
        val c = Input(UInt(8.W))
        val sum = Output(UInt(8.W))
        val product = Output(UInt(16.W))
      })
      val adder = Module(new AdderModule)
      val multiplier = Module(new MultiplierModule)

      adder.io.a := io.a
      adder.io.b := io.b
      multiplier.io.a := io.b
      multiplier.io.b := io.c

      io.sum := adder.io.sum
      io.product := multiplier.io.product
    }

    val dut1 = ChiselGeneratorAnnotation(() => new CompositeModule)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[CompositeModule]]
      .design
    val dut2 = ChiselGeneratorAnnotation(() => new CompositeModule)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[CompositeModule]]
      .design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 15: Module with FirrtlMemory with different port names but same structure
  "Select.unreliableDedupHash" should "produce identical hashes for FirrtlMemory with different port names but same structure" in {
    import chisel3.util.SRAM

    class MemoryModuleA extends Module {
      val io = IO(new Bundle {
        val addr1 = Input(UInt(8.W))
        val addr2 = Input(UInt(8.W))
        val dataIn = Input(UInt(32.W))
        val dataOut = Output(UInt(32.W))
        val writeEn = Input(Bool())
        val rwEn = Input(Bool())
        val rwWrite = Input(Bool())
        val rwDataIn = Input(UInt(32.W))
        val rwDataOut = Output(UInt(32.W))
      })

      val mem = SRAM(
        size = 256,
        tpe = UInt(32.W),
        numReadPorts = 1,
        numWritePorts = 1,
        numReadwritePorts = 1
      )

      // Read port
      mem.readPorts(0).address := io.addr1
      mem.readPorts(0).enable := true.B
      io.dataOut := mem.readPorts(0).data

      // Write port
      mem.writePorts(0).address := io.addr1
      mem.writePorts(0).enable := io.writeEn
      mem.writePorts(0).data := io.dataIn

      // Read/write port
      mem.readwritePorts(0).address := io.addr2
      mem.readwritePorts(0).enable := io.rwEn
      mem.readwritePorts(0).isWrite := io.rwWrite
      mem.readwritePorts(0).writeData := io.rwDataIn
      io.rwDataOut := mem.readwritePorts(0).readData
    }

    class MemoryModuleB extends Module {
      val io = IO(new Bundle {
        val addr1 = Input(UInt(8.W))
        val addr2 = Input(UInt(8.W))
        val dataIn = Input(UInt(32.W))
        val dataOut = Output(UInt(32.W))
        val writeEn = Input(Bool())
        val rwEn = Input(Bool())
        val rwWrite = Input(Bool())
        val rwDataIn = Input(UInt(32.W))
        val rwDataOut = Output(UInt(32.W))
      })

      val mem = SRAM(
        size = 256,
        tpe = UInt(32.W),
        numReadPorts = 1,
        numWritePorts = 1,
        numReadwritePorts = 1
      )

      // Read port
      mem.readPorts(0).address := io.addr1
      mem.readPorts(0).enable := true.B
      io.dataOut := mem.readPorts(0).data

      // Write port
      mem.writePorts(0).address := io.addr1
      mem.writePorts(0).enable := io.writeEn
      mem.writePorts(0).data := io.dataIn

      // Read/write port
      mem.readwritePorts(0).address := io.addr2
      mem.readwritePorts(0).enable := io.rwEn
      mem.readwritePorts(0).isWrite := io.rwWrite
      mem.readwritePorts(0).writeData := io.rwDataIn
      io.rwDataOut := mem.readwritePorts(0).readData
    }

    val dutA = ChiselGeneratorAnnotation(() => new MemoryModuleA)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[MemoryModuleA]]
      .design
    val dutB = ChiselGeneratorAnnotation(() => new MemoryModuleB)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[MemoryModuleB]]
      .design

    // These should have identical hashes despite potentially different internal port names
    Select.unreliableDedupHash(dutA.toDefinition) should be(Select.unreliableDedupHash(dutB.toDefinition))
  }

  // Test 16: Module with various primitive operations and edge cases
  "Select.unreliableDedupHash" should "produce identical hashes for modules with comprehensive primitive operations" in {
    class ComprehensiveModule extends Module {
      val io = IO(new Bundle {
        val a = Input(UInt(16.W))
        val b = Input(UInt(16.W))
        val sel = Input(Bool())
        val enable = Input(Bool())
        val out1 = Output(UInt(16.W))
        val out2 = Output(UInt(16.W))
        val out3 = Output(Bool())
      })

      // Various arithmetic operations
      val sum = io.a + io.b
      val diff = io.a - io.b
      val product = io.a * io.b

      // Logical operations
      val andResult = io.a & io.b
      val orResult = io.a | io.b
      val xorResult = io.a ^ io.b

      // Comparison operations
      val isEqual = io.a === io.b
      val isGreater = io.a > io.b

      // Bit manipulation
      val shifted = io.a << 1
      val extracted = io.a(7, 0)

      // Conditional assignments with complex expressions
      val reg1 = RegInit(0.U(16.W))
      val reg2 = RegInit(0.U(16.W))

      when(io.enable) {
        when(io.sel) {
          reg1 := Mux(isEqual, sum, diff)
          reg2 := andResult
        }.elsewhen(isGreater) {
          reg1 := product
          reg2 := orResult
        }.otherwise {
          reg1 := shifted
          reg2 := Cat(extracted, extracted)
        }
      }

      io.out1 := reg1
      io.out2 := reg2 + xorResult
      io.out3 := isEqual && isGreater
    }

    val dut1 = ChiselGeneratorAnnotation(() => new ComprehensiveModule)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ComprehensiveModule]]
      .design
    val dut2 = ChiselGeneratorAnnotation(() => new ComprehensiveModule)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ComprehensiveModule]]
      .design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test modules for different child module types

  // BlackBox modules for testing
  class TestBlackBoxInverter extends BlackBox {
    val io = IO(new Bundle {
      val in = Input(Bool())
      val out = Output(Bool())
    })
  }

  class TestBlackBoxAdder extends BlackBox {
    val io = IO(new Bundle {
      val a = Input(UInt(8.W))
      val b = Input(UInt(8.W))
      val sum = Output(UInt(8.W))
    })
  }

  class TestBlackBoxWithParams(width: Int) extends BlackBox(Map("WIDTH" -> width)) {
    val io = IO(new Bundle {
      val in = Input(UInt(width.W))
      val out = Output(UInt(width.W))
    })
  }

  // IntrinsicModule for testing
  class TestIntrinsicModule extends IntrinsicModule("TestIntrinsic") {
    val io = IO(new Bundle {
      val in = Input(Bool())
      val out = Output(Bool())
    })
  }

  class TestIntrinsicWithParams(name: String, value: Int) extends IntrinsicModule(name, Map("VALUE" -> value)) {
    val io = IO(new Bundle {
      val data = Input(UInt(8.W))
      val result = Output(UInt(8.W))
    })
  }

  // ExtModule for testing
  class TestExtModule extends ExtModule {
    val in = IO(Input(Bool()))
    val out = IO(Output(Bool()))
  }

  // Test 17: Module with BlackBox children
  "Select.unreliableDedupHash" should "produce identical hashes for modules with identical BlackBox children" in {
    class ModuleWithBlackBox extends Module {
      val io = IO(new Bundle {
        val in = Input(Bool())
        val out = Output(Bool())
      })
      val blackbox = Module(new TestBlackBoxInverter)
      blackbox.io.in := io.in
      io.out := blackbox.io.out
    }

    val dut1 = ChiselGeneratorAnnotation(() => new ModuleWithBlackBox)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithBlackBox]]
      .design
    val dut2 = ChiselGeneratorAnnotation(() => new ModuleWithBlackBox)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithBlackBox]]
      .design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 18: Module with IntrinsicModule children
  "Select.unreliableDedupHash" should "produce identical hashes for modules with identical IntrinsicModule children" in {
    class ModuleWithIntrinsic extends Module {
      val io = IO(new Bundle {
        val in = Input(Bool())
        val out = Output(Bool())
      })
      val intrinsic = Module(new TestIntrinsicModule)
      intrinsic.io.in := io.in
      io.out := intrinsic.io.out
    }

    val dut1 = ChiselGeneratorAnnotation(() => new ModuleWithIntrinsic)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithIntrinsic]]
      .design
    val dut2 = ChiselGeneratorAnnotation(() => new ModuleWithIntrinsic)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithIntrinsic]]
      .design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 19: Module with ExtModule children
  "Select.unreliableDedupHash" should "produce identical hashes for modules with identical ExtModule children" in {
    class ModuleWithExtModule extends Module {
      val io = IO(new Bundle {
        val in = Input(Bool())
        val out = Output(Bool())
      })
      val extmod = Module(new TestExtModule)
      extmod.in := io.in
      io.out := extmod.out
    }

    val dut1 = ChiselGeneratorAnnotation(() => new ModuleWithExtModule)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithExtModule]]
      .design
    val dut2 = ChiselGeneratorAnnotation(() => new ModuleWithExtModule)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithExtModule]]
      .design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 20: Module with multiple different child module types
  "Select.unreliableDedupHash" should "produce identical hashes for modules with mixed child module types" in {
    class ModuleWithMixedChildren extends Module {
      val io = IO(new Bundle {
        val in1 = Input(Bool())
        val in2 = Input(UInt(8.W))
        val out1 = Output(Bool())
        val out2 = Output(UInt(8.W))
      })

      // Regular Chisel module
      val reg = RegNext(io.in1)

      // BlackBox module
      val blackbox = Module(new TestBlackBoxInverter)
      blackbox.io.in := reg

      // IntrinsicModule
      val intrinsic = Module(new TestIntrinsicModule)
      intrinsic.io.in := blackbox.io.out

      // ExtModule
      val extmod = Module(new TestExtModule)
      extmod.in := intrinsic.io.out

      io.out1 := extmod.out
      io.out2 := io.in2 + 1.U
    }

    val dut1 = ChiselGeneratorAnnotation(() => new ModuleWithMixedChildren)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithMixedChildren]]
      .design
    val dut2 = ChiselGeneratorAnnotation(() => new ModuleWithMixedChildren)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithMixedChildren]]
      .design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 21: Modules with BlackBox children having different parameters should have different hashes
  "Select.unreliableDedupHash" should "produce different hashes for modules with BlackBox children having different parameters" in {
    class ModuleWithParamBlackBoxA extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      val blackbox = Module(new TestBlackBoxWithParams(8))
      blackbox.io.in := io.in
      io.out := blackbox.io.out
    }

    class ModuleWithParamBlackBoxB extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      val blackbox = Module(new TestBlackBoxWithParams(16))
      blackbox.io.in := io.in
      io.out := blackbox.io.out
    }

    val dutA = ChiselGeneratorAnnotation(() => new ModuleWithParamBlackBoxA)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithParamBlackBoxA]]
      .design
    val dutB = ChiselGeneratorAnnotation(() => new ModuleWithParamBlackBoxB)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithParamBlackBoxB]]
      .design

    Select.unreliableDedupHash(dutA.toDefinition) should not be (Select.unreliableDedupHash(dutB.toDefinition))
  }

  // Test 22: Modules with IntrinsicModule children having different parameters should have different hashes
  "Select.unreliableDedupHash" should "produce different hashes for modules with IntrinsicModule children having different parameters" in {
    class ModuleWithParamIntrinsicA extends Module {
      val io = IO(new Bundle {
        val data = Input(UInt(8.W))
        val result = Output(UInt(8.W))
      })
      val intrinsic = Module(new TestIntrinsicWithParams("TestOp", 42))
      intrinsic.io.data := io.data
      io.result := intrinsic.io.result
    }

    class ModuleWithParamIntrinsicB extends Module {
      val io = IO(new Bundle {
        val data = Input(UInt(8.W))
        val result = Output(UInt(8.W))
      })
      val intrinsic = Module(new TestIntrinsicWithParams("TestOp", 100))
      intrinsic.io.data := io.data
      io.result := intrinsic.io.result
    }

    val dutA = ChiselGeneratorAnnotation(() => new ModuleWithParamIntrinsicA)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithParamIntrinsicA]]
      .design
    val dutB = ChiselGeneratorAnnotation(() => new ModuleWithParamIntrinsicB)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithParamIntrinsicB]]
      .design

    Select.unreliableDedupHash(dutA.toDefinition) should not be (Select.unreliableDedupHash(dutB.toDefinition))
  }

  // Test 23: Modules with multiple instances of the same child module type
  "Select.unreliableDedupHash" should "produce identical hashes for modules with multiple instances of same child modules" in {
    class ModuleWithMultipleBlackBoxes extends Module {
      val io = IO(new Bundle {
        val in1 = Input(Bool())
        val in2 = Input(Bool())
        val out1 = Output(Bool())
        val out2 = Output(Bool())
      })

      val blackbox1 = Module(new TestBlackBoxInverter)
      val blackbox2 = Module(new TestBlackBoxInverter)

      blackbox1.io.in := io.in1
      blackbox2.io.in := io.in2

      io.out1 := blackbox1.io.out
      io.out2 := blackbox2.io.out
    }

    val dut1 = ChiselGeneratorAnnotation(() => new ModuleWithMultipleBlackBoxes)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithMultipleBlackBoxes]]
      .design
    val dut2 = ChiselGeneratorAnnotation(() => new ModuleWithMultipleBlackBoxes)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithMultipleBlackBoxes]]
      .design

    Select.unreliableDedupHash(dut1.toDefinition) should be(Select.unreliableDedupHash(dut2.toDefinition))
  }

  // Test 24: Modules with different types of BlackBox children should have different hashes
  "Select.unreliableDedupHash" should "produce different hashes for modules with different BlackBox child types" in {
    class ModuleWithInverterBlackBox extends Module {
      val io = IO(new Bundle {
        val in = Input(Bool())
        val out = Output(Bool())
      })
      val blackbox = Module(new TestBlackBoxInverter)
      blackbox.io.in := io.in
      io.out := blackbox.io.out
    }

    class ModuleWithAdderBlackBox extends Module {
      val io = IO(new Bundle {
        val a = Input(UInt(8.W))
        val b = Input(UInt(8.W))
        val sum = Output(UInt(8.W))
      })
      val blackbox = Module(new TestBlackBoxAdder)
      blackbox.io.a := io.a
      blackbox.io.b := io.b
      io.sum := blackbox.io.sum
    }

    val dutInverter = ChiselGeneratorAnnotation(() => new ModuleWithInverterBlackBox)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithInverterBlackBox]]
      .design
    val dutAdder = ChiselGeneratorAnnotation(() => new ModuleWithAdderBlackBox)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithAdderBlackBox]]
      .design

    Select.unreliableDedupHash(dutInverter.toDefinition) should not be (Select.unreliableDedupHash(
      dutAdder.toDefinition
    ))
  }

  // Test 25: Port direction changes should produce different hashes
  it should "produce different hashes for modules with different port directions" in {
    class ModuleWithInputPort extends Module {
      val data = IO(Input(UInt(8.W)))
      val reg = RegNext(data)
    }

    class ModuleWithOutputPort extends Module {
      val data = IO(Output(UInt(8.W)))
      val reg = Reg(UInt(8.W))
      data := reg
    }

    val dutInput = ChiselGeneratorAnnotation(() => new ModuleWithInputPort)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithInputPort]]
      .design

    val dutOutput = ChiselGeneratorAnnotation(() => new ModuleWithOutputPort)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithOutputPort]]
      .design

    Select.unreliableDedupHash(dutInput.toDefinition) should not be (Select.unreliableDedupHash(dutOutput.toDefinition))
  }

  // Test 26: Bundle field name changes should produce different hashes (demonstrating unreliability)
  it should "fail to deduplicate modules with identical structure but different bundle field names" in {
    class BundleA extends Bundle {
      val dataIn = Input(UInt(8.W))
      val dataOut = Output(UInt(8.W))
      val enable = Input(Bool())
    }

    class BundleB extends Bundle {
      val inputData = Input(UInt(8.W))
      val outputData = Output(UInt(8.W))
      val enableSignal = Input(Bool())
    }

    class ModuleWithBundleA extends Module {
      val io = IO(new BundleA)
      val reg = Reg(UInt(8.W))
      when(io.enable) {
        reg := io.dataIn
      }
      io.dataOut := reg
    }

    class ModuleWithBundleB extends Module {
      val io = IO(new BundleB)
      val reg = Reg(UInt(8.W))
      when(io.enableSignal) {
        reg := io.inputData
      }
      io.outputData := reg
    }

    val dutA = ChiselGeneratorAnnotation(() => new ModuleWithBundleA)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithBundleA]]
      .design

    val dutB = ChiselGeneratorAnnotation(() => new ModuleWithBundleB)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithBundleB]]
      .design

    // These modules are structurally identical but have different bundle field names
    // unreliableDedupHash should produce different hashes (showing it's unreliable for this case)
    Select.unreliableDedupHash(dutA.toDefinition) should not be (Select.unreliableDedupHash(dutB.toDefinition))
  }

  // Test 27: DefObject instances should be ignored for deduplication (demonstrating unreliability)
  it should "produce identical hashes for modules with different DefObject instances" in {
    import chisel3.properties.{Class, DynamicObject, Property}

    class TestClassA extends Class {
      override def desiredName = "TestClassA"
      val in = IO(Input(Property[Int]()))
      val out = IO(Output(Property[Int]()))
      out := in
    }

    class TestClassB extends Class {
      override def desiredName = "TestClassB"
      val data = IO(Input(Property[Int]()))
      val result = IO(Output(Property[Int]()))
      result := data
    }

    class ModuleWithObjectA extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      val reg = Reg(UInt(8.W))
      reg := io.in
      io.out := reg

      // Create a DefObject of TestClassA
      val objA = DynamicObject(new TestClassA)
    }

    class ModuleWithObjectB extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      val reg = Reg(UInt(8.W))
      reg := io.in
      io.out := reg

      // Create a DefObject of TestClassB (different class)
      val objB = DynamicObject(new TestClassB)
    }

    val dutA = ChiselGeneratorAnnotation(() => new ModuleWithObjectA)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithObjectA]]
      .design

    val dutB = ChiselGeneratorAnnotation(() => new ModuleWithObjectB)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithObjectB]]
      .design

    // These modules have identical hardware logic but different DefObject instances
    // unreliableDedupHash should produce identical hashes because DefObject is ignored
    // This demonstrates a false positive - modules that shouldn't be deduplicated have the same hash
    Select.unreliableDedupHash(dutA.toDefinition) should be(Select.unreliableDedupHash(dutB.toDefinition))
  }

  // Test 28: Modules with different intrinsic expressions should have different hashes
  it should "produce different hashes for modules with different intrinsic expressions" in {
    class ModuleWithIntrinsicA extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(32.W))
      })
      val reg = Reg(UInt(32.W))

      // IntrinsicExpr with parameter "TEST_PARAM" -> "FOO"
      val intrinsicResult = IntrinsicExpr("test_intrinsic", UInt(32.W), "TEST_PARAM" -> "FOO")(io.in)
      io.out := intrinsicResult + reg
    }

    class ModuleWithIntrinsicB extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(32.W))
      })
      val reg = Reg(UInt(32.W))

      // IntrinsicExpr with different parameter "TEST_PARAM" -> "BAR"
      val intrinsicResult = IntrinsicExpr("test_intrinsic", UInt(32.W), "TEST_PARAM" -> "BAR")(io.in)
      io.out := intrinsicResult + reg
    }

    val dutA = ChiselGeneratorAnnotation(() => new ModuleWithIntrinsicA)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithIntrinsicA]]
      .design

    val dutB = ChiselGeneratorAnnotation(() => new ModuleWithIntrinsicB)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[ModuleWithIntrinsicB]]
      .design

    // These modules have identical structure but different intrinsic expression parameters
    // unreliableDedupHash should produce different hashes because DefIntrinsicExpr includes parameters
    Select.unreliableDedupHash(dutA.toDefinition) should not be (Select.unreliableDedupHash(dutB.toDefinition))
  }
}
