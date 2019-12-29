// See LICENSE for license details.

package examples

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

import chiselTests.ChiselFlatSpec
import chisel3.testers.BasicTester
import chisel3._
import chisel3.stage.phases.Elaborate
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage, DesignAnnotation}
import chisel3.util._
import firrtl.Namespace

class SimpleVendingMachineIO extends Bundle {
  val nickel = Input(Bool())
  val dime   = Input(Bool())
  val dispense  = Output(Bool())
}

// Superclass for vending machines with very simple IO
abstract class SimpleVendingMachine extends Module {
  val io = IO(new SimpleVendingMachineIO)
  assert(!(io.nickel && io.dime), "Only one of nickel or dime can be input at a time!")
}

// Vending machine implemented with a Finite State Machine
@SerialVersionUID(100L)
class FSMVendingMachine extends SimpleVendingMachine with Serializable {
  UInt.apply
  val sIdle :: s5 :: s10 :: s15 :: sOk :: Nil = Enum(5)
  val state = RegInit(sIdle)

  switch (state) {
    is (sIdle) {
      when (io.nickel) { state := s5 }
      when (io.dime)   { state := s10 }
    }
    is (s5) {
      when (io.nickel) { state := s10 }
      when (io.dime)   { state := s15 }
    }
    is (s10) {
      when (io.nickel) { state := s15 }
      when (io.dime)   { state := sOk }
    }
    is (s15) {
      when (io.nickel) { state := sOk }
      when (io.dime)   { state := sOk }
    }
    is (sOk) {
      state := sIdle
    }
  }
  io.dispense := (state === sOk)
}

class VerilogVendingMachine extends BlackBox {
  // Because this is a blackbox, we must explicity add clock and reset
  val io = IO(new SimpleVendingMachineIO {
    val clock = Input(Clock())
    val reset = Input(Reset())
  })
}

// Shim because Blackbox io is slightly different than normal Chisel Modules
class VerilogVendingMachineWrapper extends SimpleVendingMachine {
  val impl = Module(new VerilogVendingMachine)
  impl.io.clock := clock
  impl.io.reset := reset
  impl.io.nickel := io.nickel
  impl.io.dime := io.dime
  io.dispense := impl.io.dispense
}

// Accept a reference to a SimpleVendingMachine so it can be constructed inside
// the tester (in a call to Module.apply as required by Chisel
class SimpleVendingMachineTester(mod: => SimpleVendingMachine) extends BasicTester {

  val dut = Module(mod)

  val (cycle, done) = Counter(true.B, 10)
  when (done) { stop(); stop() } // Stop twice because of Verilator

  val nickelInputs = VecInit(true.B, true.B, true.B, true.B, true.B, false.B, false.B, false.B, true.B, false.B)
  val dimeInputs   = VecInit(false.B, false.B, false.B, false.B, false.B, true.B, true.B, false.B, false.B, true.B)
  val expected     = VecInit(false.B, false.B, false.B, false.B, true.B , false.B, false.B, true.B, false.B, false.B)

  dut.io.nickel := nickelInputs(cycle)
  dut.io.dime := dimeInputs(cycle)
  assert(dut.io.dispense === expected(cycle))
}

class SimpleVendingMachineSpec extends ChiselFlatSpec {
  "An FSM implementation of a vending machine" should "work" in {
    assertTesterPasses { new SimpleVendingMachineTester(new FSMVendingMachine) }
  }
  "An Verilog implementation of a vending machine" should "work" in {
    assertTesterPasses(new SimpleVendingMachineTester(new VerilogVendingMachineWrapper),
                       List("/chisel3/VerilogVendingMachine.v"))
  }
}

object DynamicCachingExamples extends App {
  val stage = new ChiselStage()
  val annos = ChiselGeneratorAnnotation(() => new FSMVendingMachine).elaborate
  val vmachine = annos.collectFirst {
    case DesignAnnotation(design: FSMVendingMachine) => design
  }.get

  // (2) write the instance out to a file

  val oos = new ObjectOutputStream(new FileOutputStream("/tmp/vmachine"))
  oos.writeObject(vmachine)
  oos.close

  // (3) read the object back in
  val fsm = ChiselGeneratorAnnotation(() => {
    val in = new ObjectInputStream(new FileInputStream("/tmp/vmachine")) {
      override def resolveClass(desc: java.io.ObjectStreamClass): Class[_] = {
        try { Class.forName(desc.getName, false, getClass.getClassLoader) }
        catch { case ex: ClassNotFoundException => super.resolveClass(desc) }
      }
    }
    val ois = new ObjectInputStream(new FileInputStream("/tmp/vmachine"))
    val obj = ois.readObject.asInstanceOf[FSMVendingMachine]
    ois.close
    obj
    //val obj = in.readObject().asInstanceOf[FSMVendingMachine]
    //in.close
    //obj
  })//.reload.asInstanceOf[FSMVendingMachine]

  // (4) print the object that was read back in
  //println(fsm.state.toTarget)
  println("Done")
}