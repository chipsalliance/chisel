package Chisel.testers

import Chisel._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

trait chiselUnitRunners {
  def execute(t: => UnitTester): Boolean = TesterDriver.execute(() => t)
  def elaborate(t: => Module): Circuit = Driver.elaborate(() => t)
}

case class TestAction(op_code: Bits, port : Data, value: Data)

class UnitTester extends Module {
  override val io = new Bundle {
    val running       = Bool(INPUT)
    val error         = Bool(OUTPUT)
    val step_at_error = UInt(OUTPUT)
    val done          = Bool(OUTPUT)
  }
  var max_width = Width(0)
  val set_input_op :: wait_for_op :: expect_op :: Nil = Enum(UInt(), 3)

  // Scala stuff
  val test_actions = new ArrayBuffer[TestAction]()
  val reference_to_port = new mutable.HashMap[UInt, Data]()

  def poke(io_port: Data, value: UInt): Unit = {
    require(io_port.dir == INPUT, s"poke error: $io_port not an input")

    println(s"io_port $io_port")
    println(s"ip_port.dir ${io_port.dir}")
    test_actions += TestAction(set_input_op, io_port, value)
    max_width = max_width.max(value.width)
  }

  def expect(io_port: Data, value: UInt): Unit = {
    require(io_port.dir == OUTPUT, s"expect error: $io_port not an output")

    println(s"io_port $io_port")
    println(s"ip_port.dir ${io_port.dir}")
    test_actions += TestAction(expect_op, io_port, value)
    max_width = max_width.max(value.width)
  }

  def step(number_of_cycles: Int) {}

  def construct_fsm(dut: Module): Unit = {

//    switch(operation) {
//      is(set_input_op) {
//        for ( (io_element, index) <- dut.io.elements.zipWithIndex) {
//
//        }
//        reference_to_port(port_index) := operand_1
//      }
//      is(expect_op) {}
//    }
  }
  def install[T <: Module](dut: T): Unit = {
    def make_instruction(op_code: Int, port_index: Int, value: Int) = {
      Cat(UInt(op_code, 8), UInt(port_index, 8), UInt(value, 32))
    }

    val port_to_index   = dut.io.elements.zipWithIndex.map { case ((name, element), index) =>
      element -> index
    }.toMap

    val input_port_from_index = dut.io.elements.zipWithIndex.flatMap { case ((name, element), index) =>
      if( element.dir == INPUT) {
        Some(index -> element)
      } else None
    }.toMap

    val program = Vec(
      Array(
        make_instruction(0, 1, 4),
        make_instruction(0, 1, 4)
      )
    )
    val pc             = Reg(init=UInt(0, 8))

    val instruction = program(pc)
    val operation   = instruction(7, 0)
    val port_index  = instruction(15, 8)
    val operand_1   = instruction(48, 16)

    io.done  := Bool(false)
    io.error := Bool(false)

    /*
    Wire a series of registers in front of the DUT's inputs
     */
    val dut_input_registers = dut.io.elements.flatMap { case (name, element) =>
      if(element.dir == INPUT) {
        val new_reg = Reg(init = UInt(0, element.width))
        element := new_reg
        Some(new_reg)
      } else {
        None
      }
    }

    val new_io = dut.io.fromBits(operand_1)
    dut.io <> new_io

//    switch(operation) {
//      input_port_from_index.map { case (key, element) =>
//        is(port_index) { element := operand_1 }
//      }
//    }

    val sc = new SwitchContext(operation) {
      input_port_from_index.map { case (key, element) =>
        is(port_index) { element := operand_1 }
      }
    }

    pc := pc + UInt(1)

    for {
      io_element      <- dut.io.elements
      input_reference <- test_actions
    } {
      if ( io_element._2 == input_reference.port ) { print("---->")}
      println(s"x is ${io_element._2} $input_reference")

//      input_port_from_index(input_reference.port) := input_reference.output.toBits()
    }

    construct_fsm(dut)


  }
}

object UnitTester {
  def apply[T <: Module](gen: () => T): T = {
    gen()
  }
}
