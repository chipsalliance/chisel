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

  def install[T <: Module](dut: T): Unit = {
    /**
     * connect to the device under test by connecting each of it's io ports to an appropriate register
     */
    val io_input_registers = dut.io.elements.flatMap { case (name, element) =>
      if(element.dir == INPUT) {
        val new_reg = Reg(init = UInt(0, element.width))
        element := new_reg
        Some(new_reg)
      } else {
        None
      }
    }
    val io_input_register_from_index = io_input_registers.zipWithIndex.map { case(port, index) => index -> port }

    val io_output_ports = dut.io.elements.flatMap { case (name, element) =>
      if(element.dir == OUTPUT) Some(element) else None
    }
    val io_output_port_from_index = io_output_ports.zipWithIndex.map { case(port, index) => index -> port }.toMap

    io.done  := Bool(false)
    io.error := Bool(false)

    def make_instruction(op_code: Int, port_index: Int, value: Int) = {
      Cat(UInt(op_code, 8), UInt(port_index, 8), UInt(value, 32))
    }

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
    val operand_1   = instruction(47, 16)

    /**
     * This is how I would prefer to do the demux of the operand into the input port
     */
//    switch(operation) {
//      is(Bits(0)) {
//      switch(operation) {
//        input_port_from_index.map { case (key, element) =>
//          is(port_index) { element := element.fromBits(operand_1) }
//        }
//      }
//    }

    switch(operation) {
      is(Bits(0)) {
        new SwitchContext(operation) {
          io_input_register_from_index.map { case (key, element) =>
            is(port_index) {
              element := element.fromBits(operand_1)
            }
          }
        }
      }
      is(Bits(1)) {
        new SwitchContext(operation) {
          io_input_register_from_index.map { case (key, element) =>
            is(port_index) {
              when( ! element === element.fromBits(operand_1) ) {
                io.done := Bool(true)
                io.error := Bool(true)
                io.step_at_error := pc
              }
            }
          }
        }
      }
    }

    pc := pc + UInt(1)

    when(pc === UInt(program.length)) {
      io.done := Bool(true)
    }

  }
}

object UnitTester {
  def apply[T <: Module](gen: () => T): T = {
    gen()
  }
}
