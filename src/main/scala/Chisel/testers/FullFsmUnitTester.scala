// See LICENSE for license details.

package Chisel.testers

import Chisel._

//import scala.collection.mutable
//import scala.collection.mutable.ArrayBuffer

/**
  * This tester is a work in progress towards a generic FSM that can test arbitrary circuits
  */
//noinspection ScalaStyle
class FullFsmUnitTester extends Module {
  override val io = new Bundle {
    val running       = Bool(INPUT)
    val error         = Bool(OUTPUT)
    val step_at_error = UInt(OUTPUT)
    val done          = Bool(OUTPUT)
  }
//  var max_width = Width(0)
//  val set_input_op :: wait_for_op :: expect_op :: Nil = Enum(UInt(), 3)
//
//  // Scala stuff
//  val test_actions = new ArrayBuffer[Step]()
//  step(1) // gives us a slot to put in our input and outputs from beginning
//
//  def poke(io_port: Data, value: Int): Unit = {
//    require(io_port.dir == INPUT, s"poke error: $io_port not an input")
//    require(!test_actions.last.input_map.contains(io_port), s"second poke to $io_port without step")
//
//    println(s"io_port $io_port")
//    println(s"ip_port.dir ${io_port.dir}")
//
//    test_actions.last.input_map(io_port) = value
//  }
//
//  def expect(io_port: Data, value: Int): Unit = {
//    require(io_port.dir == OUTPUT, s"expect error: $io_port not an output")
//    require(!test_actions.last.output_map.contains(io_port), s"second expect to $io_port without step")
//
//    println(s"io_port $io_port")
//    println(s"ip_port.dir ${io_port.dir}")
//    test_actions.last.output_map(io_port) = value
//  }
//
//  def step(number_of_cycles: Int): Unit = {
//    test_actions += new Step(
//      new mutable.HashMap[Data, Int](),
//      new mutable.HashMap[Data, Int]()
//    )
//  }
//
//  def install[T <: Module](dut: T): Unit = {
//    /**
//     * connect to the device under test by connecting each of it's io ports to an appropriate register
//     */
//    val io_input_registers = dut.io.elements.flatMap { case (name, element) =>
//      if(element.dir == INPUT) {
//        val new_reg = Reg(init = UInt(0, element.width))
//        element := new_reg
//        Some(new_reg)
//      } else {
//        None
//      }
//    }
//    val io_input_register_from_index = io_input_registers.zipWithIndex.map { case(port, index) => index -> port }
//
//    val io_output_ports = dut.io.elements.flatMap { case (name, element) =>
//      if(element.dir == OUTPUT) Some(element) else None
//    }
//    val io_output_port_from_index = io_output_ports.zipWithIndex.map { case(port, index) => index -> port }.toMap
//
//    io.done  := Bool(false)
//    io.error := Bool(false)
//
//    def make_instruction(op_code: Int, port_index: Int, value: Int) = {
//      Cat(UInt(op_code, 8), UInt(port_index, 8), UInt(value, 32))
//    }
//
//    val program = Vec(
//      Array(
//        make_instruction(0, 1, 4),
//        make_instruction(0, 1, 4)
//      )
//    )
//    val pc             = Reg(init=UInt(0, 8))
//
//    val instruction = program(pc)
//    val operation   = instruction(7, 0)
//    val port_index  = instruction(15, 8)
//    val operand_1   = instruction(47, 16)
//
//    switch(operation) {
//      is(Bits(0)) {
//        io_input_register_from_index.map { case (index, port)=>
//          when(UInt(index) === port_index) {
//            port := operand_1
//          }
//        }
//      }
//      is(Bits(1)) {
//        io_output_port_from_index.map { case (index, port) =>
//          when(UInt(index) === port_index && Bool(port.fromBits(operand_1) != port)) {
//            io.done          := Bool(true)
//            io.error         := Bool(true)
//            io.step_at_error := pc
//          }
//        }
//      }
//    }
//
//    pc := pc + UInt(1)
//
//    when(pc >= UInt(program.length)) {
//      io.done := Bool(true)
//    }
//
//  }
}