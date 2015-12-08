package Chisel.testers

import Chisel._

import scala.collection.mutable.ArrayBuffer

trait chiselUnitRunners {
  def execute(t: => UnitTester): Boolean = TesterDriver.execute(() => t)
  def elaborate(t: => Module): Circuit = Driver.elaborate(() => t)
}

case class SimulationStep(input : Data, output: Data)

class UnitTester extends Module {
  override val io = new Bundle {
    val running       = Bool(INPUT)
    val error         = Bool(OUTPUT)
    val step_at_error = UInt(OUTPUT)
    val done          = Bool(OUTPUT)
  }
  var max_width = Width(0)

  val input_vector = new ArrayBuffer[SimulationStep]()

  def poke(io_port: Data, value: UInt): Unit = {
    // TODO: confirm io.dir == INPUT
    println(s"io_port $io_port")
    println(s"ip_port.dir ${io_port.dir}")
    input_vector += SimulationStep(io_port, value)
    max_width = max_width.max(value.width)
  }

  def expect(io_port: Data, value: UInt): Unit = {
    // TODO: confirm io_port.dir == OUTPUT
  }
  def step(number_of_cyles: Int) {}

  def construct_fsm(): Unit = {
    val pc   = Reg(init=UInt(0, 8))

  }
  def install[T <: Module](dut: T): Unit = {
    for {
      io_element      <- dut.io.elements
      input_reference <- input_vector
    } {
      if ( io_element._2 == input_reference.input) { print("---->")}
      println(s"x is ${io_element._2} $input_reference")

      input_reference.input := input_reference.output.toBits()
    }


  }
}

object UnitTester {
  def apply[T <: Module](gen: () => T): T = {
    gen()
  }
}
