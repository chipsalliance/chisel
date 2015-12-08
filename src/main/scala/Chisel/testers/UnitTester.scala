package Chisel.testers

import Chisel._

trait chiselUnitRunners {
  def execute(t: => UnitTester): Boolean = TesterDriver.execute(() => t)
  def elaborate(t: => Module): Circuit = Driver.elaborate(() => t)
}

case class SimulationStep(input : Data, output: Data, out_mask: Data)

class UnitTester extends Module {
  override val io = new Bundle {
    val running       = Bool(INPUT)
    val error         = Bool(OUTPUT)
    val step_at_error = UInt(OUTPUT)
    val done          = Bool(OUTPUT)
  }

  def poke(io_port: Data, value: BigInt) {}
  def expect(io_port: Data, value: BigInt) {}
}

object UnitTester {
  def apply[T <: Module](gen: () => T): T = {
    gen()
  }
}
