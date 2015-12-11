package Chisel.testers

import Chisel._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

trait UnitTestRunners {
  def execute(t: => UnitTester): Boolean = {
    TesterDriver.execute(() => t)
  }
  def elaborate(t: => Module): Circuit = {
    Driver.elaborate(() => t)
  }
}

case class Step(input_map: mutable.HashMap[Data,Int], output_map: mutable.HashMap[Data,Int])

class UnitTester extends Module {
  override val io = new Bundle {
    val running       = Bool(INPUT)
    val error         = Bool(OUTPUT)
    val step_at_error = UInt(OUTPUT)
    val done          = Bool(OUTPUT)
  }
  var max_width = Width(0)
  val set_input_op :: wait_for_op :: expect_op :: Nil = Enum(UInt(), 3)

  def port_name(dut: Module, port_to_find: Data) : String = {
    dut.io.elements.foreach { case (name, port) =>
        if( port == port_to_find) return name
    }
    port_to_find.toString
  }

  // Scala stuff
  val test_actions = new ArrayBuffer[Step]()
  step(1) // gives us a slot to put in our input and outputs from beginning

  def poke(io_port: Data, value: Int): Unit = {
//    println(s"io_port $io_port, len ${test_actions.last.input_map.size} " +
//            s"ip_port.dir ${io_port.dir}")

    require(io_port.dir == INPUT, s"poke error: $io_port not an input")
//    require(test_actions.last.input_map.contains(io_port) == false,
//      s"second poke to $io_port without step\nkeys ${test_actions.last.input_map.keys.mkString(",")}")

    test_actions.last.input_map(io_port) = value
  }

  def expect(io_port: Data, value: Int): Unit = {
    require(io_port.dir == OUTPUT, s"expect error: $io_port not an output")
    require(!test_actions.last.output_map.contains(io_port), s"second expect to $io_port without step")

//    println(s"io_port $io_port ip_port.dir ${io_port.dir}")
    test_actions.last.output_map(io_port) = value
  }

  def step(number_of_cycles: Int): Unit = {
    test_actions ++= Array.fill(number_of_cycles) {
      new Step(
        new mutable.HashMap[Data, Int](),
        new mutable.HashMap[Data, Int]()
      )
    }
  }

  def install[T <: Module](dut: T): Unit = {
    /**
     * connect to the device under test by connecting each of it's io ports to an appropriate register
     */
    val dut_inputs = dut.io.elements.flatMap { case (name, element) =>
      if(element.dir == INPUT) Some(element) else None
    }
    val dut_outputs = dut.io.elements.flatMap { case (name, element) =>
      if(element.dir == OUTPUT) Some(element) else None
    }

    /**
     * prints out a table form of input and expected outputs
     */
    println(
      "%6s".format("step") +
        dut_inputs.map { dut_input => "%8s".format(port_name(dut, dut_input))}.mkString +
        dut_outputs.map { dut_output => "%8s".format(port_name(dut, dut_output))}.mkString
    )
    def val_str(hash : mutable.HashMap[Data, Int], key: Data) : String = {
      if( hash.contains(key) ) hash(key).toString else "-"
    }
    test_actions.zipWithIndex.foreach { case (step, step_number) =>
      print("%6d".format(step_number))
      for(port <- dut_inputs) {
        print("%8s".format(val_str(step.input_map, port)))
      }
      for(port <- dut_outputs) {
        print("%8s".format(val_str(step.output_map, port)))
      }
      println()
    }
    io.done  := Bool(false)
    io.error := Bool(false)


    val pc             = Reg(init=UInt(0, 8))

    io.step_at_error := pc

    dut_inputs.foreach { input_port =>
      var default_value = 0
      val input_values = Vec(
        test_actions.map { step =>
          default_value = step.input_map.getOrElse(input_port, default_value)
          UInt(default_value, input_port.width)
        }
      )
      input_port := input_values(pc)
    }

    dut_outputs.foreach { output_port =>
      val output_values = Vec(
        test_actions.map { step =>
          output_port.fromBits(UInt(step.output_map.getOrElse(output_port, 0)))
        }
      )
      val ok_to_test_output_values = Vec(
        test_actions.map { step =>
          Bool(step.output_map.contains(output_port))
        }
      )

//      when(ok_to_test_output_values(pc) && output_port === output_values(pc))) {
      when(ok_to_test_output_values(pc)) {
        when(output_port.toBits() != output_values(pc).toBits()) {
          io.error := Bool(true)
          io.done  := Bool(true)
        }
      }
    }


    pc := pc + UInt(1)

    when(pc >= UInt(test_actions.length)) {
      io.done := Bool(true)
    }

  }
}

object UnitTester {
  def apply[T <: Module](gen: () => T): T = {
    gen()
  }
}
