package Chisel.testers

import Chisel._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

trait UnitTestRunners {
  def execute(t: => UnitTester): Boolean = TesterDriver.execute(() => t)
  def elaborate(t: => Module):   Unit    = Driver.elaborate(() => t)
}

class UnitTester extends BasicTester {
  case class Step(input_map: mutable.HashMap[Data,Int], output_map: mutable.HashMap[Data,Int])

  def rnd = Random
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
    require(io_port.dir == INPUT, s"poke error: $io_port not an input")
    require(test_actions.last.input_map.contains(io_port) == false,
      s"second poke to $io_port without step\nkeys ${test_actions.last.input_map.keys.mkString(",")}")

    test_actions.last.input_map(io_port) = value
  }

  def expect(io_port: Data, value: Int): Unit = {
    require(io_port.dir == OUTPUT, s"expect error: $io_port not an output")
    require(!test_actions.last.output_map.contains(io_port), s"second expect to $io_port without step")

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
      if( hash.contains(key) ) "%x".format(hash(key)) else "-"
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

    val pc             = Reg(init=UInt(0, 8))

    def create_vectors_for_input(input_port: Data): Unit = {
      var default_value = 0
      val input_values = Vec(
        test_actions.map { step =>
          default_value = step.input_map.getOrElse(input_port, default_value)
          UInt(default_value, input_port.width)
        }
      )
      input_port := input_values(pc)
    }

    dut_inputs.foreach { port =>
      for( vector_input <- port.flatten) {
        create_vectors_for_input(vector_input)
      }
    }

    def create_vectors_and_tests_for_output(output_port: Data): Unit = {
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

      printf("XXXX pc %x ok_to_test %x port_value %x expected %x",
        pc, ok_to_test_output_values(pc),
        output_port.toBits(),
        output_values(pc).toBits())

      when(ok_to_test_output_values(pc)) {
        when(output_port.toBits() != output_values(pc).toBits()) {
          printf(
            "Exerciser error: at step %d port io." + port_name(dut, output_port) + " value %x != %x, the expected value",
            pc,
            output_port.toBits(),
            output_values(pc).toBits()
          )
          assert(Bool(false), "Failed test")

        }
      }
    }

    dut_outputs.foreach { port =>
      for (vector_port <- port.flatten) {
        create_vectors_and_tests_for_output(vector_port)
      }
    }


    pc := pc + UInt(1)

    when(pc >= UInt(test_actions.length)) {
      printf(s"Stopping, end of tests, ${test_actions.length} steps\n")
      stop()
    }

  }
}
