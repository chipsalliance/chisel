// See LICENSE for license details.

package Chisel.testers

import Chisel._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

trait UnitTestRunners {
  def execute(t: => BasicTester): Boolean = TesterDriver.execute(() => t)
  def elaborate(t: => Module):   Unit    = Driver.elaborate(() => t)
}

/**
 * Use a UnitTester to constuct a test harness for a chisel module
 * this module will be canonically referred to as the device_under_test, often simply as c in
 * a unit test, and also dut
 * The UnitTester is used to put series of values (as chisel.Vec's) into the ports of the dut io which are INPUTs
 * At specified times it check the dut's io OUTPUT ports to see that they match a specific value
 * The vec's are assembled through the following API
 * poke, expect and step, pokes
 *
 * Example:
 *
 * class Adder(width:Int) extends Module {
 *   val io = new Bundle {
 *     val in0 : UInt(INPUT, width=width)
 *     val in1 : UInt(INPUT, width=width)
 *     val out : UInt(OUTPUT, width=width)
 *   }
 * class AdderTester extends UnitTester {
 *   val device_under_test = Module( new Adder(32) )
 *   val c = device_under_test
 *   poke(c.io.in0, 5); poke(c.io.in1, ); poke(
 *
 */
class UnitTester extends BasicTester {
  case class Step(input_map: mutable.HashMap[Data,Int], output_map: mutable.HashMap[Data,Int])

  def rnd: Random = Random  // convenience method for writing tests

  val ports_referenced = new mutable.HashSet[Data]

  // Scala stuff
  val test_actions = new ArrayBuffer[Step]()
  step(1) // gives us a slot to put in our input and outputs from beginning

  def poke(io_port: Data, value: Int): Unit = {
    require(io_port.dir == INPUT, s"poke error: $io_port not an input")
    require(!test_actions.last.input_map.contains(io_port),
      s"second poke to $io_port without step\nkeys ${test_actions.last.input_map.keys.mkString(",")}")

    ports_referenced += io_port
    test_actions.last.input_map(io_port) = value
  }
//  def poke(io_port: Data, bool_value: Boolean) = poke(io_port, if(bool_value) 1 else 0)

  def expect(io_port: Data, value: Int): Unit = {
    require(io_port.dir == OUTPUT, s"expect error: $io_port not an output")
    require(!test_actions.last.output_map.contains(io_port), s"second expect to $io_port without step")

    ports_referenced += io_port
    test_actions.last.output_map(io_port) = value
  }
  def expect(io_port: Data, bool_value: Boolean): Unit = expect(io_port, if(bool_value) 1 else 0)

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
    val dut_inputs  = dut.io.flatten.filter( port => port.dir == INPUT  && ports_referenced.contains(port) )
    val dut_outputs = dut.io.flatten.filter( port => port.dir == OUTPUT && ports_referenced.contains(port))

    val port_to_name = {
      val port_to_name_accumulator = new mutable.HashMap[Data, String]()

      println("="*80)
      println("Device under test: io bundle")
      println("%10s %10s %s".format("direction", "referenced", "name"))
      println("-"*80)
      def parse_bundle(b: Bundle, name: String = ""): Unit = {
        for ((n, e) <- b.elements) {
          val new_name = name + (if(name.length > 0 ) "." else "" ) + n
          port_to_name_accumulator(e) = new_name
          println("%10s %5s      %s".format(e.dir, if( ports_referenced.contains(e)) "Y" else " ", new_name))

          e match {
            case bb: Bundle  => parse_bundle(bb, new_name)
            case vv: Vec[_]  => parse_vecs(vv, new_name)
            case ee: Element => {}
            case _           => {
              throw new Exception(s"bad bundle member ${new_name} $e")
            }
          }
        }
      }
      def parse_vecs[T<:Data](b: Vec[T], name: String = ""): Unit = {
        for ((e, i) <- b.zipWithIndex) {
          val new_name = name + s"($i)"
          port_to_name_accumulator(e) = new_name
          println("%10s %5s      %s".format(e.dir, if( ports_referenced.contains(e)) "Y" else " ", new_name))

          e match {
            case bb: Bundle  => parse_bundle(bb, new_name)
            case vv: Vec[_]  => parse_vecs(vv, new_name)
            case ee: Element => {}
            case _           => {
              throw new Exception(s"bad bundle member ${new_name} $e")
            }
          }
        }
      }

      parse_bundle(dut.io)
      println("="*80)

      port_to_name_accumulator
    }
    /**
     *  Print a title for the testing state table
     */
    if(ports_referenced.nonEmpty) {
      val max_col_width = ports_referenced.map { port =>
        Array(port_to_name(port).length, port.getWidth / 4).max // width/4 is how wide value might be in hex
      }.max + 2
      val (string_col_template, number_col_template) = (s"%${max_col_width}s", s"%${max_col_width}x")

      println("=" * 80)
      println("UnitTester state table")
      println(
        "%6s".format("step") +
          dut_inputs.map { dut_input => string_col_template.format(port_to_name(dut_input)) }.mkString +
          dut_outputs.map { dut_output => string_col_template.format(port_to_name(dut_output)) }.mkString
      )
      println("-" * 80)
      /**
        * prints out a table form of input and expected outputs
        */
      def val_str(hash: mutable.HashMap[Data, Int], key: Data): String = {
        if (hash.contains(key)) "%x".format(hash(key)) else "-"
      }
      test_actions.zipWithIndex.foreach { case (step, step_number) =>
        print("%6d".format(step_number))
        for (port <- dut_inputs) {
          print(string_col_template.format(val_str(step.input_map, port)))
        }
        for (port <- dut_outputs) {
          print(string_col_template.format(val_str(step.output_map, port)))
        }
        println()
      }
      println("=" * 80)
    }

    val pc             = Reg(init=UInt(0, log2Up(test_actions.length)+2))

    def log_referenced_ports: Unit = {
      val format_statement = new StringBuilder()
      val port_to_display  = new ArrayBuffer[Data]()
      format_statement.append("pc: %x")
      port_to_display.append(pc)

      for( dut_input <- dut_inputs ) {
        format_statement.append(",  " + port_to_name(dut_input)+": %x")
        port_to_display.append(dut_input)
      }
      for( dut_output <- dut_outputs ) {
        format_statement.append(",   " + port_to_name(dut_output)+": %x")
        port_to_display.append(dut_output)
      }
      printf(format_statement.toString(), port_to_display.map{_.toBits()}.toSeq :_* )
    }

    log_referenced_ports


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

    dut_inputs.foreach { port => create_vectors_for_input(port) }

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

      when(ok_to_test_output_values(pc)) {
        when(output_port.toBits() === output_values(pc).toBits()) {
//          printf("    passed -- " + port_to_name(output_port) + ":  %x",
//            output_port.toBits()
//          )
        }.otherwise {
          printf("    failed -- port " + port_to_name(output_port) + ":  %x expected %x",
            output_port.toBits(),
            output_values(pc).toBits()
          )
          assert(Bool(false), "Failed test")
          // TODO: Figure out if we want to stop here
//          stop()
        }
      }
    }

    dut_outputs.foreach { port => create_vectors_and_tests_for_output(port) }

    pc := pc + UInt(1)

    when(pc >= UInt(test_actions.length - 1)) {
      printf(s"Stopping, end of tests, ${test_actions.length} steps\n")
      stop()
    }

  }
}
