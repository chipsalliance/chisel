// See LICENSE for license details.

package chisel3.iotesters

import chisel3._
import chisel3.util._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Use a UnitTester to construct a test harness for a chisel module
  * this module will be canonically referred to as the device_under_test, often simply as c in
  * a unit test, and also dut
  * The UnitTester is used to put series of values (as chisel3.Vec's) into the ports of the dut io which are INPUT
  * At specified times it check the dut io OUTPUT ports to see that they match a specific value
  * The vec's are assembled through the following API
  * poke, expect and step, pokes
  *
  * @example
  * {{{
  *
  * class Adder(width:Int) extends Module {
  *   val io = new Bundle {
  *     val in0 : UInt(INPUT, width=width)
  *     val in1 : UInt(INPUT, width=width)
  *     val out : UInt(OUTPUT, width=width)
  *   }
  * }
  * class AdderTester extends UnitTester {
  *   val device_under_test = Module( new Adder(32) )
  *
  *   testBlock {
  *     poke(c.io.in0, 5)
  *     poke(c.io.in1, 7)
  *     expect(c.io.out, 12)
  *   }
  * }
  * }}}
  */
abstract class SteppedHWIOTester extends HWIOTester {
  case class Step(input_map: mutable.HashMap[Data,BigInt], output_map: mutable.HashMap[Data,BigInt])

  // Scala stuff
  private val test_actions = new ArrayBuffer[Step]()
  step(1) // gives us a slot to put in our input and outputs from beginning

  // Since dir is no longer a method on Data, we need some help here.
  // TODO: replace Data with Element (which has a .dir), or better yet, some internal tester object
  def dir(target: Data): Direction = {
    target match {
      case e: Element => e.dir
      case _ => chisel3.NODIR
    }
  }

  def poke(io_port: Data, value: BigInt): Unit = {
    require(dir(io_port) == INPUT, s"poke error: $io_port not an input")
    require(!test_actions.last.input_map.contains(io_port),
      s"second poke to $io_port without step\nkeys ${test_actions.last.input_map.keys.mkString(",")}")

    test_actions.last.input_map(io_port) = value
  }
//  def poke(io_port: Data, bool_value: Boolean) = poke(io_port, if(bool_value) 1 else 0)

  def expect(io_port: Data, value: BigInt): Unit = {
    require(dir(io_port) == OUTPUT, s"expect error: $io_port not an output")
    require(!test_actions.last.output_map.contains(io_port), s"second expect to $io_port without step")

    test_actions.last.output_map(io_port) = value
  }
  def expect(io_port: Data, bool_value: Boolean): Unit = expect(io_port, BigInt(if(bool_value) 1 else 0))

  def step(number_of_cycles: Int): Unit = {
    test_actions ++= Array.fill(number_of_cycles) {
      new Step(new mutable.HashMap[Data, BigInt](), new mutable.HashMap[Data, BigInt]())
    }
  }

  private def name(port: Data): String = io_info.port_to_name(port)

  //noinspection ScalaStyle
  private def printStateTable(): Unit = {
    val default_table_width = 80

    if(io_info.ports_referenced.nonEmpty) {
      val max_col_width = io_info.ports_referenced.map { port =>
        Array(name(port).length, port.getWidth / 4).max // width/4 is how wide value might be in hex
      }.max + 2
      val string_col_template = s"%${max_col_width}s"
//      val number_col_template = s"%${max_col_width}x"

      println("=" * default_table_width)
      println("UnitTester state table")
      println(
        "%6s".format("step") +
          io_info.dut_inputs.map { dut_input => string_col_template.format(name(dut_input)) }.mkString +
          io_info.dut_outputs.map { dut_output => string_col_template.format(name(dut_output)) }.mkString
      )
      println("-" * default_table_width)
      /**
        * prints out a table form of input and expected outputs
        */
      def val_str(hash: mutable.HashMap[Data, BigInt], key: Data): String = {
        if (hash.contains(key)) "%d".format(hash(key)) else "-"
      }
      test_actions.zipWithIndex.foreach { case (step, step_number) =>
        print("%6d".format(step_number))
        for (port <- io_info.dut_inputs) {
          print(string_col_template.format(val_str(step.input_map, port)))
        }
        for (port <- io_info.dut_outputs) {
          print(string_col_template.format(val_str(step.output_map, port)))
        }
        println()
      }
      println("=" * default_table_width)
    }
  }

  private def createVectorsForInput(input_port: Data, counter: Counter): Unit = {
    var default_value = BigInt(0)
    val input_values = Vec(
      test_actions.map { step =>
        default_value = step.input_map.getOrElse(input_port, default_value)
        (default_value).asUInt(input_port.getWidth.W)
      }
    )
    input_port := input_values(counter.value)
  }

  private def createVectorsAndTestsForOutput(output_port: Data, counter: Counter): Unit = {
    val output_values = Vec(
      test_actions.map { step =>
        output_port.cloneType.fromBits((step.output_map.getOrElse(output_port, BigInt(0))).asUInt)
      }
    )
    val ok_to_test_output_values = Vec(
      test_actions.map { step =>
        (step.output_map.contains(output_port)).asBool
      }
    )

    when(ok_to_test_output_values(counter.value)) {
      when(output_port.asUInt() === output_values(counter.value).asUInt()) {
                  logPrintfDebug("    passed step %d -- " + name(output_port) + ":  %d\n",
                    counter.value,
                    output_port.asUInt()
                  )
      }.otherwise {
        printf("    failed on step %d -- port " + name(output_port) + ":  %d expected %d\n",
          counter.value,
          output_port.asUInt(),
          output_values(counter.value).asUInt()
        )
        // TODO: Use the following line instead of the unadorned assert when firrtl parsing error issue #111 is fixed
        // assert(false.B, "Failed test")
        assert(false.B)
        stop()
      }
    }
  }

  private def processEvents(): Unit = {
    test_actions.foreach { case step =>
      io_info.ports_referenced ++= step.input_map.keys
      io_info.ports_referenced ++= step.output_map.keys
    }
  }

  override def finish(): Unit = {
    io_info = new IOAccessor(device_under_test.io)

    processEvents()

    val pc             = Counter(test_actions.length)
    val done           = Reg(init = false.B)

    when(!done) {
      io_info.dut_inputs.filter(io_info.ports_referenced.contains).foreach { port => createVectorsForInput(port, pc) }
      io_info.dut_outputs.filter(io_info.ports_referenced.contains).foreach { port => createVectorsAndTestsForOutput(port, pc) }

      when(pc.inc()) {
        printf(s"Stopping, end of tests, ${test_actions.length} steps\n")
        done := true.B
        stop()
      }
    }
    io_info.showPorts("".r)
    printStateTable()
  }
}
