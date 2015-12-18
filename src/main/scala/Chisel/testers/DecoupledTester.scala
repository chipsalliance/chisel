package Chisel.testers

import Chisel._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

//TODO: all ins for a given event must be part of same DecoupledIO
//TODO: all outs for a given event must be part of the same ValidIO
//TODO: io not allowed directly on ready or valid
/**
 * Base class supports implementation of engines that test circuits whose use Decoupled IO
 */
abstract class DecoupledTester extends BasicTester {
  def device_under_test : Module

  case class Step(input_map: mutable.HashMap[Data,Int], output_map: mutable.HashMap[Data,Int])

  val ports_referenced = new mutable.HashSet[Data]

  // Scala stuff
  val test_actions = new ArrayBuffer[Step]()
  val dut_inputs  = device_under_test.io.flatten.filter( port => port.dir == INPUT  && ports_referenced.contains(port))
  val dut_outputs = device_under_test.io.flatten.filter( port => port.dir == OUTPUT && ports_referenced.contains(port))

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

    parse_bundle(device_under_test.io)
    println("="*80)

    port_to_name_accumulator
  }

  def poke(io_port: Data, value: Int): Unit = {
    require(io_port.dir == INPUT, s"poke error: $io_port not an input")
    require(
      test_actions.last.input_map.contains(io_port) == false,
      s"second poke to $io_port without step\nkeys ${test_actions.last.input_map.keys.mkString(",")}"
    )
    require(
      !port_to_name(io_port).endsWith(".ready"),
      s"Error: cannot poke to ready in poke(${port_to_name(io_port)}, $value)"
    )
    require(
      !port_to_name(io_port).endsWith(".valid"),
      s"Error: cannot poke to ready in poke(${port_to_name(io_port)}, $value)"
    )
    ports_referenced += io_port
    test_actions.last.input_map(io_port) = value
  }
  def expect(io_port: Data, value: Int): Unit = {
    require(io_port.dir == OUTPUT, s"expect error: $io_port not an output")
    require(!test_actions.last.output_map.contains(io_port), s"second expect to $io_port without step")

    ports_referenced += io_port
    test_actions.last.output_map(io_port) = value
  }
  def step: Unit = {
    test_actions += new Step(new mutable.HashMap[Data, Int](), new mutable.HashMap[Data, Int]())
  }

  val port_to_decoupled_io = mutable.HashMap[Data, DecoupledIO]()

  def create_vectors_for_input(input_port: Data): Unit = {
    var default_value = 0
    val input_values = Vec(
      test_actions.map { step =>
        default_value = step.input_map.getOrElse(input_port, default_value)
        UInt(default_value, input_port.width)
      }
    )
  }

  dut_inputs.foreach { port => create_vectors_for_input(port) }

  def create_vectors_and_tests_for_output(output_port: Data): Unit = {
    val output_values = Vec(
      test_actions.map { step =>
        output_port.fromBits(UInt(step.output_map.getOrElse(output_port, 0)))
      }
    )
    when(port_to_decoupled_io(output_port).ready === Bool(true)) {

    }
  }

  dut_outputs.foreach { port => create_vectors_and_tests_for_output(port) }


  //TODO: validate that all pokes ports are members of the same DecoupledIO
  //TODO: validate the same for expects ports
  /**
   * create an event in which poke values will be loaded when corresponding ready
   * expect values will be validated when corresponding valid occurs
   * @param pokes
   * @param expects
   */
  def event(pokes: Seq[Tuple2[Data, Int]], expects: Seq[Tuple2[Data, Int]] = Array.empty): Unit = {
    for( (port, value) <- pokes) poke(port, value)
    for( (port, value) <- expects) poke(port, value)

  }

  def finish(): Unit = {
    val input_events_counter             = Reg(init=UInt(0, log2Up(test_actions.length)))



  }
}
