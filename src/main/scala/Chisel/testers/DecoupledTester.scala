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
  var io_info : IOAccessor = null
  var num_events = 0

  case class Step(input_map: mutable.HashMap[Data,Int], output_map: mutable.HashMap[Data,Int])
  val test_actions = new ArrayBuffer[Step]()

  def poke(io_port: Data, value: Int): Unit = {
    test_actions.last.input_map(io_port) = value
  }
  def expect(io_port: Data, value: Int): Unit = {
    test_actions.last.output_map(io_port) = value
  }

  /**
   *   validate that all pokes ports are members of the same DecoupledIO
   */
  def verify_all_ports_are_same_decoupled(pokes:Seq[Tuple2[Data, Int]]) : Boolean = {
    pokes.flatMap {
      case (port, value) => io_info.find_parent_decoupled_port_name(io_info.port_to_name(port))
    }.size == 1
  }
  /**
   *   validate that all pokes ports are members of the same ValidIO
   */
  def verify_all_ports_are_same_valid(pokes:Seq[Tuple2[Data, Int]]) : Boolean = {
    pokes.flatMap {
      case (port, value) => io_info.find_parent_valid_port_name(io_info.port_to_name(port))
    }.size == 1
  }

  /**
   * create an event in which poke values will be loaded when corresponding ready
   * expect values will be validated when corresponding valid occurs
   * @param pokes
   * @param expects
   */
  def event(pokes: Seq[Tuple2[Data, Int]], expects: Seq[Tuple2[Data, Int]]): Unit = {
    test_actions += new Step(new mutable.HashMap[Data, Int], new mutable.HashMap[Data, Int])

    //TODO: validate that each event refers to 1 and only 1 decoupled/valid

    for( (port, value) <- pokes) poke(port, value)
    for( (port, value) <- expects) poke(port, value)

    num_events += 1
  }

  def finish(): Unit = {
    io_info = new IOAccessor(device_under_test.io)
    val port_to_decoupled_io = mutable.HashMap[Data, Data]()

//    def create_vectors_for_input(input_port: Data): Unit = {
//      var default_value = 0
//      val input_values = Vec(
//        test_actions.map { step =>
//          default_value = step.input_map.getOrElse(input_port, default_value)
//          UInt(default_value, input_port.width)
//        }
//      )
//    }
//
//    io_info.dut_inputs.foreach { port => create_vectors_for_input(port) }
//
//    def create_vectors_and_tests_for_output(output_port: Data): Unit = {
//      val output_values = Vec(
//        test_actions.map { step =>
//          output_port.fromBits(UInt(step.output_map.getOrElse(output_port, 0)))
//        }
//      )
//    }
//
//    io_info.dut_outputs.foreach { port => create_vectors_and_tests_for_output(port) }
  }
}
