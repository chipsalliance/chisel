package Chisel.testers

import Chisel._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

//TODO: io not allowed directly on ready or valid
/**
 * Base class supports implementation of engines that test circuits whose use Decoupled IO
 */
abstract class DecoupledTester extends BasicTester {
  def device_under_test : Module
  var io_info : IOAccessor = null
  val event_list = new ArrayBuffer[Tuple2[Seq[Tuple2[Data, Int]], Seq[Tuple2[Data, Int]]]]()
  var num_events = 0

  class Step(
              val input_map: mutable.HashMap[Data,Int],
              val output_map: mutable.HashMap[Data,Int],
              val decoupled_parent_index : Int,
              val valid_parent_index     : Int
              )

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
  def get_decoupled_parent(pokes:Seq[Tuple2[Data, Int]]) : Int = {
    val decoupled_parent_names = pokes.flatMap { case (port, value) =>
      io_info.find_parent_decoupled_port_name(io_info.port_to_name(port)) match {
        case None => {
          throw new Exception(s"Error: event $num_events port ${io_info.port_to_name(port)} not member of DecoupledIO")
          None
        }
        case parent => parent
      }
    }
    if( decoupled_parent_names.toSet.size != 1 ) {
      throw new Exception(
        s"Error: event $num_events multiple DecoupledIO's referenced ${decoupled_parent_names.toSet.mkString(",")}"
      )
    }
    io_info.decoupled_ports.indexOf(io_info.name_to_decoupled_port(decoupled_parent_names.head))
  }
  /**
   *   validate that all pokes ports are members of the same ValidIO
   */
  def get_valid_parent(pokes:Seq[Tuple2[Data, Int]]) : Int = {
    var found_error = false
    val valid_parent_names = pokes.flatMap { case (port, value) =>
      io_info.find_parent_valid_port_name(io_info.port_to_name(port)) match {
        case None => {
          throw new Exception(s"Error: event $num_events port ${io_info.port_to_name(port)} not member of ValidIO")
          None
        }
        case parent => parent
      }
    }
    if( valid_parent_names.toSet.size != 1 ) {
      throw new Exception(
        s"Error: event $num_events multiple ValidIO's referenced ${valid_parent_names.toSet.mkString(",")}"
      )
    }
    io_info.valid_ports.indexOf(io_info.name_to_valid_port(valid_parent_names.head))
  }

  /**
   * create an event in which poke values will be loaded when corresponding ready
   * expect values will be validated when corresponding valid occurs
   * @param pokes
   * @param expects
   */
  def event(pokes: Seq[Tuple2[Data, Int]], expects: Seq[Tuple2[Data, Int]]): Unit = {
    event_list += ((pokes, expects))
  }

  def process_events(): Unit = {
    for( (pokes, expects) <- event_list) {
      val decoupled_io_parent = get_decoupled_parent(pokes)
      val valid_io_parent     = get_decoupled_parent(expects)

      test_actions += new Step(
        new mutable.HashMap[Data, Int], new mutable.HashMap[Data, Int],
        decoupled_io_parent, valid_io_parent)

      for( (port, value) <- pokes) poke(port, value)
      for( (port, value) <- expects) poke(port, value)

      io_info.ports_referenced ++= (pokes ++ expects).map { case (port, value) => port}

      num_events += 1
    }
  }

  def finish(): Unit = {
    io_info = new IOAccessor(device_under_test.io)
    val port_to_decoupled_io = mutable.HashMap[Data, Data]()

    process_events()

    val event_selector = Vec(
      test_actions.map { case (step) => UInt(step.decoupled_parent_index, width = 16) }
    )
    val decoupled_vec = Vec(
      io_info.decoupled_ports
    )

    val ticker = Reg(init = UInt(0, width = 32 + log2Up(num_events)))
    ticker := ticker + UInt(1)
    printf("ticker %d", ticker)

    when(ticker > UInt(100)) {
      stop()
    }

    for(port <- io_info.decoupled_ports) {
      println(s"building logger for ${io_info.port_to_name(port)}")
      printf(
        s"device ${io_info.port_to_name(port)} ready %d, valid %d", port.ready, port.valid
      )
    }

    val input_complete  = Reg(init = Bool(false))
    val output_complete = Reg(init = Bool(false))

    when(input_complete && output_complete) {
      stop()
    }

    val input_event_counter  = Reg(init = UInt(0, width = log2Up(num_events)))
    val output_event_counter = Reg(init = UInt(0, width = log2Up(num_events)))

    /**
     * the decoupled inputs are run here
     */

    def create_vectors_for_input(input_port: Data): Unit = {
      val input_values = Vec(
        test_actions.map { step =>
          val default_value = step.input_map.getOrElse(input_port, 0)
          UInt(default_value, input_port.width)
        }
      )
      when(!input_complete && decoupled_vec(event_selector(input_event_counter)).ready) {
        printf("input_event_counter %d", input_event_counter)
        input_port := input_values(input_event_counter)
      }
    }
    when(!input_complete && decoupled_vec(event_selector(input_event_counter)).ready) {
      decoupled_vec(event_selector(input_event_counter)).valid := Bool(true)
      input_event_counter := input_event_counter + UInt(1)
      input_complete := input_event_counter >= UInt(num_events)
    }
    io_info.referenced_inputs.foreach { port => create_vectors_for_input(port) }


    /**
     * figure out how to do the ready valid
     * @param output_port
     */
    def create_vectors_for_output(output_port: Data): Unit = {
      val output_values = Vec(
        test_actions.map { step =>
          val default_value = step.output_map.getOrElse(output_port, 0)
          UInt(default_value, output_port.width)
        }
      )
      when(!output_complete && decoupled_vec(event_selector(output_event_counter)).valid) {
        when(output_port.asInstanceOf[UInt] === output_values(output_event_counter)) {
          printf(s"Error: event %d ${io_info.port_to_name(output_port)} was %x should be %x",
            output_event_counter, output_port.toBits(), output_values(output_event_counter))
        }
      }
    }

    when(!output_complete && decoupled_vec(event_selector(output_event_counter)).valid) {
      decoupled_vec(event_selector(output_event_counter)).ready := Bool(true)
      output_event_counter := output_event_counter + UInt(1)
      output_complete := output_event_counter >= UInt(num_events)
    }

    io_info.referenced_outputs.foreach { port => create_vectors_for_output(port) }
  }
}
