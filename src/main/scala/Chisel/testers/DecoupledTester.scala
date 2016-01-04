package Chisel.testers

import Chisel._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.immutable

//TODO: io not allowed directly on ready or valid, this must be enforced
/**
 * Base class supports implementation of test circuits of modules
 * that use Decoupled inputs and either Decoupled or Valid outputs
 * Multiple decoupled inputs are supported.
 * Testers that subclass this will be strictly ordered.
 * Input will flow into their devices asynchronosly but in order they were generated
 * be compared in the order they are generated
 */

abstract class DecoupledTester extends BasicTester {
  def device_under_test : Module
  var io_info           : IOAccessor = null

  val input_event_list     = new ArrayBuffer[Seq[(Data, Int)]]()
  val output_event_list    = new ArrayBuffer[Seq[(Data, Int)]]()
  val parent_to_child_port = new mutable.HashMap[Data, mutable.HashSet[Data]] {
    override def default(port: Data) = new mutable.HashSet[Data]()
  }
  val port_to_decoupled    = new mutable.HashMap[Data, DecoupledIO[Data]]
  val port_to_valid        = new mutable.HashMap[Data, ValidIO[Data]]
  var num_events = 0

  case class InputStep[+T<:Data](pokes:    Map[Data,Int], parent_port: DecoupledIO[T])
  case class OutputStep(expects: Map[Data,Int], parent_port: Either[DecoupledIO[Data],ValidIO[Data]])

  val input_steps  = new ArrayBuffer[InputStep[Data]]()
  val output_steps = new ArrayBuffer[OutputStep]()

  /**
   * Validate that all pokes ports are members of the same DecoupledIO
   * makes a list of all decoupled parents based on the ports referenced in pokes
   */
  def check_and_get_common_decoupled_or_valid_parent_port_and_name(
                                                 pokes:             Seq[(Data, Int)],
                                                 must_be_decoupled: Boolean = true
                                               ) : (String, Either[DecoupledIO[Data],ValidIO[Data]]) = {
    val decoupled_parent_names = pokes.flatMap { case (port, value) =>
      io_info.find_parent_decoupled_port_name(io_info.port_to_name(port)) match {
        case None => {
          if (must_be_decoupled) {
            throw new Exception(s"Error: event $num_events port ${io_info.port_to_name(port)} not member of DecoupledIO")
            None
          }
          else {
            return get_common_valid_parent_port_and_name(pokes)
          }
        }
        case Some(parent) => {
          val decoupled_port = io_info.name_to_decoupled_port(parent)
          port_to_decoupled(port) = decoupled_port
          Some(parent)
        }
      }
    }
    if( decoupled_parent_names.toSet.size != 1 ) {
      throw new Exception(
        s"Error: event $num_events multiple DecoupledIO's referenced ${decoupled_parent_names.toSet.mkString(",")}"
      )
    }

    (decoupled_parent_names.head, Left(io_info.name_to_decoupled_port(decoupled_parent_names.head)))
  }
  /**
   * Validate that all pokes ports are members of the same DecoupledIO or ValidIO
   * makes a list of all decoupled parents based on the ports referenced in pokes
   */
  def get_common_valid_parent_port_and_name(expects: Seq[(Data, Int)]) : (String, Either[DecoupledIO[Data],ValidIO[Data]]) = {
    val valid_parent_names = expects.flatMap { case (port, value) =>
      io_info.find_parent_valid_port_name(io_info.port_to_name(port)) match {
        case None => {
          throw new Exception(s"Error: event $num_events port ${io_info.port_to_name(port)} not member of ValidIO")
          None
        }
        case Some(parent) => {
          val valid_port = io_info.name_to_valid_port(parent)
          port_to_valid(port) = valid_port
          Some(parent)
        }
      }
    }
    if( valid_parent_names.toSet.size != 1 ) {
      throw new Exception(
        s"Error: event $num_events multiple ValidIO's referenced ${valid_parent_names.toSet.mkString(",")}"
      )
    }
    (valid_parent_names.head, Right(io_info.name_to_valid_port(valid_parent_names.head)))
  }

  def input_event(pokes: Seq[(Data, Int)]): Unit = {
    input_event_list += pokes
  }
  def output_event(expects: Seq[(Data, Int)]): Unit = {
    output_event_list += expects
  }

  /**
   * iterate over recorded events, checking constraints on ports referenced, etc.
   * use poke and expect to record
   */
  def process_input_events(): Unit = {
    input_event_list.foreach { case (pokes) =>
      val (parent_name, Left(parent_port)) = check_and_get_common_decoupled_or_valid_parent_port_and_name(pokes, must_be_decoupled=true) // also validates input event

      input_steps += new InputStep(pokes.toMap, parent_port)
      parent_to_child_port(parent_port) ++= pokes.map(_._1)
      io_info.referenced_inputs ++= pokes.map(_._1)
      io_info.referenced_decoupled_ports += parent_port
    }
  }

  def process_output_events(): Unit = {
    println(s"processing ${output_event_list.size} output events")
    output_event_list.foreach { case (expects) =>
      io_info.referenced_outputs ++= expects.map(_._1)
      check_and_get_common_decoupled_or_valid_parent_port_and_name(expects, must_be_decoupled=false) match {
        case (parent_name, Left(decoupled_port)) => {
          output_steps += OutputStep(expects.toMap, Left(decoupled_port))
          parent_to_child_port(decoupled_port) ++= expects.map(_._1)
        }
        case (parent_name, Right(valid_port)) => {
          output_steps += OutputStep(expects.toMap, Right(valid_port))
          parent_to_child_port(valid_port) ++= expects.map(_._1)
        }
      }
    }
  }

  def finish(): Unit = {
    io_info = new IOAccessor(device_under_test.io)
    val port_to_decoupled_io = mutable.HashMap[Data, Data]()

    process_input_events()
    process_output_events()

    io_info.ports_referenced ++= (io_info.referenced_inputs ++ io_info.referenced_outputs)

    val ticker = Reg(init = UInt(0, width = 32 + log2Up(num_events)))
    ticker := ticker + UInt(1)
    printf("ticker %d", ticker)

    when(ticker > UInt(10)) {
      stop()
    }

    for(port <- io_info.decoupled_ports) {
      println(s"building logger for ${io_info.port_to_name(port)}")
      printf(
        s"device ${io_info.port_to_name(port)} ready %d, valid %d", port.ready, port.valid
      )
    }

    val input_event_counter  = Reg(init = UInt(0, width = log2Up(input_steps.size)))
    val output_event_counter = Reg(init = UInt(0, width = log2Up(output_steps.size)))
    val input_complete       = Reg(init = Bool(false))
    val output_complete      = Reg(init = Bool(false))

    input_complete  := input_event_counter  >= UInt(input_steps.size)
    output_complete := output_event_counter >= UInt(output_steps.size)
    when(input_complete && output_complete) {
      printf("All input and output events completed")
      stop()
    }

    /**
     * the decoupled inputs are run here
     */
    def create_vectors_for_input(input_port: Data): Unit = {
      println(
        s"bulding input loaders for ${io_info.port_to_name(input_port)}" +
        s"controlled by ${io_info.port_to_name(port_to_decoupled(input_port))}"
      )
      val decoupled_port = port_to_decoupled(input_port)
      val port_used_this_event = Vec(input_steps.map {case step => Bool(step.pokes.contains(input_port))})
      val input_values = Vec(
        input_steps.map { step =>
          val default_value = step.pokes.getOrElse(input_port, 0)
          UInt(default_value, input_port.width)
        }
      )
      when(!input_complete && decoupled_port.ready) {
        printf(s"loading ${io_info.port_to_name(input_port)} value %d input_event_counter %d",
          input_values(input_event_counter), input_event_counter)
        input_port := input_values(input_event_counter)
        decoupled_port.valid := Bool(true)
      }
    }
    io_info.referenced_inputs.foreach { port => create_vectors_for_input(port) }

    io_info.referenced_decoupled_ports.foreach { port =>
      println(s"building valid for port ${io_info.port_to_name(port)}")
      val decoupled_used_this_step = Vec(
        input_steps.map { case step =>
          println(s"Creating bool ${io_info.port_to_name(port)} ${io_info.port_to_name(step.parent_port)} " +
            s" => ${port == step.parent_port}")
          Bool(port == step.parent_port)
        }
      )
      println(s"used_this_step ${decoupled_used_this_step.map(_.asUInt()).mkString(",")}")
      when(!input_complete && decoupled_used_this_step(input_event_counter) && port.ready) {
//        port.valid := Bool(true)
        input_event_counter := input_event_counter + UInt(1)
      }
    }

    /**
     * Test values on ports moderated with a decoupled interface
     */
    def create_vectors_for_output(output_port: Data): Unit = {
      println(
        s"bulding output test for ${io_info.port_to_name(output_port)} " +
        s"controlled by ${io_info.port_to_name(port_to_decoupled(output_port))}"
      )
      val port_used_this_event = Vec(
        output_steps.map {case step => Bool(step.expects.contains(output_port))}
      )
      val output_test_values   = Vec(
        output_steps.map { step => UInt(step.expects.getOrElse(output_port, 0), width=output_port.width) }
      )

      when(!output_complete && port_used_this_event(output_event_counter) && port_to_decoupled(output_port).valid) {
        printf("output_event_counter %d", output_event_counter)
        when(output_port.asInstanceOf[UInt] != output_test_values(output_event_counter)) {
          printf(s"Error: event %d ${io_info.port_to_name(output_port)} was %x should be %x",
            output_event_counter, output_port.toBits(), output_test_values(output_event_counter))

        }
        .otherwise {
          //TODO: set ready if the port is a DecoupledIO
          output_event_counter := output_event_counter + UInt(1)
        }
      }
    }

    println(s"creating output tests [${io_info.referenced_outputs.mkString(",")}]")
    io_info.referenced_outputs.foreach { port => create_vectors_for_output(port) }
  }
}
