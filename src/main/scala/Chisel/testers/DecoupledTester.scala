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

  case class InputStep[+T<:Data](pokes:    Map[Data,Int], event_number: Int)
  case class OutputStep(expects: Map[Data,Int], event_number: Int)

  val control_port_to_input_steps  = new mutable.HashMap[DecoupledIO[Data], ArrayBuffer[InputStep[Data]]] {
    override def default(key: DecoupledIO[Data]) = new ArrayBuffer[InputStep[Data]]()
  }
  val decoupled_control_port_to_output_steps = new mutable.HashMap[DecoupledIO[Data], ArrayBuffer[InputStep[Data]]] {
    override def default(key: DecoupledIO[Data]) = new ArrayBuffer[InputStep[Data]]()
  }
  val valid_control_port_to_output_steps = new mutable.HashMap[ValidIO[Data], ArrayBuffer[InputStep[Data]]] {
    override def default(key: ValidIO[Data]) = new ArrayBuffer[InputStep[Data]]()
  }
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
    input_event_list.zipWithIndex.foreach { case (pokes, event_number) =>
      val (parent_name, Left(parent_port)) = check_and_get_common_decoupled_or_valid_parent_port_and_name(pokes, must_be_decoupled=true) // also validates input event

      control_port_to_input_steps(parent_port) += new InputStep(pokes.toMap, event_number)
      parent_to_child_port(parent_port) ++= pokes.map(_._1)
      io_info.referenced_inputs ++= pokes.map(_._1)
      io_info.referenced_decoupled_ports += parent_port
    }
  }

  def process_output_events(): Unit = {
    output_event_list.zipWithIndex.foreach { case (expects, event_number) =>
      check_and_get_common_decoupled_or_valid_parent_port_and_name(expects, must_be_decoupled = false) match {
        case (parent_name, Left(parent_port)) => {
          decoupled_control_port_to_output_steps(parent_port) += new InputStep(expects.toMap, event_number)
          parent_to_child_port(parent_port) ++= expects.map(_._1)
          io_info.referenced_outputs ++= expects.map(_._1)
          io_info.referenced_decoupled_ports += parent_port
        }
        case (parent_name, Right(parent_port)) => {
          valid_control_port_to_output_steps(parent_port) += new InputStep(expects.toMap, event_number)
          parent_to_child_port(parent_port) ++= expects.map(_._1)
          io_info.referenced_outputs ++= expects.map(_._1)
//          io_info.referenced_valid_ports += parent_port
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

    val input_event_counter  = Reg(init = UInt(0, width = log2Up(input_event_list.size)))
    val output_event_counter = Reg(init = UInt(0, width = log2Up(output_event_list.size)))
    val input_complete       = Reg(init = Bool(false))
    val output_complete      = Reg(init = Bool(false))

    input_complete  := input_event_counter  >= UInt(input_event_list.size)
    output_complete := output_event_counter >= UInt(output_event_list.size)
    when(input_complete && output_complete) {
      printf("All input and output events completed")
      stop()
    }

    /**
     * for each decoupled controller referenced, see if it is it's turn, if it is check
     * for ready, and when ready load with the values associated with that event
     */

    control_port_to_input_steps.foreach { case (controlling_port, steps) =>

      val counter_for_this_decoupled = Reg(init = UInt(0, width = log2Up(steps.length)))

      val associated_event_numbers = steps.map { step => step.event_number }.toSet
      val ports_referenced_for_this_controlling_port = new mutable.HashSet[Data]()
      steps.foreach { step =>
        step.pokes.foreach { case (port, value) => ports_referenced_for_this_controlling_port += port }
      }
      val is_this_my_turn = Vec(
        (0 until input_event_list.length).map {event_number => Bool(associated_event_numbers.contains(event_number))}
      )
      val port_vector_values = ports_referenced_for_this_controlling_port.map { port =>
        port -> Vec(steps.map { step => UInt(step.pokes.getOrElse(port, 0))})
      }.toMap

      when(!input_complete && is_this_my_turn(input_event_counter)) {
        when(controlling_port.ready) {
          ports_referenced_for_this_controlling_port.foreach { port =>
            port := port_vector_values(port)(counter_for_this_decoupled)
          }
          controlling_port.valid      := Bool(true)
          counter_for_this_decoupled := counter_for_this_decoupled + UInt(1)
          input_event_counter        := input_event_counter + UInt(1)
        }
      }
    }

    /**
     * Test values on ports moderated with a decoupled interface
     */

    decoupled_control_port_to_output_steps.foreach { case (controlling_port, steps) =>
      val counter_for_this_decoupled = Reg(init = UInt(0, width = output_event_list.size))
      val associated_event_numbers = steps.map { step => step.event_number }.toSet

      val ports_referenced_for_this_controlling_port = new mutable.HashSet[Data]()
      steps.foreach { step =>
        step.pokes.foreach { case (port, value) => ports_referenced_for_this_controlling_port += port }
      }
      val is_this_my_turn = Vec(
        (0 until output_event_list.length).map {event_number => Bool(associated_event_numbers.contains(event_number))}
      )
      val port_vector_values = ports_referenced_for_this_controlling_port.map { port =>
        port -> Vec(steps.map { step => UInt(step.pokes.getOrElse(port, 0))})
      }.toMap

      when(!output_complete && is_this_my_turn(output_event_counter)) {
        when(controlling_port.valid) {
          ports_referenced_for_this_controlling_port.foreach { port =>
            when(port.asInstanceOf[UInt] != port_vector_values(port)(output_event_counter)) {
              printf(s"Error: event %d ${io_info.port_to_name(port)} was %x should be %x",
                output_event_counter, port.toBits(), port_vector_values(port)(output_event_counter))
              assert(Bool(false))
            }
          }
          controlling_port.ready      := Bool(true)
          counter_for_this_decoupled  := counter_for_this_decoupled + UInt(1)
          output_event_counter        := output_event_counter + UInt(1)
        }
      }

    }
    /**
     * Test values on output ports moderated with a valid interface
     */

    valid_control_port_to_output_steps.foreach { case (controlling_port, steps) =>
      val counter_for_this_valid = Reg(init = UInt(0, width = output_event_list.size))
      val associated_event_numbers = steps.map { step => step.event_number }.toSet

      val ports_referenced_for_this_controlling_port = new mutable.HashSet[Data]()
      steps.foreach { step =>
        step.pokes.foreach { case (port, value) => ports_referenced_for_this_controlling_port += port }
      }
      val is_this_my_turn = Vec(
        (0 until output_event_list.length).map {event_number => Bool(associated_event_numbers.contains(event_number))}
      )
      val port_vector_values = ports_referenced_for_this_controlling_port.map { port =>
        port -> Vec(steps.map { step => UInt(step.pokes.getOrElse(port, 0))})
      }.toMap

      when(!output_complete && is_this_my_turn(output_event_counter)) {
        when(controlling_port.valid) {
          ports_referenced_for_this_controlling_port.foreach { port =>
            when(port.asInstanceOf[UInt] != port_vector_values(port)(output_event_counter)) {
              printf(s"Error: event %d ${io_info.port_to_name(port)} was %x should be %x",
                output_event_counter, port.toBits(), port_vector_values(port)(output_event_counter))
              assert(Bool(false))
            }
          }
          counter_for_this_valid      := counter_for_this_valid + UInt(1)
          output_event_counter        := output_event_counter + UInt(1)
        }
      }

    }
  }
}
