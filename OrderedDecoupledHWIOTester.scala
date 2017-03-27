// See LICENSE for license details.

package chisel3.iotesters

import chisel3._
import chisel3.util._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Base class supports implementation of test circuits of modules
  * that use Decoupled inputs and either Decoupled or Valid outputs
  * Multiple decoupled inputs are supported.
  * Testers that subclass this will be strictly ordered.
  * Input will flow into their devices asynchronously but in order they were generated
  * be compared in the order they are generated
  *
  * @example
  * {{{
  * class XTimesXTester extends [[OrderedDecoupledHWIOTester]] {
  *   val device_under_test = new XTimesY
  *   test_block {
  *     for {
  *       i <- 0 to 10
  *       j <- 0 to 10
  *     } {
  *       input_event(device_under_test.io.in.x -> i, device_under_test.in.y -> j)
  *       output_event(device_under_test.io.out.z -> i*j)
  *     }
  *   }
  * }
  * }}}
  * an input event is a series of values that will be gated into the decoupled input interface at the same time
  * an output event is a series of values that will be tested at the same time
  *
  * independent small state machines are set up for input and output interface
  * all inputs regardless of interfaces are submitted to the device under test in the order in which they were created
  * likewise,
  * all outputs regardless of which interface are tested in the same order that they were created
  */
abstract class OrderedDecoupledHWIOTester extends HWIOTester {
  val input_event_list  = new ArrayBuffer[Seq[(Data, BigInt)]]()
  val output_event_list = new ArrayBuffer[Seq[(Data, BigInt)]]()

  val port_to_decoupled = new mutable.HashMap[Data, DecoupledIO[Data]]
  val port_to_valid     = new mutable.HashMap[Data, ValidIO[Data]]

  case class TestingEvent(port_values: Map[Data, BigInt], event_number: Int)

  val control_port_to_input_values  = new mutable.HashMap[DecoupledIO[Data], ArrayBuffer[TestingEvent]] {
    override def default(key: DecoupledIO[Data]) = {
      this(key) = new ArrayBuffer[TestingEvent]()
      this(key)
    }
  }
  val decoupled_control_port_to_output_values = new mutable.HashMap[DecoupledIO[Data], ArrayBuffer[TestingEvent]] {
    override def default(key: DecoupledIO[Data]) = {
      this(key) = new ArrayBuffer[TestingEvent]()
      this(key)
    }
  }
  val valid_control_port_to_output_values = new mutable.HashMap[ValidIO[Data], ArrayBuffer[TestingEvent]] {
    override def default(key: ValidIO[Data]) = {
      this(key) = new ArrayBuffer[TestingEvent]()
      this(key)
    }
  }

  /**
    * Validate that all pokes ports are members of the same DecoupledIO
    * makes a list of all decoupled parents based on the ports referenced in pokes
    */
  def checkAndGetCommonDecoupledOrValidParentPort(
                                                   pokes:             Seq[(Data, BigInt)],
                                                   must_be_decoupled: Boolean = true,
                                                   event_number:      Int
                                                 ) : Either[DecoupledIO[Data],ValidIO[Data]] = {
    val decoupled_parent_names = pokes.flatMap { case (port, value) =>
      Predef.assert(
        !io_info.port_to_name(port).endsWith(".valid"),
        s"Error: port ${io_info.port_to_name(port)}. input_event and output_event cannot directly reference valid"
      )
      Predef.assert(
        !io_info.port_to_name(port).endsWith(".ready"),
        s"Error: port ${io_info.port_to_name(port)}, input_event and output_event cannot directly reference ready"
      )

      io_info.findParentDecoupledPortName(io_info.port_to_name(port)) match {
        case None =>
          if (must_be_decoupled) {
            throw new Exception(
              s"Error: event $event_number port ${io_info.port_to_name(port)} not member of DecoupledIO"
            )
            None
          }
          else {
            return getCommonValidParentPort(pokes, event_number)
          }
        case Some(parent) =>
          val decoupled_port = io_info.name_to_decoupled_port(parent)
          port_to_decoupled(port) = decoupled_port
          Some(parent)
      }
    }
    if( decoupled_parent_names.toSet.size != 1 ) {
      throw new Exception(
        s"Error: event $event_number multiple DecoupledIO's referenced " +
          decoupled_parent_names.toSet.mkString(",")
      )
    }

    Left(io_info.name_to_decoupled_port(decoupled_parent_names.head))
  }
  /**
    * Validate that all pokes ports are members of the same DecoupledIO or ValidIO
    * makes a list of all decoupled parents based on the ports referenced in pokes
    */
  def getCommonValidParentPort(
                                expects: Seq[(Data, BigInt)],
                                event_number: Int
                              ): Either[DecoupledIO[Data], ValidIO[Data]] = {
    val valid_parent_names = expects.flatMap { case (port, value) =>
      io_info.findParentValidPortName(io_info.port_to_name(port)) match {
        case None =>
          throw new Exception(s"Error: event $event_number " +
            s"port ${io_info.port_to_name(port)} not member of ValidIO")
          None
        case Some(parent) =>
          val valid_port = io_info.name_to_valid_port(parent)
          port_to_valid(port) = valid_port
          Some(parent)
      }
    }
    if (valid_parent_names.toSet.size != 1) {
      throw new Exception(
        s"Error: event $event_number multiple ValidIO's referenced ${valid_parent_names.toSet.mkString(",")}"
      )
    }
    Right(io_info.name_to_valid_port(valid_parent_names.head))
  }

  def inputEvent(pokes: (Data, BigInt)*): Unit = {
    input_event_list += pokes
  }

  def outputEvent(expects: (Data, BigInt)*): Unit = {
    output_event_list += expects
  }

  /**
    * iterate over recorded events, checking constraints on ports referenced, etc.
    * use poke and expect to record
    */
  def processInputEvents(): Unit = {
    input_event_list.zipWithIndex.foreach { case (pokes, event_number) =>
      val Left(parent_port) = checkAndGetCommonDecoupledOrValidParentPort(
        pokes, must_be_decoupled = true, event_number
      )

      control_port_to_input_values(parent_port) += new TestingEvent(pokes.toMap, event_number)
      io_info.ports_referenced ++= pokes.map(_._1)
    }
    logScalaDebug(
      s"Processing input events done, referenced controlling ports " +
        control_port_to_input_values.keys.map { p => io_info.port_to_name(p) }.mkString(",")
    )
  }

  def processOutputEvents(): Unit = {
    output_event_list.zipWithIndex.foreach { case (expects, event_number) =>
      checkAndGetCommonDecoupledOrValidParentPort(
        expects,
        must_be_decoupled = false,
        event_number = event_number
      ) match {
        case Left(parent_port) =>
          decoupled_control_port_to_output_values(parent_port) += new TestingEvent(expects.toMap, event_number)
          io_info.ports_referenced ++= expects.map(_._1)

        case Right(parent_port) =>
          valid_control_port_to_output_values(parent_port) += new TestingEvent(expects.toMap, event_number)
          io_info.ports_referenced ++= expects.map(_._1)

      }
    }
    logScalaDebug(
      s"Processing output events done, referenced controlling ports" +
        (
          if (decoupled_control_port_to_output_values.nonEmpty) {
            decoupled_control_port_to_output_values.keys.map {
              p => io_info.port_to_name(p)
            }.mkString(", decoupled : ", ",", "")
          }
          else {
            ""
          }
          ) +
        (
          if (valid_control_port_to_output_values.nonEmpty) {
            valid_control_port_to_output_values.keys.map {
              p => io_info.port_to_name(p)
            }.mkString(", valid : ", ",", "")
          }
          else {
            ""
          }
          )
    )
  }

  private def name(port: Data): String = io_info.port_to_name(port)

  /**
    * creates a Vec of Booleans that indicate if the io interface in question
    * is operational at particular io event_number
    *
    * @param events is a list of events and their associated event numbers
    * @return
    */
  private def createIsMyTurnTable(events: ArrayBuffer[TestingEvent]): Vec[Bool] = {
    val associated_event_numbers = events.map { event => event.event_number }.toSet
    logScalaDebug(s"  associated event numbers ${associated_event_numbers.toArray.sorted.mkString(",")}")

    Vec(
      input_event_list.indices.map { event_number => (associated_event_numbers.contains(event_number)).asBool } ++
        List(false.B) // We append a false at the end so no-one tries to go when counter done
    )
  }

  /**
    * build a set of all ports referenced by all events associated with a particular
    * io interface
 *
    * @param events  a set of events
    * @return
    */
  private def portsReferencedByEvents(events: ArrayBuffer[TestingEvent]): mutable.HashSet[Data] = {
    val ports_referenced = new mutable.HashSet[Data]()
    events.foreach { event =>
      event.port_values.foreach { case (port, value) => ports_referenced += port }
    }
    ports_referenced
  }

  private def buildValuesVectorForEachPort(
                                            io_interface     : Data,
                                            referenced_ports : mutable.HashSet[Data],
                                            events           : ArrayBuffer[TestingEvent]
                                          ): Map[Data, Vec[UInt]] = {
    val port_vector_events = referenced_ports.map { port =>
      port -> Vec(events.map { event => (event.port_values.getOrElse(port, BigInt(0))).asUInt } ++ List(0.U)) //0 added to end
    }.toMap

    logScalaDebug(s"Input controller ${io_info.port_to_name(io_interface)} : ports " +
      s" ${referenced_ports.map { port => name(port) }.mkString(",")}")
    port_vector_events
  }
  /**
    * for each input event only one controller is active (determined by it's private is_my_turn vector)
    * each controller has a private counter indicating which event specific to that controller
    * is on deck.  those values are wired to the inputs of the decoupled input and the valid is asserted
    * IMPORTANT NOTE: the lists generated here has an extra 0 element added to the end because the counter
    * used will stop at a value one higher than the number of test elements
    */
  private def buildInputEventHandlers(event_counter: GlobalEventCounter) {
    control_port_to_input_values.foreach { case (controlling_port, events) =>
      val ports_referenced_for_this_controlling_port = portsReferencedByEvents(events)
      val is_this_my_turn = createIsMyTurnTable(events)

      val counter_for_this_decoupled = Counter(events.length)

      val port_vector_events = buildValuesVectorForEachPort(
        controlling_port,
        ports_referenced_for_this_controlling_port,
        events
      )

      logScalaDebug(s"Input controller ${io_info.port_to_name(controlling_port)} : ports " +
        s" ${ports_referenced_for_this_controlling_port.map { port => name(port) }.mkString(",")}")

      ports_referenced_for_this_controlling_port.foreach { port =>
        port := port_vector_events(port)(counter_for_this_decoupled.value)
      }
      controlling_port.valid := is_this_my_turn(event_counter.value)

      when(controlling_port.valid && controlling_port.ready) {
        counter_for_this_decoupled.inc()
        event_counter.inc()
      }
    }
  }

  /**
    * Test values on ports moderated with a decoupled interface
    * IMPORTANT NOTE: the lists generated here has an extra 0 element added to the end because the counter
    * used will stop at a value one higher than the number of test elements
    */
  private def buildDecoupledOutputEventHandlers(event_counter: GlobalEventCounter) {
    decoupled_control_port_to_output_values.foreach { case (controlling_port, events) =>
      val ports_referenced_for_this_controlling_port = portsReferencedByEvents(events)
      val is_this_my_turn = createIsMyTurnTable(events)

      val counter_for_this_decoupled = Counter(output_event_list.length)
      logScalaDebug(s"Output decoupled controller ${name(controlling_port)} : ports " +
        s" ${ports_referenced_for_this_controlling_port.map { port => name(port) }.mkString(",")}")

      val port_vector_events = buildValuesVectorForEachPort(
        controlling_port,
        ports_referenced_for_this_controlling_port,
        events
      )

      controlling_port.ready := is_this_my_turn(event_counter.value)

      when(controlling_port.ready && controlling_port.valid) {
        ports_referenced_for_this_controlling_port.foreach { port =>
          printf(s"output test event %d testing ${name(port)} = %d, should be %d\n",
            event_counter.value, port.asInstanceOf[UInt], port_vector_events(port)(counter_for_this_decoupled.value)
          )
          when(port.asInstanceOf[UInt] != port_vector_events(port)(counter_for_this_decoupled.value)) {
            printf(s"Error: event %d ${name(port)} was %d should be %d\n",
              event_counter.value, port.asUInt, port_vector_events(port)(counter_for_this_decoupled.value))
            assert(false.B)
            stop()
          }
        }
        counter_for_this_decoupled.inc()
        event_counter.inc()
      }
    }
  }

  /**
    * Test events on output ports moderated with a valid interface
    * IMPORTANT NOTE: the lists generated here has an extra 0 element added to the end because the counter
    * used will stop at a value one higher than the number of test elements
    */
  private def buildValidIoPortEventHandlers(event_counter: GlobalEventCounter) {
    valid_control_port_to_output_values.foreach { case (controlling_port, events) =>
      val ports_referenced_for_this_controlling_port = portsReferencedByEvents(events)
      val is_this_my_turn = createIsMyTurnTable(events)

      val counter_for_this_valid = Counter(output_event_list.length)
      logScalaDebug(s"Output decoupled controller ${name(controlling_port)} : ports " +
        s" ${ports_referenced_for_this_controlling_port.map { port => name(port) }.mkString(",")}")

      val port_vector_events = buildValuesVectorForEachPort(
        controlling_port,
        ports_referenced_for_this_controlling_port,
        events
      )

      when(is_this_my_turn(event_counter.value)) {
        when(controlling_port.valid) {
          ports_referenced_for_this_controlling_port.foreach { port =>
            printf(s"output test event %d testing ${name(port)} = %d, should be %d",
              event_counter.value, port.asInstanceOf[UInt], port_vector_events(port)(counter_for_this_valid.value)
            )
            when(port.asInstanceOf[UInt] =/= port_vector_events(port)(counter_for_this_valid.value)) {
              printf(s"Error: event %d ${name(port)} was %x should be %x",
                event_counter.value, port.asUInt, port_vector_events(port)(counter_for_this_valid.value))
              assert(false.B)
            }
          }
          counter_for_this_valid.inc()
          event_counter.inc()
        }
      }
    }
  }

  class GlobalEventCounter(val max_count: Int) {
    val counter     = RegInit(0.U((log2Ceil(max_count) + 2).W))
    val reached_end = RegInit(false.B)

    def value: UInt = counter

    def inc(): Unit = {
      when(! reached_end ) {
        when(counter === (max_count-1).asUInt) {
          reached_end := true.B
        }
        counter := counter + 1.U
      }
    }
  }

  /**
   * this builds a circuit to load inputs and circuits to test outputs that are controlled
   * by either a decoupled or valid
   */
  override def finish(): Unit = {
    io_info = new IOAccessor(device_under_test.io)

    processInputEvents()
    processOutputEvents()

    val input_event_counter  = new GlobalEventCounter(input_event_list.length)
    val output_event_counter = new GlobalEventCounter(output_event_list.length)

    when(input_event_counter.reached_end && output_event_counter.reached_end) {
      printf("All input and output events completed\n")
      stop()
    }

    val ti = RegInit(0.U(log2Ceil(OrderedDecoupledHWIOTester.max_tick_count).W))
    ti := ti + 1.U
    when(ti > (OrderedDecoupledHWIOTester.max_tick_count).asUInt) {
      printf(
        "Exceeded maximum allowed %d ticks in OrderedDecoupledHWIOTester, If you think code is correct use:\n" +
        "DecoupleTester.max_tick_count = <some-higher-value>\n" +
        "in the OrderedDecoupledHWIOTester subclass\n",
        (OrderedDecoupledHWIOTester.max_tick_count).asUInt
      )
      stop()
    }

    buildInputEventHandlers(input_event_counter)
    buildDecoupledOutputEventHandlers(output_event_counter)
    buildValidIoPortEventHandlers(output_event_counter)

    logPrintfDebug(s"in_event_counter %d, out_event_counter %d\n",
      input_event_counter.value, output_event_counter.value)
    if(enable_scala_debug || enable_all_debug) {
      io_info.showPorts("".r)
    }
  }
}

object OrderedDecoupledHWIOTester {
  val default_max_tick_count = 10000
  var max_tick_count         = default_max_tick_count
}


