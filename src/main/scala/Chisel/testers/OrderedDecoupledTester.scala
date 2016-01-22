// See LICENSE for license details.

package Chisel.testers

import Chisel._

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
  * class XTimesXTester extends [[OrderedDecoupledTester]] {
  *   val device_under_test = new XTimesY
  *   test_block { () =>
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
  */
abstract class OrderedDecoupledTester extends BasicTester {
  val device_under_test     : Module
  var io_info               : IOAccessor = null

  def testBlock(block: () => Unit): Unit = {
    block()
    finish()
  }

  var verbose           = false
  def logScala(msg: => String): Unit = {
    //noinspection ScalaStyle
    if(verbose) println(msg)
  }
  def logSimulator(fmt: String, args: Bits*): Unit = {
    if(verbose) printf(fmt, args :_*)
  }

  val input_event_list  = new ArrayBuffer[Seq[(Data, Int)]]()
  val output_event_list = new ArrayBuffer[Seq[(Data, Int)]]()

  val port_to_decoupled = new mutable.HashMap[Data, DecoupledIO[Data]]
  val port_to_valid     = new mutable.HashMap[Data, ValidIO[Data]]

  //
  case class InputStep(pokes: Map[Data,Int], event_number: Int)
  case class OutputStep(expects: Map[Data,Int], event_number: Int)

  val control_port_to_input_steps  = new mutable.HashMap[DecoupledIO[Data], ArrayBuffer[InputStep]] {
    override def default(key: DecoupledIO[Data]) = {
      this(key) = new ArrayBuffer[InputStep]()
      this(key)
    }
  }
  val decoupled_control_port_to_output_steps = new mutable.HashMap[DecoupledIO[Data], ArrayBuffer[OutputStep]] {
    override def default(key: DecoupledIO[Data]) = {
      this(key) = new ArrayBuffer[OutputStep]()
      this(key)
    }
  }
  val valid_control_port_to_output_steps = new mutable.HashMap[ValidIO[Data], ArrayBuffer[OutputStep]] {
    override def default(key: ValidIO[Data]) = {
      this(key) = new ArrayBuffer[OutputStep]()
      this(key)
    }
  }

  /**
   * Validate that all pokes ports are members of the same DecoupledIO
   * makes a list of all decoupled parents based on the ports referenced in pokes
   */
  def checkAndGetCommonDecoupledOrValidParentPort(
                                                   pokes:             Seq[(Data, Int)],
                                                   must_be_decoupled: Boolean = true,
                                                   event_number:      Int
                                                 ) : Either[DecoupledIO[Data],ValidIO[Data]] = {
    val decoupled_parent_names = pokes.flatMap { case (port, value) =>
      Predef.assert(
        ! io_info.port_to_name(port).endsWith(".valid"),
        s"Error: port ${io_info.port_to_name(port)}. input_event and output_event cannot directly reference valid"
      )
      Predef.assert(
        ! io_info.port_to_name(port).endsWith(".ready"),
        s"Error: port ${io_info.port_to_name(port)}, input_event and output_event cannot directly reference ready"
      )

      io_info.find_parent_decoupled_port_name(io_info.port_to_name(port)) match {
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
                                    expects: Seq[(Data, Int)],
                                    event_number: Int
                                  ) : Either[DecoupledIO[Data],ValidIO[Data]] = {
    val valid_parent_names = expects.flatMap { case (port, value) =>
      io_info.find_parent_valid_port_name(io_info.port_to_name(port)) match {
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
    if( valid_parent_names.toSet.size != 1 ) {
      throw new Exception(
        s"Error: event $event_number multiple ValidIO's referenced ${valid_parent_names.toSet.mkString(",")}"
      )
    }
    Right(io_info.name_to_valid_port(valid_parent_names.head))
  }

  def inputEvent(pokes: (Data, Int)*): Unit = {
    input_event_list += pokes
  }
  def outputEvent(expects: (Data, Int)*): Unit = {
    output_event_list += expects
  }

  /**
   * iterate over recorded events, checking constraints on ports referenced, etc.
   * use poke and expect to record
   */
  def processInputEvents(): Unit = {
    input_event_list.zipWithIndex.foreach { case (pokes, event_number) =>
      val Left(parent_port) = checkAndGetCommonDecoupledOrValidParentPort(
        pokes, must_be_decoupled=true, event_number
      )

      control_port_to_input_steps(parent_port) += new InputStep(pokes.toMap, event_number)
      io_info.referenced_inputs ++= pokes.map(_._1)
      io_info.ports_referenced ++= pokes.map(_._1)
    }
    logScala(
      s"Processing input events done, referenced controlling ports " +
      control_port_to_input_steps.keys.map{p => io_info.port_to_name(p)}.mkString(",")
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
          decoupled_control_port_to_output_steps(parent_port) += new OutputStep(expects.toMap, event_number)
          io_info.referenced_outputs ++= expects.map(_._1)
          io_info.ports_referenced ++= expects.map(_._1)

        case Right(parent_port) =>
          valid_control_port_to_output_steps(parent_port) += new OutputStep(expects.toMap, event_number)
          io_info.referenced_outputs ++= expects.map(_._1)
          io_info.ports_referenced ++= expects.map(_._1)

      }
    }
    logScala(
      s"Processing output events done, referenced controlling ports" +
        (
          if(decoupled_control_port_to_output_steps.nonEmpty) {
            decoupled_control_port_to_output_steps.keys.map {
              p => io_info.port_to_name(p)
            }.mkString(", decoupled : ", ",", "")
          }
          else {
            ""
          }
        ) +
        (
          if(valid_control_port_to_output_steps.nonEmpty) {
            valid_control_port_to_output_steps.keys.map {
              p => io_info.port_to_name(p)
            }.mkString(", valid : ", ",", "")
          }
          else {
            ""
          }
        )
    )
  }

  def name(port: Data): String = io_info.port_to_name(port)

  /**
    * for each input event only one controller is active (determined by it's private is_my_turn vector)
    * each controller has a private counter indicating which event specific to that controller
    * is on deck.  those values are wired to the inputs of the decoupled input and the valid is asserted
    */
  // TODO: look into using mapIndex instead of all the different vec's here
  private def buildInputEventHandlers(event_counter: GlobalEventCounter) {
    control_port_to_input_steps.foreach { case (controlling_port, steps) =>
      val counter_for_this_decoupled = Counter(steps.length)

      val associated_event_numbers = steps.map { step => step.event_number }.toSet
      val ports_referenced_for_this_controlling_port = new mutable.HashSet[Data]()
      steps.foreach { step =>
        step.pokes.foreach { case (port, value) => ports_referenced_for_this_controlling_port += port }
      }
      val is_this_my_turn = Vec(
        input_event_list.indices.map { event_number => Bool(associated_event_numbers.contains(event_number)) } ++
          List(Bool(false))  // We append a false at the end so no-one tries to go when counter done
      )
      val port_vector_values = ports_referenced_for_this_controlling_port.map { port =>
        port -> Vec(steps.map { step => UInt(step.pokes.getOrElse(port, 0)) } ++ List(UInt(0))) //0 added to end
      }.toMap

      logScala(s"Input controller ${io_info.port_to_name(controlling_port)} : ports " +
        s" ${ports_referenced_for_this_controlling_port.map { port => name(port) }.mkString(",")}")
      logScala(s"  associated event numbers ${associated_event_numbers.toArray.sorted.mkString(",")}")

      ports_referenced_for_this_controlling_port.foreach { port =>
        port := port_vector_values(port)(counter_for_this_decoupled.value)
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
    */
  private def buildDecoupledOutputEventHandlers(event_counter: GlobalEventCounter) {
    decoupled_control_port_to_output_steps.foreach { case (controlling_port, steps) =>
      val counter_for_this_decoupled = Counter(output_event_list.length)
      val associated_event_numbers = steps.map { step => step.event_number }.toSet
      val ports_referenced_for_this_controlling_port = new mutable.HashSet[Data]()
      steps.foreach { step =>
        step.expects.foreach { case (port, value) => ports_referenced_for_this_controlling_port += port }
      }
      val is_this_my_turn = Vec(
        output_event_list.indices.map { event_number => Bool(associated_event_numbers.contains(event_number)) } ++
        List(Bool(false)) // We append a false at the end so no-one tries to go when counter done
      )
      logScala(s"Output decoupled controller ${name(controlling_port)} : ports " +
        s" ${ports_referenced_for_this_controlling_port.map { port => name(port) }.mkString(",")}")
      logScala(s"  associated event numbers ${associated_event_numbers.toArray.sorted.mkString(",")}")

      val port_vector_values = ports_referenced_for_this_controlling_port.map { port =>
        val values_vector = steps.map { step => step.expects.getOrElse(port, 0) }
        logScala(s"  output vector generation for ${name(port)} : ${values_vector.mkString(",")}")
        port -> Vec(values_vector.map { value => UInt(value) } ++ List(UInt(0))) // 0 added to end of list
      }.toMap

      controlling_port.ready := is_this_my_turn(event_counter.value)

      when(controlling_port.ready && controlling_port.valid) {
        ports_referenced_for_this_controlling_port.foreach { port =>
          printf(s"output test event %d testing ${name(port)} = %d, should be %d",
            event_counter.value, port.asInstanceOf[UInt], port_vector_values(port)(counter_for_this_decoupled.value)
          )
          when(port.asInstanceOf[UInt] != port_vector_values(port)(counter_for_this_decoupled.value)) {
            printf(s"Error: event %d ${name(port)} was %d should be %d",
              event_counter.value, port.toBits(), port_vector_values(port)(counter_for_this_decoupled.value))
            assert(Bool(false))
            stop()
          }
        }
        counter_for_this_decoupled.inc()
        event_counter.inc()
      }
    }
  }

  /**
    * Test values on output ports moderated with a valid interface
    */
  private def buildValidIoPortEventHandlers(event_counter: GlobalEventCounter) {
    valid_control_port_to_output_steps.foreach { case (controlling_port, steps) =>
      val counter_for_this_valid = Counter(event_counter.max_count)
      val associated_event_numbers = steps.map { step => step.event_number }.toSet

      val ports_referenced_for_this_controlling_port = new mutable.HashSet[Data]()
      steps.foreach { step =>
        step.expects.foreach { case (port, value) => ports_referenced_for_this_controlling_port += port }
      }
      val is_this_my_turn = Vec(
        output_event_list.indices.map { event_number => Bool(associated_event_numbers.contains(event_number)) } ++
        List(Bool(false))  // We append a false at the end so no-one tries to go when counter done
      )
      val port_vector_values = ports_referenced_for_this_controlling_port.map { port =>
        port -> Vec(steps.map { step => UInt(step.expects.getOrElse(port, 0)) })
      }.toMap

      // finish
      when(is_this_my_turn(event_counter.value)) {
        when(controlling_port.valid) {
          ports_referenced_for_this_controlling_port.foreach { port =>
            printf(s"output test event %d testing ${name(port)} = %d, should be %d",
              event_counter.value, port.asInstanceOf[UInt], port_vector_values(port)(counter_for_this_valid.value)
            )
            when(port.asInstanceOf[UInt] != port_vector_values(port)(counter_for_this_valid.value)) {
              printf(s"Error: event %d ${name(port)} was %x should be %x",
                event_counter.value, port.toBits(), port_vector_values(port)(counter_for_this_valid.value))
              assert(Bool(false))
            }
          }
          counter_for_this_valid.inc()
          event_counter.inc()
        }
      }
    }
  }

  class GlobalEventCounter(val max_count: Int) {
    val counter     = Reg(init = UInt(0, width = log2Up(max_count) + 2))
    val reached_end = Reg(init = Bool(false))

    def value: UInt = counter

    def inc(): Unit = {
      when(! reached_end ) {
        when(counter === UInt(max_count-1)) {
          reached_end := Bool(true)
        }
        counter := counter + UInt(1)
      }
    }
  }

  /**
   * this builds a circuit to load inputs and circuits to test outputs that are controlled
   * by either a decoupled or valid
   */
  def finish(show_io_table: Boolean = false): Unit = {
    io_info = new IOAccessor(device_under_test.io)

    processInputEvents()
    processOutputEvents()

    val input_event_counter  = new GlobalEventCounter(input_event_list.length)
    val output_event_counter = new GlobalEventCounter(output_event_list.length)

    when(input_event_counter.reached_end && output_event_counter.reached_end) {
      printf("All input and output events completed")
      stop()
    }


    val ti = Reg(init= UInt(0, width = log2Up(DecoupledTester.max_tick_count)))
    ti := ti + UInt(1)
    when(ti > UInt(DecoupledTester.max_tick_count)) {
      printf(
        "Exceeded maxiumum allowed %d ticks in DecoupledTester, If you think code is correct use:\n" +
        "DecoupleTester.max_tick_count = <some-higher-value>\n" +
        "in the DecoupledTester subclass",
        UInt(DecoupledTester.max_tick_count)
      )
      stop()
    }

    buildInputEventHandlers(input_event_counter)
    buildDecoupledOutputEventHandlers(output_event_counter)
    buildValidIoPortEventHandlers(output_event_counter)

    logSimulator(s"in_event_counter %d, out_event_counter %d", input_event_counter.value, output_event_counter.value)
    io_info.show_ports("".r)
  }
}


