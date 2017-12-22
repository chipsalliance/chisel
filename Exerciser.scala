// See LICENSE for license details.

package chisel3.iotesters

import chisel3._
import chisel3.testers.BasicTester

/**
  * experimental version of a Tester that allows arbitrary testing circuitry to be run
  * in some order
  */
abstract class Exerciser extends BasicTester {
  val device_under_test: Module

  val internal_counter_width = 32
  val max_ticker = 100

  case class StopCondition(condition: Bool, max_ticks: Option[Int] = None)

  val ticker              = RegInit(0.U(internal_counter_width.W))
  val max_ticks_for_state = RegInit(0.U(internal_counter_width.W))
  val state_number        = RegInit(0.U(internal_counter_width.W))
  val state_locked        = RegInit(true.B)

  var current_states = internal_counter_width + 1

  ticker := ticker + 1.U
  when(!state_locked) {
    ticker       := 0.U
    state_number := state_number + 1.U
  }
  .elsewhen(ticker > max_ticks_for_state) {
    printf("current state %d has run too many cycles, ticks %d max %d",
      state_number, ticker, max_ticks_for_state)
    state_locked := false.B
    state_number := state_number + 1.U
  }
  when(ticker > (max_ticker).asUInt) {
    printf("Too many cycles ticker %d current_state %d state_locked %x",
          ticker, state_number, state_locked)
    stop()
  }

  override def finish() {
    when(state_number > (current_states).asUInt) {
      printf("All states processed")
      stop()
    }
  }
  def buildState(name: String = s"$current_states")(stop_condition : StopCondition)(generator: () => Unit): Unit = {
    //noinspection ScalaStyle
    device_under_test.io := DontCare  // Support invalidate API.
    println(s"building state $current_states $name")
    when(state_number === (current_states).asUInt) {
      when(! state_locked) {
        printf(s"Entering state $name state_number %d ticker %d", state_number, ticker)
        state_locked        := true.B
        ticker              := 0.U
        max_ticks_for_state := (stop_condition.max_ticks.getOrElse(max_ticker)).asUInt
        state_number := (current_states).asUInt
      }
      generator()

      when(stop_condition.condition) {
        printf(s"Leaving state  $name state_number %d ticker %d", state_number, ticker)
        state_locked := false.B
        state_number := state_number + 1.U
      }
    }
    current_states += 1
  }
}
