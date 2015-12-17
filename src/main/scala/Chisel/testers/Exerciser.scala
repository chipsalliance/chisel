package Chisel.testers

import Chisel._

/**
 * Created by chick on 12/16/15.
 */
abstract class Exerciser extends BasicTester {
  def device_under_test: Module

  case class StopCondition(condition: Bool, max_ticks: Option[Int] = None)

  val ticker              = Reg(init=UInt(0, width=32))
  val max_ticks_for_state = Reg(init=UInt(0, width=32))
  val state_number        = Reg(init=UInt(0, width=32))
  val state_locked        = Reg(init=Bool(true))

  var current_states = 33

  ticker := ticker + UInt(1)
  when(!state_locked) {
    ticker       := UInt(0)
    state_number := state_number + UInt(1)
  }
  .elsewhen(ticker > max_ticks_for_state) {
    printf("current state %d has run too many cycles, ticks %d max %d",
      state_number, ticker, max_ticks_for_state)
    state_locked := Bool(false)
    state_number := state_number + UInt(1)
  }
  when(ticker > UInt(100)) {
    printf("Too many cycles ticker %d current_state %d state_locked %x",
          ticker, state_number, state_locked)
    stop()
  }

  def finish() {
    when(state_number > UInt(current_states)) {
      printf("All states processed")
      stop()
    }
  }
  def build_state(name: String = s"$current_states")(stop_condition : StopCondition)(generator: () => Unit): Unit = {
    println(s"building state $current_states $name")
    when(state_number === UInt(current_states)) {
      when(! state_locked) {
        printf(s"Entering state $name state_number %d ticker %d", state_number, ticker)
        state_locked        := Bool(true)
        ticker              := UInt(0)
        max_ticks_for_state := UInt(stop_condition.max_ticks.getOrElse(120))
        state_number := UInt(current_states)
      }
      generator()

      when(stop_condition.condition) {
        printf(s"Leaving state  $name state_number %d ticker %d", state_number, ticker)
        state_locked := Bool(false)
        state_number := state_number + UInt(1)
      }
    }
    current_states += 1
  }
}
