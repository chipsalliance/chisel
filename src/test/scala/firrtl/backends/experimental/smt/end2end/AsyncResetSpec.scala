// See LICENSE for license details.

package firrtl.backends.experimental.smt.end2end

import firrtl.annotations.CircuitTarget
import firrtl.backends.experimental.smt.{GlobalClockAnnotation, StutteringClockTransform}
import firrtl.options.Dependency
import firrtl.stage.RunFirrtlTransformAnnotation

class AsyncResetSpec extends EndToEndSMTBaseSpec {
  def annos(name: String) = Seq(
    RunFirrtlTransformAnnotation(Dependency[StutteringClockTransform]),
    GlobalClockAnnotation(CircuitTarget(name).module(name).ref("global_clock")))

  "a module with asynchronous reset" should "allow a register to change between clock edges" taggedAs(RequiresZ3) in {
    def in(resetType: String) =
      s"""circuit AsyncReset00:
         |  module AsyncReset00:
         |    input global_clock: Clock
         |    input c: Clock
         |    input reset: $resetType
         |    input preset: AsyncReset
         |
         |    ; a register with async reset
         |    reg r: UInt<4>, c with: (reset => (reset, UInt(3)))
         |
         |    ; a counter/toggler connected to the clock c
         |    reg count: UInt<1>, c with: (reset => (preset, UInt(0)))
         |    count <= add(count, UInt(1))
         |
         |    ; the past machinery and the assertion uses the global clock
         |    reg past_valid: UInt<1>, global_clock with: (reset => (preset, UInt(0)))
         |    past_valid <= UInt(1)
         |    reg past_r: UInt<4>, global_clock
         |    past_r <= r
         |    reg past_count: UInt<1>, global_clock
         |    past_count <= count
         |
         |    ; can the value of r change without the count changing?
         |    assert(global_clock, or(not(eq(count, past_count)), eq(r, past_r)), past_valid, "count = past(count) |-> r = past(r)")
         |""".stripMargin
    test(in("AsyncReset"), MCFail(1), kmax=2, annos=annos("AsyncReset00"))
    test(in("UInt<1>"), MCSuccess, kmax=2, annos=annos("AsyncReset00"))
  }

}
