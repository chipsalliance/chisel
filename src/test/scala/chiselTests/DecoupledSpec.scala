// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.util.Decoupled

class DecoupledSpec extends ChiselFlatSpec {
  "Decoupled() and Decoupled.empty" should "give DecoupledIO with empty payloads" in {
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle {
        val in = Flipped(Decoupled())
        val out = Decoupled.empty
      })
      io.out <> io.in
      assert(io.asUInt.widthOption.get === 4)
    })
  }

  "Decoupled.map" should "apply a function to a wrapped Data" in {
    val chirrtl = ChiselStage
      .convert(
        ChiselStage.elaborate(new Module {
          val enq = IO(Flipped(Decoupled(UInt(8.W))))
          val deq = IO(Decoupled(UInt(8.W)))
          deq <> enq.map(_ + 1.U)
        })
      )
      .serialize

    // Check for data assignment
    chirrtl should include("""node _deq_res_T = add(enq.bits, UInt<1>("h1")""")
    chirrtl should include("""node deq_res = tail(_deq_res_T, 1)""")
    chirrtl should include("""deq_wire.bits <= deq_res""")
    chirrtl should include("""deq <= deq_wire""")

    // Check for back-pressure (ready signal is driven in the opposite direction of bits + valid)
    chirrtl should include("""enq.ready <= deq_wire.ready""")
  }

  "Decoupled.map" should "apply a function to a wrapped Bundle" in {
    class TestBundle extends Bundle {
      val foo = UInt(8.W)
      val bar = UInt(8.W)
      val fizz = Bool()
      val buzz = Bool()
    }

    // Add one to foo, subtract one from bar, set fizz to false and buzz to true
    def func(t: TestBundle): TestBundle = {
      val res = Wire(new TestBundle)

      res.foo := t.foo + 1.U
      res.bar := t.bar - 1.U
      res.fizz := false.B
      res.buzz := true.B

      res
    }

    val chirrtl = ChiselStage
      .convert(
        ChiselStage.elaborate(new Module {
          val enq = IO(Flipped(Decoupled(new TestBundle)))
          val deq = IO(Decoupled(new TestBundle))
          deq <> enq.map(func)
        })
      )
      .serialize

    // Check for data assignment
    chirrtl should include("""wire deq_res : { foo : UInt<8>, bar : UInt<8>, fizz : UInt<1>, buzz : UInt<1>}""")

    chirrtl should include("""node _deq_res_res_foo_T = add(enq.bits.foo, UInt<1>("h1")""")
    chirrtl should include("""node _deq_res_res_foo_T_1 = tail(_deq_res_res_foo_T, 1)""")
    chirrtl should include("""deq_res.foo <= _deq_res_res_foo_T_1""")

    chirrtl should include("""node _deq_res_res_bar_T = sub(enq.bits.bar, UInt<1>("h1")""")
    chirrtl should include("""node _deq_res_res_bar_T_1 = tail(_deq_res_res_bar_T, 1)""")
    chirrtl should include("""deq_res.bar <= _deq_res_res_bar_T_1""")

    chirrtl should include("""deq_res.fizz <= UInt<1>("h0")""")
    chirrtl should include("""deq_res.buzz <= UInt<1>("h1")""")

    chirrtl should include("""deq_wire.bits <= deq_res""")
    chirrtl should include("""deq <= deq_wire""")

    // Check for back-pressure (ready signal is driven in the opposite direction of bits + valid)
    chirrtl should include("""enq.ready <= deq_wire.ready""")
  }
}
