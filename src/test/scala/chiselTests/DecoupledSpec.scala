// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import chisel3.util.Decoupled

class DecoupledSpec extends ChiselFlatSpec {
  "Decoupled() and Decoupled.empty" should "give DecoupledIO with empty payloads" in {
    ChiselStage.emitCHIRRTL(new Module {
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
      .emitCHIRRTL(new Module {
        val enq = IO(Flipped(Decoupled(UInt(8.W))))
        val deq = IO(Decoupled(UInt(8.W)))
        deq <> enq.map(_ + 1.U)
      })

    // Check for data assignment
    chirrtl should include("""node _deq_map_bits_T = add(enq.bits, UInt<1>(0h1)""")
    chirrtl should include("""node _deq_map_bits = tail(_deq_map_bits_T, 1)""")
    chirrtl should include("""connect _deq_map.bits, _deq_map_bits""")
    chirrtl should include("""connect deq, _deq_map""")

    // Check for back-pressure (ready signal is driven in the opposite direction of bits + valid)
    chirrtl should include("""connect enq.ready, _deq_map.ready""")
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
      .emitCHIRRTL(new Module {
        val enq = IO(Flipped(Decoupled(new TestBundle)))
        val deq = IO(Decoupled(new TestBundle))
        deq <> enq.map(func)
      })

    // Check for data assignment
    chirrtl should include("""wire _deq_map_bits : { foo : UInt<8>, bar : UInt<8>, fizz : UInt<1>, buzz : UInt<1>}""")

    chirrtl should include("""node _deq_map_bits_res_foo_T = add(enq.bits.foo, UInt<1>(0h1)""")
    chirrtl should include("""node _deq_map_bits_res_foo_T_1 = tail(_deq_map_bits_res_foo_T, 1)""")
    chirrtl should include("""connect _deq_map_bits.foo, _deq_map_bits_res_foo_T_1""")

    chirrtl should include("""node _deq_map_bits_res_bar_T = sub(enq.bits.bar, UInt<1>(0h1)""")
    chirrtl should include("""node _deq_map_bits_res_bar_T_1 = tail(_deq_map_bits_res_bar_T, 1)""")
    chirrtl should include("""connect _deq_map_bits.bar, _deq_map_bits_res_bar_T_1""")

    chirrtl should include("""connect _deq_map_bits.fizz, UInt<1>(0h0)""")
    chirrtl should include("""connect _deq_map_bits.buzz, UInt<1>(0h1)""")

    chirrtl should include("""connect _deq_map.bits, _deq_map_bits""")
    chirrtl should include("""connect deq, _deq_map""")

    // Check for back-pressure (ready signal is driven in the opposite direction of bits + valid)
    chirrtl should include("""connect enq.ready, _deq_map.ready""")
  }

  "Decoupled.map" should "apply a function to a wrapped Bundle and return a different typed DecoupledIO" in {
    class TestBundle extends Bundle {
      val foo = UInt(8.W)
      val bar = UInt(8.W)
    }

    val chirrtl = ChiselStage
      .emitCHIRRTL(new Module {
        val enq = IO(Flipped(Decoupled(new TestBundle)))
        val deq = IO(Decoupled(UInt(8.W)))
        deq <> enq.map(bundle => bundle.foo & bundle.bar)
      })

    // Check that the _map wire wraps a UInt and not a TestBundle
    chirrtl should include("""wire _deq_map : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}""")

    // Check for data assignment
    chirrtl should include("""node _deq_map_bits = and(enq.bits.foo, enq.bits.bar)""")
    chirrtl should include("""connect _deq_map.bits, _deq_map_bits""")
    chirrtl should include("""connect deq, _deq_map""")

    // Check for back-pressure (ready signal is driven in the opposite direction of bits + valid)
    chirrtl should include("""connect enq.ready, _deq_map.ready""")
  }
}
