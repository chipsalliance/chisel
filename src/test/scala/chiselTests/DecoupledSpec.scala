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
    val chirrtl = ChiselStage.convert(
      ChiselStage.elaborate(new Module {
        val enq = IO(Flipped(Decoupled(UInt(8.W))))
        val deq = IO(Decoupled(UInt(8.W)))
        deq <> enq.map(_ + 1.U)
      })
    )

    chirrtl.serialize should include("""node _deq_res_T = add(enq.bits, UInt<1>("h1")""")
    chirrtl.serialize should include("""node deq_res = tail(_deq_res_T, 1)""")
    chirrtl.serialize should include("""deq_wire.bits <= deq_res""")
    chirrtl.serialize should include("""deq <= deq_wire""")
  }

  "Decoupled.map" should "apply a function to a wrapped Bundle" in {
    val chirrtl = ChiselStage.convert(
      ChiselStage.elaborate(new Module {
        val enq = IO(Flipped(Decoupled(UInt(8.W))))
        val deq = IO(Decoupled(UInt(8.W)))
        deq <> enq.map(_ + 1.U)
      })
    )

    chirrtl.serialize should include("""node _deq_res_T = add(enq.bits, UInt<1>("h1")""")
    chirrtl.serialize should include("""node deq_res = tail(_deq_res_T, 1)""")
    chirrtl.serialize should include("""deq_wire.bits <= deq_res""")
    chirrtl.serialize should include("""deq <= deq_wire""")
  }
}
