// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.util._
import chisel3.testers.BasicTester

class Arbiter[T <: Data: Manifest](n: Int, private val gen: T) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Vec(n, new DecoupledIO(gen)))
    val out = new DecoupledIO(gen)
  })

  def arbitrateTwo(a: DecoupledIO[T], b: DecoupledIO[T]) = {

    val idleA :: idleB :: hasA :: hasB :: Nil = Enum(4)
    val regData = Reg(gen)
    val regState = RegInit(idleA)
    val out = Wire(new DecoupledIO(gen))

    a.ready := regState === idleA
    b.ready := regState === idleB
    out.valid := (regState === hasA || regState === hasB)

    switch(regState) {
      is(idleA) {
        when(a.valid) {
          regData := a.bits
          regState := hasA
        }.otherwise {
          regState := idleB
        }
      }
      is(idleB) {
        when(b.valid) {
          regData := b.bits
          regState := hasB
        }.otherwise {
          regState := idleA
        }
      }
      is(hasA) {
        when(out.ready) {
          regState := idleB
        }
      }
      is(hasB) {
        when(out.ready) {
          regState := idleA
        }
      }
    }

    out.bits := regData.asUInt + 1.U
    out
  }

  io.out <> io.in.reduceTree(arbitrateTwo)
}

class ReduceTreeBalancedTester(nodes: Int) extends BasicTester {

  val cnt = RegInit(0.U(8.W))
  val min = RegInit(99.U(8.W))
  val max = RegInit(0.U(8.W))

  val dut = Module(new Arbiter(nodes, UInt(16.W)))
  for (i <- 0 until nodes) {
    dut.io.in(i).valid := true.B
    dut.io.in(i).bits := 0.U
  }
  dut.io.out.ready := true.B

  when(dut.io.out.valid) {
    val hops = dut.io.out.bits
    when(hops < min) {
      min := hops
    }
    when(hops > max) {
      max := hops
    }
  }

  when(!(max === 0.U || min === 99.U)) {
    assert(max - min <= 1.U)
  }

  cnt := cnt + 1.U
  when(cnt === 10.U) {
    stop()
  }
}

class ReduceTreeBalancedSpec extends ChiselPropSpec {
  property("Tree shall be fair and shall have a maximum difference of one hop for each node") {

    // This test will fail for 5 nodes due to an unbalanced tree.
    // A fix is on the way.
    for (n <- 1 to 5) {
      assertTesterPasses {
        new ReduceTreeBalancedTester(n)
      }
    }
  }
}
