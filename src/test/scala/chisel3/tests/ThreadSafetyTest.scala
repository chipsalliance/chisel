package chisel3.tests

import org.scalatest._

import chisel3._
import chisel3.testers2._

class ThreadSafetyTest extends FlatSpec with ChiselScalatestTester {
  behavior of "Testers2 thread safety checker"

  it should "disallow simultaneous pokes from two threads" in {
    assertThrows[ThreadOrderDependentException] {
      test(new Module {
        val io = IO(new Bundle {
          val in = Input(Bool())
        })
      }) { c =>
        fork {
          c.io.in.poke(true.B)
          c.clock.step(1)  // TODO: remove need for explicit clock
        }
        fork {
          c.io.in.poke(true.B)
          c.clock.step(1)
        }
        c.clock.step(2)  // TODO: use thread joins
      }
    }
  }

  it should "disallow simultaneous pokes from two threads, even when overridden" in {
    assertThrows[ThreadOrderDependentException] {
      test(new Module {
        val io = IO(new Bundle {
          val in = Input(Bool())
        })
      }) { c =>
        c.io.in.poke(true.B)
        fork {
          c.io.in.weakPoke(true.B)
          c.clock.step(1)  // TODO: remove need for explicit clock
        }
        fork {
          c.io.in.weakPoke(true.B)
          c.clock.step(1)
        }
        c.clock.step(2)  // TODO: use thread joins
      }
    }
  }

  it should "disallow simultaneous peeks and pokes from two threads" in {
    assertThrows[ThreadOrderDependentException] {
      test(new Module {
        val io = IO(new Bundle {
          val in = Input(Bool())
        })
      }) { c =>
        c.io.in.poke(true.B)  // TODO: expect should not trigger until end of timestep
        fork {
          c.io.in.expect(true.B)
          c.clock.step(1)  // TODO: remove need for explicit clock
        }
        fork {
          c.io.in.poke(true.B)
          c.clock.step(1)
        }
        c.clock.step(2)  // TODO: use thread joins
      }
    }
  }
}
