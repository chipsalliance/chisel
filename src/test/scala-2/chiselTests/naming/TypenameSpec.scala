package chiselTests
package naming

import chisel3._
import chisel3.experimental.Analog
import chisel3.util.{Decoupled, Queue}
import circt.stage.ChiselStage

class TypenameSpec extends ChiselFlatSpec {
  "Queues" should "have stable, type-parameterized default names" in {
    class Test extends Module {
      class AnalogTest[T <: Analog](gen: T) extends Module {
        val bus = IO(gen)

        override def desiredName = s"AnalogTest_${gen.typeName}"
      }

      val io = IO(new Bundle {
        val uint = Decoupled(UInt(4.W))
        val sint = Decoupled(SInt(8.W))
        val bool = Decoupled(Bool())
        val vec = Decoupled(Vec(3, UInt(8.W)))
        val asyncReset = Decoupled(AsyncReset())
        val reset = Decoupled(Reset())
        val clock = Decoupled(Clock())
        val analog = Analog(32.W)
      })

      val uintEnq = Wire(Decoupled(UInt(4.W)))
      val uintDeq = Queue(uintEnq, 16) // Queue16_UInt4
      uintEnq <> io.uint

      val uintInferredEnq = Wire(Decoupled(UInt()))
      val uintInferredDeq = Queue(uintInferredEnq, 15) // Queue16_UInt
      uintInferredEnq <> io.uint

      val sintEnq = Wire(Decoupled(SInt(8.W)))
      val sintDeq = Queue(sintEnq, 8) // Queue8_SInt8
      sintEnq <> io.sint

      val boolEnq = Wire(Decoupled(Bool()))
      val boolDeq = Queue(boolEnq, 5) // Queue5_Bool
      boolEnq <> io.bool

      val vecEnq = Wire(Decoupled(Vec(3, UInt(8.W))))
      val vecDeq = Queue(vecEnq, 32) // Queue32_Vec3_UInt8
      vecEnq <> io.vec

      val asyncResetEnq = Wire(Decoupled(AsyncReset()))
      val asyncResetDeq = Queue(asyncResetEnq, 17) // Queue17_AsyncReset
      asyncResetEnq <> io.asyncReset

      val resetEnq = Wire(Decoupled(Reset()))
      val resetDeq = Queue(resetEnq, 3) // Queue3_Reset
      resetEnq <> io.reset

      val clockEnq = Wire(Decoupled(Clock()))
      val clockDeq = Queue(clockEnq, 20) // Queue20_Clock
      clockEnq <> io.clock

      val analogTest = Module(new AnalogTest(Analog(32.W)))
      analogTest.bus <> io.analog
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Test)
    chirrtl should include("module Queue16_UInt4")
    chirrtl should include("module Queue15_UInt")
    chirrtl should include("module Queue8_SInt8")
    chirrtl should include("module Queue5_Bool")
    chirrtl should include("module Queue32_Vec3_UInt8")
    chirrtl should include("module Queue17_AsyncReset")
    chirrtl should include("module Queue3_Reset")
    chirrtl should include("module Queue20_Clock")
    chirrtl should include("module AnalogTest_Analog32")
  }
}
