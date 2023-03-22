package chiselTests.naming

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.util.{Decoupled, Queue}
import chiselTests.ChiselFlatSpec

class TypenameSpec extends ChiselFlatSpec { 
  "Queues" should "have stable, type-parameterized default names" in {
    class Test extends Module {
      val io = IO(new Bundle {
        val foo = Decoupled(UInt(4.W))
        val bar = Decoupled(Bool())
        val fizzbuzz = Decoupled(Vec(3, UInt(8.W)))
      })

      val fooEnq = Wire(Decoupled(UInt(4.W)))
      val fooDeq = Queue(fooEnq, 16) // Queue16_UInt4
      fooEnq <> io.foo

      val barEnq = Wire(Decoupled(Bool()))
      val barDeq = Queue(barEnq, 5) // Queue5_Bool
      barEnq <> io.bar
      
      val fizzbuzzEnq = Wire(Decoupled(Vec(3, UInt(8.W))))
      val fizzbuzzDeq = Queue(fizzbuzzEnq, 32) // Queue32_Vec3_UInt8
      fizzbuzzEnq <> io.fizzbuzz
    }

    val chirrtl = ChiselStage.emitChirrtl(new Test)
    chirrtl should include("module Queue16_UInt4")
    chirrtl should include("module Queue5_Bool")
    chirrtl should include("module Queue32_Vec3_UInt8")
  }
}
