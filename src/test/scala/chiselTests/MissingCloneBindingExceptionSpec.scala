// See LICENSE for license details.

package chiselTests
import Chisel.ChiselException
import chisel3.stage.ChiselStage
import org.scalatest._
import org.scalatest.matchers.should.Matchers

class MissingCloneBindingExceptionSpec extends ChiselFlatSpec with Matchers with Utils {
  behavior of "missing cloneType in Chisel3"
  ( the [ChiselException] thrownBy extractCause[ChiselException] {
    import chisel3._

    class Test extends Module {
      class TestIO(w: Int) extends Bundle {
        val a = Input(Vec(4, UInt(w.W)))
      }

      val io = IO(new TestIO(32))
    }

    class TestTop extends Module {
      val io = IO(new Bundle {})

      val subs = VecInit(Seq.fill(2) {
        Module(new Test).io
      })
    }

    ChiselStage.elaborate(new TestTop)
  }).getMessage should include("make all parameters immutable")

  behavior of "missing cloneType in Chisel2"
  ( the [ChiselException] thrownBy extractCause[ChiselException] {
    import Chisel._

    class Test extends Module {
      class TestIO(w: Int) extends Bundle {
        val a = Vec(4, UInt(width = w)).asInput
      }

      val io = IO(new TestIO(32))
    }

    class TestTop extends Module {
      val io = IO(new Bundle {})

      val subs = Vec.fill(2) {
        Module(new Test).io
      }
    }

    ChiselStage.elaborate(new TestTop)
  }).getMessage should include("make all parameters immutable")
}
