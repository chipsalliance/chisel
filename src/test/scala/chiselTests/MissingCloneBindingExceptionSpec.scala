// See LICENSE for license details.

package chiselTests
import Chisel.ChiselException
import org.scalatest._

class MissingCloneBindingExceptionSpec extends ChiselFlatSpec with Matchers {
  behavior of "missing cloneType in Chisel3"
  ( the[ChiselException] thrownBy {
    import chisel3._

    class TestIO(w: Int) extends Bundle {
      val a = Input(Vec(4, UInt(w.W)))

      //override def cloneType = (new TestIO(w)).asInstanceOf[this.type]
    }

    class Test extends Module {
      val io = IO(new TestIO(32))
    }

    class TestTop extends Module {
      val io = IO(new Bundle {})

      val subs = VecInit(Seq.fill(2) {
        Module(new Test).io
      })
    }

    elaborate(new TestTop)
  }).getMessage should include("needs cloneType method")

  behavior of "missing cloneType in Chisel2"
  ( the[ChiselException] thrownBy {
    import Chisel._

    class TestIO(w: Int) extends Bundle {
      val a = Vec(4, UInt(width = w)).asInput

      //override def cloneType = (new TestIO(w)).asInstanceOf[this.type]
    }

    class Test extends Module {
      val io = IO(new TestIO(32))
    }

    class TestTop extends Module {
      val io = IO(new Bundle {})

      val subs = Vec.fill(2) {
        Module(new Test).io
      }
    }

    elaborate(new TestTop)
  }).getMessage should include("needs cloneType method")
}
