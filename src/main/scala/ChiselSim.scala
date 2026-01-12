
import chisel3._
import chisel3.experimental.hierarchy._
import chisel3.simulator.{ChiselSim, ChiselSimMain}
import chisel3.simulator.stimulus.{RunUntilFinished, RunUntilSuccess}
import chisel3.testing.HasTestingDirectory
import chisel3.util.Counter

import java.nio.file.Paths
import java.nio.file.Path

class Foo(w: Int) extends Module {
  val a, b = IO(Input(UInt(w.W)))
  val c = IO(Output(chiselTypeOf(a)))

  private val r = Reg(chiselTypeOf(a))

  r :<= a +% b
  c :<= r
}

object ChiselSimExample extends ChiselSim with App {

  val w = args(0).toInt
  val nCycles = args(1).toInt

  implicit def testdir: HasTestingDirectory = new HasTestingDirectory {
    override def getDirectory: Path = Paths.get("testdir")
  }

  simulate(new Foo(w)) { foo =>
    // Poke different values on the two input ports.
    foo.a.poke(1)
    foo.b.poke(2)

    // Step the clock by one cycle.
    foo.clock.step(nCycles)

    // Expect that the sum of the two inputs is on the output port.
    foo.c.expect(3)
  }
}

object ChiselSimExample2 extends ChiselSimMain(new Foo(8)) {

  override def testdir: HasTestingDirectory = new HasTestingDirectory {
    override def getDirectory: Path = Paths.get("testdir2")
  }

  def test(dut: Foo): Unit = {
    // Poke different values on the two input ports.
    dut.a.poke(1)
    dut.b.poke(2)

    // Step the clock by one cycle.
    dut.clock.step(2)

    // Expect that the sum of the two inputs is on the output port.
    dut.c.expect(3)
  }
}
