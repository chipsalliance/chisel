package ChiselTests
import Chisel._

class Coord extends Bundle {
  val x = UInt(width = 32)
  val y = UInt(width = 32)
}

class BundleWire extends Module {
  val io = new Bundle {
    val in   = (new Coord).asInput
    val outs = Vec((new Coord).asOutput, 4)
  }
  val coords = Wire(Vec(new Coord, 4))
  for (i <- 0 until 4) {
    coords(i)  := io.in
    io.outs(i) := coords(i)
  }
}

class BundleWireTester(c: BundleWire) extends Tester(c) {
  for (t <- 0 until 4) {
    val test_in_x = rnd.nextInt(256)
    val test_in_y = rnd.nextInt(256)
    poke(c.io.in.x, test_in_x)
    poke(c.io.in.y, test_in_y)
    step(1)
    for (i <- 0 until 4) {
      expect(c.io.outs(i).x, test_in_x)
      expect(c.io.outs(i).y, test_in_y)
    }
  }
}

