package ChiselTests
import Chisel._
import Chisel.testers._

object Counter {
  def wrapAround(n: UInt, max: UInt) = 
    Mux(n > max, UInt(0), n)
  def apply(max: UInt, en: Bool, amt: UInt): UInt = {
    val x = Reg(init=UInt(0, max.getWidth))
    when (en) { x := wrapAround(x +% amt, max) }
    x
  }
}

class Counter extends Module {
  val io = new Bundle {
    val inc = Bool(INPUT)
    val amt = UInt(INPUT,  4)
    val tot = UInt(OUTPUT, 8)
  }
  io.tot := Counter(UInt(255), io.inc, io.amt)
}

class CounterTester(c: Counter) extends Tester(c) {
  val maxInt  = 16
  var curCnt  = 0

  def intWrapAround(n: Int, max: Int) = 
    if(n > max) 0 else n

  // let it spin for a bit
  for (i <- 0 until 5) {
    step(1)
  }

  for (i <- 0 until 10) {
    val inc = rnd.nextBoolean()
    val amt = rnd.nextInt(maxInt)
    poke(c.io.inc, if (inc) 1 else 0)
    poke(c.io.amt, amt)
    step(1)
    curCnt = if(inc) intWrapAround(curCnt + amt, 255) else curCnt
    expect(c.io.tot, curCnt)
  }
}
