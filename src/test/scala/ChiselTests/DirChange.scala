package ChiselTests
import Chisel._
import Chisel.testers._

class DirChange extends Module {
  val io = new Bundle {
    val test1 = UInt(INPUT, 5).asOutput
    val test2 = UInt(OUTPUT, 5).asInput
    val test3 = Vec( UInt(OUTPUT, 2), 10)
    val test4 = new Bundle {
      val test41 = SInt(INPUT, 5)
      val test42 = SInt(OUTPUT, 5)
    }.asInput
  }.flip
}

class DirChangeTester(c: DirChange) extends Tester(c) {
}
