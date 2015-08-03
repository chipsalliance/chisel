package ChiselTests
import Chisel._

class Complex[T <: Data](val re: T, val im: T) extends Bundle {
  override def cloneType: this.type =
    new Complex(re.cloneType, im.cloneType).asInstanceOf[this.type]
}

class ComplexAssign(W: Int) extends Module {
  val io = new Bundle {
    val e   = new Bool(INPUT)
    val in  = new Complex(Bits(width = W), Bits(width = W)).asInput
    val out = new Complex(Bits(width = W), Bits(width = W)).asOutput
  }
  when (io.e) {
    val w = Wire(new Complex(Bits(width = W), Bits(width = W)))
    w := io.in
    io.out.re := w.re
    io.out.im := w.im
  } .otherwise {
    io.out.re := Bits(0)
    io.out.im := Bits(0)
  }
}

class ComplexAssignTester(c: ComplexAssign) extends Tester(c) {
  for (t <- 0 until 4) {
    val test_e     = rnd.nextInt(2)
    val test_in_re = rnd.nextInt(256)
    val test_in_im = rnd.nextInt(256)

    poke(c.io.e,     test_e)
    poke(c.io.in.re, test_in_re)
    poke(c.io.in.im, test_in_im)
    step(1)
    expect(c.io.out.re, if (test_e == 1) test_in_re else 0)
    expect(c.io.out.im, if (test_e == 1) test_in_im else 0)
  }
}
