import chisel3._

val Pi = math.Pi
def sinTable(amp: Double, n: Int) = {
  val times =
    (0 until n).map(i => (i*2*Pi)/(n.toDouble-1) - Pi)
  val inits =
    times.map(t => Math.round(amp * math.sin(t)).asSInt(32.W))
  VecInit(inits)
}

//sinTable(10.0, 16)
