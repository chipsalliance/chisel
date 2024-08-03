package chiselTests.simulator

import chisel3._
import chisel3.util._

// Test module: splits input stream into output streams based on remainder to n
class SplitMod(val n: Int, w: Int) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(UInt(w.W)))
    val out = Vec(n, Decoupled(UInt(w.W)))
  })

  val inQ = Queue(io.in, 5, pipe = true)

  inQ.ready := io.out.zipWithIndex.map {
    case (out, i) =>
      val in = Wire(inQ.cloneType)
      in.bits :#= inQ.bits
      val isMod = n match {
        case x if isPow2(x) =>
          in.bits.take(log2Ceil(x)) === i.U
        case _ =>
          in.bits % n.U === i.U
      }
      in.valid := inQ.valid && isMod
      out :<>= Queue(in, 3 + i, pipe = true)
      in.ready && isMod
  }.reduce(_ || _)
}
