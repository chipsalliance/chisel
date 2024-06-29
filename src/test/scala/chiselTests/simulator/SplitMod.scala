package chiselTests.simulator

import chisel3._
import chisel3.util._

// Test module: splits input stream into output streams based on remainder to n
class SplitMod(val n: Int, w: Int) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(UInt(w.W)))
    val out = Vec(n, Decoupled(UInt(w.W)))
  })

  val inQ = Queue(io.in, 3, pipe = true)

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
      out :<>= Queue(in, 3, pipe = true)
      in.ready && isMod
  }.reduce(_ || _)
}

// combine input streams into a single output stream by adding odd inputs
class OddsSum(val n: Int, w: Int) extends Module {
  val io = IO(new Bundle {
    val in = Vec(n, Flipped(Decoupled(UInt(w.W))))
    val out = Decoupled(UInt(w.W))
  })

  io.in.zipWithIndex.foreach {
    case (in, i) =>
      val othersValid = io.in.zipWithIndex.filterNot(_._2 == i).map(_._1.valid).reduce(_ && _)
      in.ready := othersValid && io.out.ready
  }

  io.out.bits :#= io.in.map(in => Mux(in.bits(0), in.bits, 0.U)).reduce(_ + _)
  io.out.valid := io.in.map(_.valid).reduce(_ && _)
}

// combine input streams into a single output stream by adding them up
class Adder(val n: Int, w: Int) extends Module {
  val io = IO(new Bundle {
    val in = Vec(n, Flipped(Decoupled(UInt(w.W))))
    val out = Decoupled(UInt(w.W))
  })

  val out = Wire(io.out.cloneType)

  io.in.zipWithIndex.foreach {
    case (in, i) =>
      val othersValid = io.in.zipWithIndex.filterNot(_._2 == i).map(_._1.valid).reduce(_ && _)
      in.ready := othersValid && out.ready
  }

  out.bits :#= io.in.map(_.bits).reduce(_ + _)
  out.valid := io.in.map(_.valid).reduce(_ && _)

  // io.out :<>= Queue(out, 0, pipe = true)
  io.out :<>= out
}
