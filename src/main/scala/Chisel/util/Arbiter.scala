// See LICENSE for license details.

/** Arbiters in all shapes and sizes.
  */

package Chisel

/** An I/O bundle for the Arbiter */
class ArbiterIO[T <: Data](gen: T, n: Int) extends Bundle {
  val in  = Flipped(Vec(n, DecoupledIO(gen)))
  val out = DecoupledIO(gen)
  val chosen = Output(UInt(log2Up(n)))
}

/** Arbiter Control determining which producer has access */
private object ArbiterCtrl
{
  def apply(request: Seq[Bool]): Seq[Bool] = request.length match {
    case 0 => Seq()
    case 1 => Seq(Bool(true))
    case _ => Bool(true) +: request.tail.init.scanLeft(request.head)(_ || _).map(!_)
  }
}

abstract class LockingArbiterLike[T <: Data](gen: T, n: Int, count: Int, needsLock: Option[T => Bool]) extends Module {
  def grant: Seq[Bool]
  def choice: UInt
  val io = IO(new ArbiterIO(gen, n))

  io.chosen := choice
  io.out.valid := io.in(io.chosen).valid
  io.out.bits := io.in(io.chosen).bits

  if (count > 1) {
    val lockCount = Counter(count)
    val lockIdx = Reg(UInt())
    val locked = lockCount.value =/= UInt(0)
    val wantsLock = needsLock.map(_(io.out.bits)).getOrElse(Bool(true))

    when (io.out.firing && wantsLock) {
      lockIdx := io.chosen
      lockCount.inc()
    }

    when (locked) { io.chosen := lockIdx }
    for ((in, (g, i)) <- io.in zip grant.zipWithIndex)
      in.ready := Mux(locked, lockIdx === UInt(i), g) && io.out.ready
  } else {
    for ((in, g) <- io.in zip grant)
      in.ready := g && io.out.ready
  }
}

class LockingRRArbiter[T <: Data](gen: T, n: Int, count: Int, needsLock: Option[T => Bool] = None)
    extends LockingArbiterLike[T](gen, n, count, needsLock) {
  lazy val lastGrant = RegEnable(io.chosen, io.out.firing)
  lazy val grantMask = (0 until n).map(UInt(_) > lastGrant)
  lazy val validMask = io.in zip grantMask map { case (in, g) => in.valid && g }

  override def grant: Seq[Bool] = {
    val ctrl = ArbiterCtrl((0 until n).map(i => validMask(i)) ++ io.in.map(_.valid))
    (0 until n).map(i => ctrl(i) && grantMask(i) || ctrl(i + n))
  }

  override lazy val choice = Wire(init=UInt(n-1))
  for (i <- n-2 to 0 by -1)
    when (io.in(i).valid) { choice := UInt(i) }
  for (i <- n-1 to 1 by -1)
    when (validMask(i)) { choice := UInt(i) }
}

class LockingArbiter[T <: Data](gen: T, n: Int, count: Int, needsLock: Option[T => Bool] = None)
    extends LockingArbiterLike[T](gen, n, count, needsLock) {
  def grant: Seq[Bool] = ArbiterCtrl(io.in.map(_.valid))

  override lazy val choice = Wire(init=UInt(n-1))
  for (i <- n-2 to 0 by -1)
    when (io.in(i).valid) { choice := UInt(i) }
}

/** Hardware module that is used to sequence n producers into 1 consumer.
  Producers are chosen in round robin order.

  Example usage:
    val arb = new RRArbiter(2, UInt())
    arb.io.in(0) <> producer0.io.out
    arb.io.in(1) <> producer1.io.out
    consumer.io.in <> arb.io.out
  */
class RRArbiter[T <: Data](gen:T, n: Int) extends LockingRRArbiter[T](gen, n, 1)

/** Hardware module that is used to sequence n producers into 1 consumer.
 Priority is given to lower producer

 Example usage:
   val arb = Module(new Arbiter(2, UInt()))
   arb.io.in(0) <> producer0.io.out
   arb.io.in(1) <> producer1.io.out
   consumer.io.in <> arb.io.out
 */
class Arbiter[T <: Data](gen: T, n: Int) extends Module {
  val io = IO(new ArbiterIO(gen, n))

  io.chosen := UInt(n-1)
  io.out.bits := io.in(n-1).bits
  for (i <- n-2 to 0 by -1) {
    when (io.in(i).valid) {
      io.chosen := UInt(i)
      io.out.bits := io.in(i).bits
    }
  }

  val grant = ArbiterCtrl(io.in.map(_.valid))
  for ((in, g) <- io.in zip grant)
    in.ready := g && io.out.ready
  io.out.valid := !grant.last || io.in.last.valid
}
