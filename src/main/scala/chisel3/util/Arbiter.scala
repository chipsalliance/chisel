// SPDX-License-Identifier: Apache-2.0

/** Arbiters in all shapes and sizes.
  */

package chisel3.util

import chisel3._

/** IO bundle definition for an Arbiter, which takes some number of ready-valid inputs and outputs
  * (selects) at most one.
  * @groupdesc Signals The actual hardware fields of the Bundle
  *
  * @param gen data type
  * @param n number of inputs
  */
class ArbiterIO[T <: Data](private val gen: T, val n: Int) extends Bundle {
  // See github.com/freechipsproject/chisel3/issues/765 for why gen is a private val and proposed replacement APIs.

  /** Input data, one per potential sender
    *
    * @group Signals
    */
  val in = Flipped(Vec(n, Decoupled(gen)))

  /** Output data after arbitration
    *
    * @group Signals
    */
  val out = Decoupled(gen)

  /** Index indicating which sender was chosen as output
    *
    * @group Signals
    */
  val chosen = Output(UInt(log2Ceil(n).W))
}

/** Arbiter Control determining which producer has access
  */
private object ArbiterCtrl {
  def apply(request: Seq[Bool]): Seq[Bool] = request.length match {
    case 0 => Seq()
    case 1 => Seq(true.B)
    case _ => true.B +: request.tail.init.scanLeft(request.head)(_ || _).map(!_)
  }
}

abstract class LockingArbiterLike[T <: Data](gen: T, n: Int, count: Int, needsLock: Option[T => Bool]) extends Module {
  def grant:  Seq[Bool]
  def choice: UInt
  val io = IO(new ArbiterIO(gen, n))

  io.chosen := choice
  io.out.valid := io.in(io.chosen).valid
  io.out.bits := io.in(io.chosen).bits

  if (count > 1) {
    val lockCount = Counter(count)
    val lockIdx = Reg(UInt())
    val locked = lockCount.value =/= 0.U
    val wantsLock = needsLock.map(_(io.out.bits)).getOrElse(true.B)

    when(io.out.fire && wantsLock) {
      lockIdx := io.chosen
      lockCount.inc()
    }

    when(locked) { io.chosen := lockIdx }
    for ((in, (g, i)) <- io.in.zip(grant.zipWithIndex))
      in.ready := Mux(locked, lockIdx === i.asUInt, g) && io.out.ready
  } else {
    for ((in, g) <- io.in.zip(grant))
      in.ready := g && io.out.ready
  }
}

class LockingRRArbiter[T <: Data](gen: T, n: Int, count: Int, needsLock: Option[T => Bool] = None)
    extends LockingArbiterLike[T](gen, n, count, needsLock) {
  // this register is not initialized on purpose, see #267
  lazy val lastGrant = RegEnable(io.chosen, io.out.fire)
  lazy val grantMask = (0 until n).map(_.asUInt > lastGrant)
  lazy val validMask = io.in.zip(grantMask).map { case (in, g) => in.valid && g }

  override def grant: Seq[Bool] = {
    val ctrl = ArbiterCtrl((0 until n).map(i => validMask(i)) ++ io.in.map(_.valid))
    (0 until n).map(i => ctrl(i) && grantMask(i) || ctrl(i + n))
  }

  override lazy val choice = WireDefault((n - 1).asUInt)
  for (i <- n - 2 to 0 by -1)
    when(io.in(i).valid) { choice := i.asUInt }
  for (i <- n - 1 to 1 by -1)
    when(validMask(i)) { choice := i.asUInt }
}

class LockingArbiter[T <: Data](gen: T, n: Int, count: Int, needsLock: Option[T => Bool] = None)
    extends LockingArbiterLike[T](gen, n, count, needsLock) {
  def grant: Seq[Bool] = ArbiterCtrl(io.in.map(_.valid))

  override lazy val choice = WireDefault((n - 1).asUInt)
  for (i <- n - 2 to 0 by -1)
    when(io.in(i).valid) { choice := i.asUInt }
}

/** Hardware module that is used to sequence n producers into 1 consumer.
  * Producers are chosen in round robin order.
  *
  * @param gen data type
  * @param n number of inputs
  * @example {{{
  * val arb = Module(new RRArbiter(UInt(), 2))
  * arb.io.in(0) <> producer0.io.out
  * arb.io.in(1) <> producer1.io.out
  * consumer.io.in <> arb.io.out
  * }}}
  */
class RRArbiter[T <: Data](val gen: T, val n: Int) extends LockingRRArbiter[T](gen, n, 1)

/** Hardware module that is used to sequence n producers into 1 consumer.
  * Priority is given to lower producer.
  *
  * @param gen data type
  * @param n number of inputs
  *
  * @example {{{
  * val arb = Module(new Arbiter(UInt(), 2))
  * arb.io.in(0) <> producer0.io.out
  * arb.io.in(1) <> producer1.io.out
  * consumer.io.in <> arb.io.out
  * }}}
  */
class Arbiter[T <: Data](val gen: T, val n: Int) extends Module {

  /** Give this Arbiter a default, stable desired name using the supplied `Data`
    * generator's `typeName` and input count parameter
    */
  override def desiredName = s"Arbiter${n}_${gen.typeName}"

  val io = IO(new ArbiterIO(gen, n))

  io.chosen := (n - 1).asUInt
  io.out.bits := io.in(n - 1).bits
  for (i <- n - 2 to 0 by -1) {
    when(io.in(i).valid) {
      io.chosen := i.asUInt
      io.out.bits := io.in(i).bits
    }
  }

  val grant = ArbiterCtrl(io.in.map(_.valid))
  for ((in, g) <- io.in.zip(grant))
    in.ready := g && io.out.ready
  io.out.valid := !grant.last || io.in.last.valid
}
