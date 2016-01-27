// See LICENSE for license details.

/** Arbiters in all shapes and sizes.
  */

package Chisel

/** An I/O bundle for the Arbiter */
class ArbiterIO[T <: Data](gen: T, n: Int) extends Bundle {
  val in  = Vec(n, Decoupled(gen)).flip
  val out = Decoupled(gen)
  val chosen = UInt(OUTPUT, log2Up(n))
}

/** Arbiter Control determining which producer has access */
object ArbiterCtrl
{
  def apply(request: Seq[Bool]): Seq[Bool] = {
    Bool(true) +: (1 until request.length).map(i => !request.slice(0, i).foldLeft(Bool(false))(_ || _))
  }
}

abstract class LockingArbiterLike[T <: Data](gen: T, n: Int, count: Int, needsLock: Option[T => Bool] = None)
    extends Module {
  require(isPow2(count))
  def grant: Seq[Bool]
  val io = new ArbiterIO(gen, n)
  val locked  = if(count > 1) Reg(init=Bool(false)) else Bool(false)
  val lockIdx = if(count > 1) Reg(init=UInt(n-1)) else UInt(n-1)
  val chosen = Wire(UInt(width = log2Up(n)))

  io.out.valid := io.in(chosen).valid
  io.out.bits := io.in(chosen).bits
  io.chosen := chosen

  io.in(chosen).ready := Bool(false) // XXX FIRRTL workaround
  for ((g, i) <- grant.zipWithIndex)
    io.in(i).ready := Mux(locked, lockIdx === UInt(i), g) && io.out.ready

  if(count > 1){
    val cnt = Reg(init=UInt(0, width = log2Up(count)))
    val cnt_next = cnt + UInt(1)
    when(io.out.fire()) {
      when(needsLock.map(_(io.out.bits)).getOrElse(Bool(true))) {
        cnt := cnt_next
        when(!locked) {
          locked := Bool(true)
          lockIdx := Vec(io.in.map{ in => in.fire()}).indexWhere{i: Bool => i}
        }
      }
      when(cnt_next === UInt(0)) {
        locked := Bool(false)
      }
    }
  }
}

class LockingRRArbiter[T <: Data](gen: T, n: Int, count: Int, needsLock: Option[T => Bool] = None)
    extends LockingArbiterLike[T](gen, n, count, needsLock) {
  lazy val last_grant = Reg(init=UInt(0, log2Up(n)))
  override def grant: Seq[Bool] = {
    val ctrl = ArbiterCtrl((0 until n).map(i => io.in(i).valid && UInt(i) > last_grant) ++ io.in.map(_.valid))
    (0 until n).map(i => ctrl(i) && UInt(i) > last_grant || ctrl(i + n))
  }

  var choose = UInt(n-1)
  for (i <- n-2 to 0 by -1)
    choose = Mux(io.in(i).valid, UInt(i), choose)
  for (i <- n-1 to 1 by -1)
    choose = Mux(io.in(i).valid && UInt(i) > last_grant, UInt(i), choose)
  chosen := Mux(locked, lockIdx, choose)

  when (io.out.fire()) { last_grant := chosen }
}

class LockingArbiter[T <: Data](gen: T, n: Int, count: Int, needsLock: Option[T => Bool] = None)
    extends LockingArbiterLike[T](gen, n, count, needsLock) {
  def grant: Seq[Bool] = ArbiterCtrl(io.in.map(_.valid))

  var choose = UInt(n-1)
  for (i <- n-2 to 0 by -1) {
    choose = Mux(io.in(i).valid, UInt(i), choose)
  }
  chosen := Mux(locked, lockIdx, choose)
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
class Arbiter[T <: Data](gen: T, n: Int) extends LockingArbiter[T](gen, n, 1)
