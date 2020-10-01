// See LICENSE for license details.

package chisel3.internal

import chisel3._
import chisel3.internal.firrtl._

trait MemoizableCommand

class MemoizableDefPrim[T <: Data](val dp: DefPrim[T]) extends MemoizableCommand {
  override val hashCode: Int = dp.op.hashCode ^ dp.args.hashCode

  override def equals(that: Any): Boolean = that match {
    case m: MemoizableDefPrim[_] => typesEquivalent(dp.id, m.dp.id) && dp.op == m.dp.op && dp.args == m.dp.args
    case _ => false
  }

  private def typesEquivalent(t0: Data, t1: Data): Boolean = (t0, t1) match {
    case (_: Bool, _: Bool) => true
    case (u0: UInt, u1: UInt) => u0.width == u1.width
    case (s0: SInt, s1: SInt) => s0.width == s1.width
    case _ => false
  }
}

trait CommandMemoization {
  private val memoizedCommands = collection.mutable.HashMap[MemoizableCommand, Data]()

  private[chisel3] def addMemoizableCommand(c: MemoizableCommand, d: Data) {
    memoizedCommands += c -> d
  }

  private[chisel3] def getMemoizedCommand(c: MemoizableCommand): Option[Data] = {
    memoizedCommands.get(c)
  }
}

trait DisableCommandMemoization extends CommandMemoization {
  private[chisel3] override def addMemoizableCommand(c: MemoizableCommand, d: Data) { }
}
