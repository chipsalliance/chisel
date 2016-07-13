// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo, DeprecatedSourceInfo, UnlocatableSourceInfo, MemTransform}

object Mem {
  @deprecated("Mem argument order should be size, t; this will be removed by the official release", "chisel3")
  def apply[T <: Data](t: T, size: Int): Mem[T] = do_apply(size, t)(UnlocatableSourceInfo)

  /** Creates a combinational-read, sequential-write [[Mem]].
    *
    * @param size number of elements in the memory
    * @param t data type of memory element
    */
  def apply[T <: Data](size: Int, t: T): Mem[T] = macro MemTransform.apply[T]

  def do_apply[T <: Data](size: Int, t: T)(implicit sourceInfo: SourceInfo): Mem[T] = {
    val mt  = t.cloneType
    val mem = new Mem(mt, size)
    pushCommand(DefMemory(sourceInfo, mem, mt, size)) // TODO multi-clock
    mem
  }
}

sealed abstract class MemBase[T <: Data](t: T, val length: Int) extends HasId with VecLike[T] {
  // REVIEW TODO: make accessors (static/dynamic, read/write) combinations consistent.

  /** Creates a read accessor into the memory with static addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def apply(idx: Int): T = apply(UInt(idx))

  /** Creates a read/write accessor into the memory with dynamic addressing.
    * See the class documentation of the memory for more detailed information.
    */
  def apply(idx: UInt): T = makePort(UnlocatableSourceInfo, idx, MemPortDirection.INFER, None)

  /** Creates a read accessor into the memory with dynamic addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def read(idx: UInt): T = makePort(UnlocatableSourceInfo, idx, MemPortDirection.READ, None)
  def readOnClock(idx: UInt, clock: Clock): T = makePort(UnlocatableSourceInfo, idx, MemPortDirection.READ, Some(clock))

  /** Creates a write accessor into the memory.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    */
  private def writeSimpleHelper(idx: UInt, data: T, clock: Option[Clock] = None): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo
    makePort(UnlocatableSourceInfo, idx, MemPortDirection.WRITE, clock) := data
  }
  def writeOnClock(idx: UInt, data: T, clock: Clock) = writeSimpleHelper(idx, data, Some(clock))
  def write(idx: UInt, data: T) = writeSimpleHelper(idx, data, None)

  /** Creates a masked write accessor into the memory.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    * @param mask write mask as a Vec of Bool: a write to the Vec element in
    * memory is only performed if the corresponding mask index is true.
    *
    * @note this is only allowed if the memory's element data type is a Vec
    */
  def writeMaskHelper(idx: UInt, data: T, mask: Seq[Bool], clock: Option[Clock] = None) (implicit evidence: T <:< Vec[_]): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo
    val accessor = makePort(sourceInfo, idx, MemPortDirection.WRITE, clock).asInstanceOf[Vec[Data]]
    val dataVec = data.asInstanceOf[Vec[Data]]
    if (accessor.length != dataVec.length) {
      Builder.error(s"Mem write data must contain ${accessor.length} elements (found ${dataVec.length})")
    }
    if (accessor.length != mask.length) {
      Builder.error(s"Mem write mask must contain ${accessor.length} elements (found ${mask.length})")
    }
    for (((cond, port), datum) <- mask zip accessor zip dataVec)
      when (cond) { port := datum }
  }
  def writeOnClock(idx: UInt, data: T, mask: Seq[Bool], clock: Clock) (implicit evidence: T <:< Vec[_]) = writeMaskHelper(idx, data, mask, Some(clock))
  def write(idx: UInt, data: T, mask: Seq[Bool]) (implicit evidence: T <:< Vec[_]) = writeMaskHelper(idx, data, mask, None)

  private def makePort(sourceInfo: SourceInfo, idx: UInt, dir: MemPortDirection, clock: Option[Clock]): T =
    pushCommand(DefMemPort(sourceInfo,
        t.cloneType, Node(this), dir, idx.ref, Node(clock.getOrElse(idx._parent.get.clock)))).id
}

/** A combinational-read, sequential-write memory.
  *
  * Writes take effect on the rising clock edge after the request. Reads are
  * combinational (requests will return data on the same cycle).
  * Read-after-write hazards are not an issue.
  *
  * @note when multiple conflicting writes are performed on a Mem element, the
  * result is undefined (unlike Vec, where the last assignment wins)
  */
sealed class Mem[T <: Data](t: T, length: Int) extends MemBase(t, length)

object SeqMem {
  @deprecated("SeqMem argument order should be size, t; this will be removed by the official release", "chisel3")
  def apply[T <: Data](t: T, size: Int): SeqMem[T] = do_apply(size, t)(DeprecatedSourceInfo)

  /** Creates a sequential-read, sequential-write [[SeqMem]].
    *
    * @param size number of elements in the memory
    * @param t data type of memory element
    */
  def apply[T <: Data](size: Int, t: T): SeqMem[T] = macro MemTransform.apply[T]

  def do_apply[T <: Data](size: Int, t: T)(implicit sourceInfo: SourceInfo): SeqMem[T] = {
    val mt  = t.cloneType
    val mem = new SeqMem(mt, size)
    pushCommand(DefSeqMemory(sourceInfo, mem, mt, size)) // TODO multi-clock
    mem
  }
}

/** A sequential-read, sequential-write memory.
  *
  * Writes take effect on the rising clock edge after the request. Reads return
  * data on the rising edge after the request. Read-after-write behavior (when
  * a read and write to the same address are requested on the same cycle) is
  * undefined.
  *
  * @note when multiple conflicting writes are performed on a Mem element, the
  * result is undefined (unlike Vec, where the last assignment wins)
  */
sealed class SeqMem[T <: Data](t: T, n: Int) extends MemBase[T](t, n) {
  def read(addr: UInt, enable: Bool): T = {
    implicit val sourceInfo = UnlocatableSourceInfo
    val a = Wire(UInt())
    when (enable) { a := addr }
    read(a)
  }
  def readOnClock(addr: UInt, enable: Bool, clock: Clock): T = {
    implicit val sourceInfo = UnlocatableSourceInfo
    val a = Wire(UInt())
    when (enable) { a := addr }
    readOnClock(a, clock)
  }
}
