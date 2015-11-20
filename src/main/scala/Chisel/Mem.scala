// See LICENSE for license details.

package Chisel

import internal._
import internal.Builder.pushCommand
import firrtl._

object Mem {
  @deprecated("Mem argument order should be size, t; this will be removed by the official release", "chisel3")
  def apply[T <: Data](t: T, size: Int): Mem[T] = apply(size, t)

  /** Creates a combinational-read, sequential-write [[Mem]].
    *
    * @param size number of elements in the memory
    * @param t data type of memory element
    */
  def apply[T <: Data](size: Int, t: T): Mem[T] = {
    val mt  = t.cloneType
    val mem = new Mem(mt, size)
    pushCommand(DefMemory(mem, mt, size, Node(mt._parent.get.clock))) // TODO multi-clock
    mem
  }
}

sealed abstract class MemBase[T <: Data](t: T, val length: Int) extends HasId with VecLike[T] {
  // REVIEW TODO: make accessors (static/dynamic, read/write) combinations consistent.

  /** Creates a read accessor into the memory with static addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def apply(idx: Int): T = apply(UInt(idx))

  /** Creates a read accessor into the memory with dynamic addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def apply(idx: UInt): T =
    pushCommand(DefAccessor(t.cloneType, Node(this), NO_DIR, idx.ref)).id

  def read(idx: UInt): T = apply(idx)

  /** Creates a write accessor into the memory.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    */
  def write(idx: UInt, data: T): Unit = apply(idx) := data

  /** Creates a masked write accessor into the memory.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    * @param mask write mask as a Vec of Bool: a write to the Vec element in
    * memory is only performed if the corresponding mask index is true.
    *
    * @note this is only allowed if the memory's element data type is a Vec
    */
  def write(idx: UInt, data: T, mask: Vec[Bool]) (implicit evidence: T <:< Vec[_]): Unit = {
    // REVIEW TODO: error checking to detect zip length mismatch?

    val accessor = apply(idx).asInstanceOf[Vec[Data]]
    for (((cond, port), datum) <- mask zip accessor zip data.asInstanceOf[Vec[Data]])
      when (cond) { port := datum }
  }
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
  def apply[T <: Data](t: T, size: Int): SeqMem[T] = apply(size, t)

  /** Creates a sequential-read, sequential-write [[SeqMem]].
    *
    * @param size number of elements in the memory
    * @param t data type of memory element
    */
  def apply[T <: Data](size: Int, t: T): SeqMem[T] = {
    val mt  = t.cloneType
    val mem = new SeqMem(mt, size)
    pushCommand(DefSeqMemory(mem, mt, size, Node(mt._parent.get.clock))) // TODO multi-clock
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
  def read(addr: UInt, enable: Bool): T =
    read(Mux(enable, addr, Poison(addr)))
}
