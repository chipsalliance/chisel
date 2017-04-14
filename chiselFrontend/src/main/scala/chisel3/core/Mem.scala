// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo, DeprecatedSourceInfo, UnlocatableSourceInfo, MemTransform}
// TODO: remove this once we have CompileOptions threaded through the macro system.
import chisel3.core.ExplicitCompileOptions.NotStrict

object Mem {
  @deprecated("Mem argument order should be size, t; this will be removed by the official release", "chisel3")
  def apply[T <: Data](t: T, size: Int): Mem[T] = do_apply(size, t)(UnlocatableSourceInfo)

  /** Creates a combinational/asynchronous-read, sequential/synchronous-write [[Mem]].
    *
    * @param size number of elements in the memory
    * @param t data type of memory element
    */
  def apply[T <: Data](size: Int, t: T): Mem[T] = macro MemTransform.apply[T]
  def do_apply[T <: Data](size: Int, t: T)(implicit sourceInfo: SourceInfo): Mem[T] = {
    val mt  = t.chiselCloneType
    Binding.bind(mt, NoDirectionBinder, "Error: fresh t")
    // TODO(twigg): Remove need for this Binding

    val mem = new Mem(mt, size)
    pushCommand(DefMemory(sourceInfo, mem, mt, size))
    mem
  }
}

sealed abstract class MemBase[T <: Data](t: T, val length: Int) extends HasId with VecLike[T] {
  // REVIEW TODO: make accessors (static/dynamic, read/write) combinations consistent.

  /** Creates a read accessor into the memory with static addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def apply(idx: Int): T = {
    require(idx >= 0 && idx < length)
    apply(idx.asUInt)
  }

  /** Creates a read/write accessor into the memory with dynamic addressing.
    * See the class documentation of the memory for more detailed information.
    */
  def apply(idx: UInt): T = makePort(UnlocatableSourceInfo, idx, MemPortDirection.INFER)

  /** Creates a read accessor into the memory with dynamic addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def read(idx: UInt): T = makePort(UnlocatableSourceInfo, idx, MemPortDirection.READ)

  /** Creates a write accessor into the memory.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    */
  def write(idx: UInt, data: T): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo
    makePort(UnlocatableSourceInfo, idx, MemPortDirection.WRITE) := data
  }

  /** Creates a masked write accessor into the memory.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    * @param mask write mask as a Seq of Bool: a write to the Vec element in
    * memory is only performed if the corresponding mask index is true.
    *
    * @note this is only allowed if the memory's element data type is a Vec
    */
  def write(idx: UInt, data: T, mask: Seq[Bool]) (implicit evidence: T <:< Vec[_]): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo
    val accessor = makePort(sourceInfo, idx, MemPortDirection.WRITE).asInstanceOf[Vec[Data]]
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

  private def makePort(sourceInfo: SourceInfo, idx: UInt, dir: MemPortDirection): T = {
    Binding.checkSynthesizable(idx, s"'idx' ($idx)")
    val i = Vec.truncateIndex(idx, length)(sourceInfo)

    val port = pushCommand(
      DefMemPort(sourceInfo,
       t.chiselCloneType, Node(this), dir, i.ref, Node(Builder.forcedClock))
    ).id
    // Bind each element of port to being a MemoryPort
    Binding.bind(port, MemoryPortBinder(Builder.forcedUserModule), "Error: Fresh t")
    port
  }
}

/** A combinational/asynchronous-read, sequential/synchronous-write memory.
  *
  * Writes take effect on the rising clock edge after the request. Reads are
  * combinational (requests will return data on the same cycle).
  * Read-after-write hazards are not an issue.
  *
  * @note when multiple conflicting writes are performed on a Mem element, the
  * result is undefined (unlike Vec, where the last assignment wins)
  */
sealed class Mem[T <: Data](t: T, length: Int) extends MemBase(t, length)

object SyncReadMem {
  @deprecated("SeqMem/SyncReadMem argument order should be size, t; this will be removed by the official release", "chisel3")
  def apply[T <: Data](t: T, size: Int): SyncReadMem[T] = do_apply(size, t)(DeprecatedSourceInfo)

  /** Creates a sequential/synchronous-read, sequential/synchronous-write [[SyncReadMem]].
    *
    * @param size number of elements in the memory
    * @param t data type of memory element
    */
  def apply[T <: Data](size: Int, t: T): SyncReadMem[T] = macro MemTransform.apply[T]

  def do_apply[T <: Data](size: Int, t: T)(implicit sourceInfo: SourceInfo): SyncReadMem[T] = {
    val mt  = t.chiselCloneType
    Binding.bind(mt, NoDirectionBinder, "Error: fresh t")
    // TODO(twigg): Remove need for this Binding

    val mem = new SyncReadMem(mt, size)
    pushCommand(DefSeqMemory(sourceInfo, mem, mt, size))
    mem
  }
}

/** A sequential/synchronous-read, sequential/synchronous-write memory.
  *
  * Writes take effect on the rising clock edge after the request. Reads return
  * data on the rising edge after the request. Read-after-write behavior (when
  * a read and write to the same address are requested on the same cycle) is
  * undefined.
  *
  * @note when multiple conflicting writes are performed on a Mem element, the
  * result is undefined (unlike Vec, where the last assignment wins)
  */
sealed class SyncReadMem[T <: Data](t: T, n: Int) extends MemBase[T](t, n) {
  def read(addr: UInt, enable: Bool): T = {
    implicit val sourceInfo = UnlocatableSourceInfo
    val a = Wire(UInt())
    var port: Option[T] = None
    when (enable) {
      a := addr
      port = Some(read(a))
    }
    port.get
  }
}
