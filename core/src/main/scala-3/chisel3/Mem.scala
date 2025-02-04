// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

object Mem extends ObjectMemImpl with SourceInfoDoc {

  /** Creates a combinational/asynchronous-read, sequential/synchronous-write [[Mem]].
    *
    * @param size number of elements in the memory
    * @param t data type of memory element
    */
  def apply[T <: Data](
    size: BigInt,
    t:    T
  )(
    using SourceInfo
  ): Mem[T] = _applyImpl(size, t)

  /** Creates a combinational/asynchronous-read, sequential/synchronous-write [[Mem]].
    *
    * @param size number of elements in the memory
    * @param t data type of memory element
    */
  def apply[T <: Data](size: Int, t: T)(using SourceInfo): Mem[T] = _applyImpl(size, t)
}

sealed abstract class MemBase[T <: Data](val t: T, val length: BigInt, protected val sourceInfo: SourceInfo)
    extends MemBaseImpl[T]
    with SourceInfoDoc {

  // REVIEW TODO: make accessors (static/dynamic, read/write) combinations consistent.

  /** Creates a read accessor into the memory with static addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def apply(idx: BigInt)(using SourceInfo): T = _applyImpl(idx)

  /** Creates a read accessor into the memory with static addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def apply(idx: Int)(using SourceInfo): T = _applyImpl(idx)

  /** Creates a read/write accessor into the memory with dynamic addressing.
    * See the class documentation of the memory for more detailed information.
    */
  def apply(idx: UInt)(using SourceInfo): T = _applyImpl(idx)

  def apply(idx: UInt, clock: Clock)(using SourceInfo): T = _applyImpl(idx, clock)

  /** Creates a read accessor into the memory with dynamic addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def read(idx: UInt)(using SourceInfo): T = _readImpl(idx)

  /** Creates a read accessor into the memory with dynamic addressing.
    * Takes a clock parameter to bind a clock that may be different
    * from the implicit clock. See the class documentation of the memory
    * for more detailed information.
    */
  def read(idx: UInt, clock: Clock)(using SourceInfo): T = _readImpl(idx, clock)

  /** Creates a write accessor into the memory.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    */
  def write(idx: UInt, data: T)(using SourceInfo): Unit = _writeImpl(idx, data)

  /** Creates a write accessor into the memory with a clock
    * that may be different from the implicit clock.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    * @param clock clock to bind to this accessor
    */
  def write(idx: UInt, data: T, clock: Clock)(using SourceInfo): Unit = _writeImpl(idx, data, clock)

  /** Creates a masked write accessor into the memory.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    * @param mask write mask as a Seq of Bool: a write to the Vec element in
    * memory is only performed if the corresponding mask index is true.
    *
    * @note this is only allowed if the memory's element data type is a Vec
    */
  def write(
    idx:  UInt,
    data: T,
    mask: Seq[Bool]
  )(
    using T <:< Vec[_],
    SourceInfo
  ): Unit = _writeImpl(idx, data, mask)

  /** Creates a masked write accessor into the memory with a clock
    * that may be different from the implicit clock.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    * @param mask write mask as a Seq of Bool: a write to the Vec element in
    * memory is only performed if the corresponding mask index is true.
    * @param clock clock to bind to this accessor
    *
    * @note this is only allowed if the memory's element data type is a Vec
    */
  def write(
    idx:   UInt,
    data:  T,
    mask:  Seq[Bool],
    clock: Clock
  )(
    using T <:< Vec[_],
    SourceInfo
  ): Unit = _writeImpl(idx, data, mask, clock)
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
sealed class Mem[T <: Data] private[chisel3] (t: T, length: BigInt, sourceInfo: SourceInfo)
    extends MemBase(t, length, sourceInfo)
    with MemImpl[T]

object SyncReadMem extends ObjectSyncReadMemImpl {

  /** Creates a sequential/synchronous-read, sequential/synchronous-write [[SyncReadMem]].
    *
    * @param size number of elements in the memory
    * @param t data type of memory element
    */
  def apply[T <: Data](
    size: Int,
    t:    T
  )(
    using SourceInfo
  ): SyncReadMem[T] = _applyImpl(size, t)

  def apply[T <: Data](
    size: BigInt,
    t:    T,
    ruw:  ReadUnderWrite = Undefined
  )(
    using SourceInfo
  ): SyncReadMem[T] = _applyImpl(size, t, ruw)
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
sealed class SyncReadMem[T <: Data] private[chisel3] (
  t:                  T,
  n:                  BigInt,
  val readUnderWrite: SyncReadMem.ReadUnderWrite,
  sourceInfo:         SourceInfo
) extends MemBase[T](t, n, sourceInfo)
    with SyncReadMemImpl[T] {

  override def read(idx: UInt)(using SourceInfo): T = _readImpl(idx)

  def read(idx: UInt, en: Bool)(using SourceInfo): T = _readImpl(idx, en)

  def read(idx: UInt, en: Bool, clock: Clock)(using SourceInfo): T = _readImpl(idx, en, clock)

  /** Generates an explicit read-write port for this SyncReadMem. Note that this does not infer
    * port directionality based on connection semantics and the `when` context unlike SyncReadMem.apply(),
    * so the behavior of the port must be controlled by changing the values of the input parameters.
    *
    * @param idx memory element index to write into
    * @param writeData new data to write
    * @param enable enables access to the memory
    * @param isWrite performs a write instead of a read when enable is true; the return
    * value becomes undefined when this parameter is true
    *
    * @return The read data of the memory, which gives the value at idx when enable is true and isWrite is false,
    * or an undefined value otherwise, on the following clock cycle.
    *
    * @example Controlling a read/write port with IO signals
    * {{{
    * class MyMemWrapper extends Module {
    *   val width = 2
    *
    *   val io = IO(new Bundle {
    *     val address = Input(UInt())
    *     val wdata = Input(UInt(width.W))
    *     val enable = Input(Bool())
    *     val isWrite = Input(Bool())
    *     val rdata = Output(UInt(width.W))
    *   })
    *
    *   val mem = SyncReadMem(2, UInt(width.W))
    *   io.rdata := mem.readWrite(io.address, io.wdata, io.enable, io.isWrite)
    * }
    *
    * }}}
    */
  def readWrite(idx: UInt, writeData: T, en: Bool, isWrite: Bool)(using SourceInfo): T =
    _readWriteImpl(idx, writeData, en, isWrite)

  /** Generates an explicit read-write port for this SyncReadMem, using a clock that may be
    * different from the implicit clock.
    *
    * @param idx memory element index to write into
    * @param writeData new data to write
    * @param enable enables access to the memory
    * @param isWrite performs a write instead of a read when enable is true; the return
    * value becomes undefined when this parameter is true
    * @param clock clock to bind to this read-write port
    *
    * @return The read data of the memory, which gives the value at idx when enable is true and isWrite is false,
    * or an undefined value otherwise, on the following clock cycle.
    */
  def readWrite(
    idx:     UInt,
    data:    T,
    en:      Bool,
    isWrite: Bool,
    clock:   Clock
  )(
    using SourceInfo
  ): T = _readWriteImpl(idx, data, en, isWrite, clock)

  /** Generates an explicit read-write port for this SyncReadMem, with a bytemask for
    * performing partial writes to a Vec element.
    *
    * @param idx memory element index to write into
    * @param writeData new data to write
    * @param mask the write mask as a Seq of Bool: a write to the Vec element in
    * memory is only performed if the corresponding mask index is true.
    * @param enable enables access to the memory
    * @param isWrite performs a write instead of a read when enable is true; the return
    * value becomes undefined when this parameter is true
    *
    * @return The read data Vec of the memory at idx when enable is true and isWrite is false,
    * or an undefined value otherwise, on the following clock cycle
    *
    * @example Controlling a read/masked write port with IO signals
    * {{{
    * class MyMaskedMemWrapper extends Module {
    *   val width = 2
    *
    *   val io = IO(new Bundle {
    *     val address = Input(UInt())
    *     val wdata = Input(Vec(2, UInt(width.W)))
    *     val mask = Input(Vec(2, Bool()))
    *     val enable = Input(Bool())
    *     val isWrite = Input(Bool())
    *     val rdata = Output(Vec(2, UInt(width.W)))
    *   })
    *
    *   val mem = SyncReadMem(2, Vec(2, UInt(width.W)))
    *   io.rdata := mem.readWrite(io.address, io.wdata, io.mask, io.enable, io.isWrite)
    * }
    * }}}
    *
    * @note this is only allowed if the memory's element data type is a Vec
    */
  def readWrite(
    idx:       UInt,
    writeData: T,
    mask:      Seq[Bool],
    en:        Bool,
    isWrite:   Bool
  )(
    using T <:< Vec[_],
    SourceInfo
  ): T = _readWriteImpl(idx, writeData, mask, en, isWrite)

  /** Generates an explicit read-write port for this SyncReadMem, with a bytemask for
    * performing partial writes to a Vec element and a clock that may be different from
    * the implicit clock.
    *
    * @param idx memory element index to write into
    * @param writeData new data to write
    * @param mask the write mask as a Seq of Bool: a write to the Vec element in
    * memory is only performed if the corresponding mask index is true.
    * @param enable enables access to the memory
    * @param isWrite performs a write instead of a read when enable is true; the return
    * value becomes undefined when this parameter is true
    * @param clock clock to bind to this read-write port
    *
    * @return The read data Vec of the memory at idx when enable is true and isWrite is false,
    * or an undefined value otherwise, on the following clock cycle
    *
    * @note this is only allowed if the memory's element data type is a Vec
    */
  def readWrite(
    idx:       UInt,
    writeData: T,
    mask:      Seq[Bool],
    en:        Bool,
    isWrite:   Bool,
    clock:     Clock
  )(
    using T <:< Vec[_],
    SourceInfo
  ): T = _readWriteImpl(idx, writeData, mask, en, isWrite, clock)
}
