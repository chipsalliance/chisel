// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros

import firrtl.{ir => fir}

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{MemTransform, SourceInfoTransform}
import chisel3.experimental.{SourceInfo, SourceLine}

object Mem {

  /** Creates a combinational/asynchronous-read, sequential/synchronous-write [[Mem]].
    *
    * @param size number of elements in the memory
    * @param t data type of memory element
    */
  def apply[T <: Data](size: BigInt, t: T): Mem[T] = macro MemTransform.apply[T]

  /** Creates a combinational/asynchronous-read, sequential/synchronous-write [[Mem]].
    *
    * @param size number of elements in the memory
    * @param t data type of memory element
    */
  def apply[T <: Data](size: Int, t: T): Mem[T] = macro MemTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](
    size: BigInt,
    t:    T
  )(
    implicit sourceInfo: SourceInfo
  ): Mem[T] = {
    requireIsChiselType(t, "memory type")
    val mt = t.cloneTypeFull
    val mem = new Mem(mt, size, sourceInfo)
    mt.bind(MemTypeBinding(mem))
    pushCommand(DefMemory(sourceInfo, mem, mt, size))
    mem
  }

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](size: Int, t: T)(implicit sourceInfo: SourceInfo): Mem[T] =
    do_apply(BigInt(size), t)(sourceInfo)
}

sealed abstract class MemBase[T <: Data](val t: T, val length: BigInt, sourceInfo: SourceInfo)
    extends HasId
    with NamedComponent
    with SourceInfoDoc {

  if (t.isConst) Builder.error("Mem type cannot be const.")(sourceInfo)

  requireNoProbeTypeModifier(t, "Cannot make a Mem of a Chisel type with a probe modifier.")(sourceInfo)

  _parent.foreach(_.addId(this))

  // if the memory is created in a scope with an implicit clock (-> clockInst is defined), we will perform checks that
  // ensure memory ports are created with the same clock unless explicitly specified to use a different clock
  private val clockInst: Option[Clock] = Builder.currentClock

  protected def clockWarning(sourceInfo: Option[SourceInfo], dir: MemPortDirection): Unit = {
    // Turn into pretty String if possible, if not, Builder.deprecated will find one via stack trace
    val infoStr = sourceInfo.collect { case s => s.makeMessage(x => x) }
    Builder.deprecated(
      "The clock used to initialize the memory is different than the one used to initialize the port. " +
        "If this is intentional, please pass the clock explicitly when creating the port. This behavior will be an error in 3.6.0",
      infoStr
    )
  }
  // REVIEW TODO: make accessors (static/dynamic, read/write) combinations consistent.

  /** Creates a read accessor into the memory with static addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def apply(x: BigInt): T = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  def do_apply(idx: BigInt)(implicit sourceInfo: SourceInfo): T = {
    require(idx >= 0 && idx < length)
    apply(idx.asUInt, Builder.forcedClock)
  }

  /** Creates a read accessor into the memory with static addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def apply(x: Int): T = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  def do_apply(idx: Int)(implicit sourceInfo: SourceInfo): T =
    do_apply(BigInt(idx))(sourceInfo)

  /** Creates a read/write accessor into the memory with dynamic addressing.
    * See the class documentation of the memory for more detailed information.
    */
  def apply(x: UInt): T = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  def do_apply(idx: UInt)(implicit sourceInfo: SourceInfo): T =
    do_apply_impl(idx, Builder.forcedClock, MemPortDirection.INFER, true)

  def apply(x: UInt, y: Clock): T = macro SourceInfoTransform.xyArg

  def do_apply(idx: UInt, clock: Clock)(implicit sourceInfo: SourceInfo): T =
    do_apply_impl(idx, clock, MemPortDirection.INFER, false)

  /** Creates a read accessor into the memory with dynamic addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def read(x: UInt): T = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  def do_read(idx: UInt)(implicit sourceInfo: SourceInfo): T =
    do_apply_impl(idx, Builder.forcedClock, MemPortDirection.READ, true)

  /** Creates a read accessor into the memory with dynamic addressing.
    * Takes a clock parameter to bind a clock that may be different
    * from the implicit clock. See the class documentation of the memory
    * for more detailed information.
    */
  def read(x: UInt, y: Clock): T = macro SourceInfoTransform.xyArg

  /** @group SourceInfoTransformMacro */
  def do_read(idx: UInt, clock: Clock)(implicit sourceInfo: SourceInfo): T =
    do_apply_impl(idx, clock, MemPortDirection.READ, false)

  protected def do_apply_impl(
    idx:   UInt,
    clock: Clock,
    dir:   MemPortDirection,
    warn:  Boolean
  )(
    implicit sourceInfo: SourceInfo
  ): T = {
    if (warn && clockInst.isDefined && clock != clockInst.get) {
      clockWarning(Some(sourceInfo), dir)
    }
    makePort(sourceInfo, idx, dir, clock)
  }

  /** Creates a write accessor into the memory.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    */
  def write(idx: UInt, data: T): Unit = macro SourceInfoTransform.idxDataArg

  /** @group SourceInfoTransformMacro */
  def do_write(idx: UInt, data: T)(implicit sourceInfo: SourceInfo): Unit =
    write_impl(idx, data, Builder.forcedClock, true)

  /** Creates a write accessor into the memory with a clock
    * that may be different from the implicit clock.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    * @param clock clock to bind to this accessor
    */
  def write(idx: UInt, data: T, clock: Clock): Unit = macro SourceInfoTransform.idxDataClockArg

  /** @group SourceInfoTransformMacro */
  def do_write(idx: UInt, data: T, clock: Clock)(implicit sourceInfo: SourceInfo): Unit =
    write_impl(idx, data, clock, false)

  private def write_impl(
    idx:   UInt,
    data:  T,
    clock: Clock,
    warn:  Boolean
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    if (warn && clockInst.isDefined && clock != clockInst.get) {
      clockWarning(None, MemPortDirection.WRITE)
    }
    makePort(sourceInfo, idx, MemPortDirection.WRITE, clock) := data
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
  def write(
    idx:       UInt,
    writeData: T,
    mask:      Seq[Bool]
  )(
    implicit evidence: T <:< Vec[_]
  ): Unit = macro SourceInfoTransform.idxDataMaskArg

  def do_write(
    idx:  UInt,
    data: T,
    mask: Seq[Bool]
  )(
    implicit evidence: T <:< Vec[_],
    sourceInfo:        SourceInfo
  ): Unit =
    masked_write_impl(idx, data, mask, Builder.forcedClock, true)

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
    idx:       UInt,
    writeData: T,
    mask:      Seq[Bool],
    clock:     Clock
  )(
    implicit evidence: T <:< Vec[_]
  ): Unit = macro SourceInfoTransform.idxDataMaskClockArg

  def do_write(
    idx:   UInt,
    data:  T,
    mask:  Seq[Bool],
    clock: Clock
  )(
    implicit evidence: T <:< Vec[_],
    sourceInfo:        SourceInfo
  ): Unit =
    masked_write_impl(idx, data, mask, clock, false)

  private def masked_write_impl(
    idx:   UInt,
    data:  T,
    mask:  Seq[Bool],
    clock: Clock,
    warn:  Boolean
  )(
    implicit evidence: T <:< Vec[_],
    sourceInfo:        SourceInfo
  ): Unit = {
    if (warn && clockInst.isDefined && clock != clockInst.get) {
      clockWarning(None, MemPortDirection.WRITE)
    }
    val accessor = makePort(sourceInfo, idx, MemPortDirection.WRITE, clock).asInstanceOf[Vec[Data]]
    val dataVec = data.asInstanceOf[Vec[Data]]
    if (accessor.length != dataVec.length) {
      Builder.error(s"Mem write data must contain ${accessor.length} elements (found ${dataVec.length})")
    }
    if (accessor.length != mask.length) {
      Builder.error(s"Mem write mask must contain ${accessor.length} elements (found ${mask.length})")
    }
    for (((cond, port), datum) <- mask.zip(accessor).zip(dataVec))
      when(cond) { port := datum }
  }

  private def makePort(
    sourceInfo: SourceInfo,
    idx:        UInt,
    dir:        MemPortDirection,
    clock:      Clock
  ): T = {
    if (Builder.currentModule != _parent) {
      throwException(
        s"Cannot create a memory port in a different module (${Builder.currentModule.get.name}) than where the memory is (${_parent.get.name})."
      )
    }
    requireIsHardware(idx, "memory port index")
    val i = Vec.truncateIndex(idx, length)(sourceInfo)

    val port = pushCommand(
      DefMemPort(sourceInfo, t.cloneTypeFull, Node(this), dir, i.ref, clock.ref)
    ).id
    // Bind each element of port to being a MemoryPort
    port.bind(MemoryPortBinding(Builder.forcedUserModule, Builder.currentWhen))
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
sealed class Mem[T <: Data] private[chisel3] (t: T, length: BigInt, sourceInfo: SourceInfo)
    extends MemBase(t, length, sourceInfo) {
  override protected def clockWarning(sourceInfo: Option[SourceInfo], dir: MemPortDirection): Unit = {
    // Do not issue clock warnings on reads, since they are not clocked
    if (dir != MemPortDirection.READ)
      super.clockWarning(sourceInfo, dir)
  }
}

object SyncReadMem {

  type ReadUnderWrite = fir.ReadUnderWrite.Value
  val Undefined = fir.ReadUnderWrite.Undefined
  val ReadFirst = fir.ReadUnderWrite.Old
  val WriteFirst = fir.ReadUnderWrite.New

  /** Creates a sequential/synchronous-read, sequential/synchronous-write [[SyncReadMem]].
    *
    * @param size number of elements in the memory
    * @param t data type of memory element
    */
  def apply[T <: Data](size: BigInt, t: T): SyncReadMem[T] = macro MemTransform.apply[T]
  def apply[T <: Data](size: BigInt, t: T, ruw: ReadUnderWrite): SyncReadMem[T] = macro MemTransform.apply_ruw[T]

  /** Creates a sequential/synchronous-read, sequential/synchronous-write [[SyncReadMem]].
    *
    * @param size number of elements in the memory
    * @param t data type of memory element
    */
  def apply[T <: Data](size: Int, t: T): SyncReadMem[T] = macro MemTransform.apply[T]
  def apply[T <: Data](size: Int, t: T, ruw: ReadUnderWrite): SyncReadMem[T] = macro MemTransform.apply_ruw[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](
    size: BigInt,
    t:    T,
    ruw:  ReadUnderWrite = Undefined
  )(
    implicit sourceInfo: SourceInfo
  ): SyncReadMem[T] = {
    requireIsChiselType(t, "memory type")
    val mt = t.cloneTypeFull
    val mem = new SyncReadMem(mt, size, ruw, sourceInfo)
    mt.bind(MemTypeBinding(mem))
    pushCommand(DefSeqMemory(sourceInfo, mem, mt, size, ruw))
    mem
  }

  /** @group SourceInfoTransformMacro */
  // Alternate signatures can't use default parameter values
  def do_apply[T <: Data](
    size: Int,
    t:    T
  )(
    implicit sourceInfo: SourceInfo
  ): SyncReadMem[T] =
    do_apply(BigInt(size), t)(sourceInfo)

  /** @group SourceInfoTransformMacro */
  // Alternate signatures can't use default parameter values
  def do_apply[T <: Data](
    size: Int,
    t:    T,
    ruw:  ReadUnderWrite
  )(
    implicit sourceInfo: SourceInfo
  ): SyncReadMem[T] =
    do_apply(BigInt(size), t, ruw)(sourceInfo)
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
  sourceInfo:         SourceInfo)
    extends MemBase[T](t, n, sourceInfo) {

  override def read(x: UInt): T = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  override def do_read(idx: UInt)(implicit sourceInfo: SourceInfo): T =
    do_read(idx = idx, en = true.B)

  def read(x: UInt, en: Bool): T = macro SourceInfoTransform.xEnArg

  /** @group SourceInfoTransformMacro */
  def do_read(idx: UInt, en: Bool)(implicit sourceInfo: SourceInfo): T =
    _read_impl(idx, en, Builder.forcedClock, true)

  def read(idx: UInt, en: Bool, clock: Clock): T = macro SourceInfoTransform.idxEnClockArg

  /** @group SourceInfoTransformMacro */
  def do_read(idx: UInt, en: Bool, clock: Clock)(implicit sourceInfo: SourceInfo): T =
    _read_impl(idx, en, clock, false)

  /** @group SourceInfoTransformMacro */
  private def _read_impl(
    addr:   UInt,
    enable: Bool,
    clock:  Clock,
    warn:   Boolean
  )(
    implicit sourceInfo: SourceInfo
  ): T = {
    var _port: Option[T] = None
    val _a = WireDefault(chiselTypeOf(addr), DontCare)
    when(enable) {
      _a := addr
      _port = Some(super.do_apply_impl(_a, clock, MemPortDirection.READ, warn))
    }
    _port.get
  }
  // note: we implement do_read(addr) for SyncReadMem in terms of do_read(addr, en) in order to ensure that
  //       `mem.read(addr)` will always behave the same as `mem.read(addr, true.B)`

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
  def readWrite(idx: UInt, writeData: T, en: Bool, isWrite: Bool): T = macro SourceInfoTransform.idxDataEnIswArg

  /** @group SourceInfoTransformMacro */
  def do_readWrite(idx: UInt, writeData: T, en: Bool, isWrite: Bool)(implicit sourceInfo: SourceInfo): T =
    _readWrite_impl(idx, writeData, en, isWrite, Builder.forcedClock, true)

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
  def readWrite(idx: UInt, writeData: T, en: Bool, isWrite: Bool, clock: Clock): T =
    macro SourceInfoTransform.idxDataEnIswClockArg

  /** @group SourceInfoTransformMacro */
  def do_readWrite(
    idx:     UInt,
    data:    T,
    en:      Bool,
    isWrite: Bool,
    clock:   Clock
  )(
    implicit sourceInfo: SourceInfo
  ): T =
    _readWrite_impl(idx, data, en, isWrite, clock, false)

  /** @group SourceInfoTransformMacro */
  private def _readWrite_impl(
    addr:    UInt,
    data:    T,
    enable:  Bool,
    isWrite: Bool,
    clock:   Clock,
    warn:    Boolean
  )(
    implicit sourceInfo: SourceInfo
  ): T = {
    var _port: Option[T] = None
    val _a = WireDefault(chiselTypeOf(addr), DontCare)
    when(enable) {
      _a := addr
      _port = Some(super.do_apply_impl(_a, clock, MemPortDirection.RDWR, warn))

      when(isWrite) {
        _port.get := data
      }
    }
    _port.get
  }

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
    implicit evidence: T <:< Vec[_]
  ): T = macro SourceInfoTransform.idxDataMaskEnIswArg

  def do_readWrite(
    idx:       UInt,
    writeData: T,
    mask:      Seq[Bool],
    en:        Bool,
    isWrite:   Bool
  )(
    implicit evidence: T <:< Vec[_],
    sourceInfo:        SourceInfo
  ): T = masked_readWrite_impl(idx, writeData, mask, en, isWrite, Builder.forcedClock, true)

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
    implicit evidence: T <:< Vec[_]
  ): T = macro SourceInfoTransform.idxDataMaskEnIswClockArg

  def do_readWrite(
    idx:       UInt,
    writeData: T,
    mask:      Seq[Bool],
    en:        Bool,
    isWrite:   Bool,
    clock:     Clock
  )(
    implicit evidence: T <:< Vec[_],
    sourceInfo:        SourceInfo
  ) = masked_readWrite_impl(idx, writeData, mask, en, isWrite, clock, false)

  private def masked_readWrite_impl(
    addr:    UInt,
    data:    T,
    mask:    Seq[Bool],
    enable:  Bool,
    isWrite: Bool,
    clock:   Clock,
    warn:    Boolean
  )(
    implicit evidence: T <:< Vec[_],
    sourceInfo:        SourceInfo
  ): T = {
    var _port: Option[T] = None
    val _a = WireDefault(chiselTypeOf(addr), DontCare)
    when(enable) {
      _a := addr
      _port = Some(super.do_apply_impl(_a, clock, MemPortDirection.RDWR, warn))
      val accessor = _port.get.asInstanceOf[Vec[Data]]

      when(isWrite) {
        val dataVec = data.asInstanceOf[Vec[Data]]
        if (accessor.length != dataVec.length) {
          Builder.error(s"Mem write data must contain ${accessor.length} elements (found ${dataVec.length})")
        }
        if (accessor.length != mask.length) {
          Builder.error(s"Mem write mask must contain ${accessor.length} elements (found ${mask.length})")
        }

        for (((cond, p), datum) <- mask.zip(accessor).zip(dataVec))
          when(cond) { p := datum }
      }
    }
    _port.get
  }
}
