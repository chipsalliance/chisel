// SPDX-License-Identifier: Apache-2.0

package chisel3

import firrtl.{ir => fir}

import chisel3.internal._
import chisel3.internal.binding._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.ir._
import chisel3.experimental.{requireIsChiselType, requireIsHardware, SourceInfo, SourceLine}

private[chisel3] trait ObjectMemImpl {

  protected def _applyImpl[T <: Data](
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

  protected def _applyImpl[T <: Data](size: Int, t: T)(implicit sourceInfo: SourceInfo): Mem[T] =
    _applyImpl(BigInt(size), t)(sourceInfo)
}

private[chisel3] trait MemBaseImpl[T <: Data] extends HasId with NamedComponent {

  def t:      T
  def length: BigInt

  protected def sourceInfo: SourceInfo

  if (t.isConst) Builder.error("Mem type cannot be const.")(sourceInfo)

  requireNoProbeTypeModifier(t, "Cannot make a Mem of a Chisel type with a probe modifier.")(sourceInfo)

  _parent.foreach(_.addId(this))

  // if the memory is created in a scope with an implicit clock (-> clockInst is defined), we will perform checks that
  // ensure memory ports are created with the same clock unless explicitly specified to use a different clock
  private val clockInst: Option[Clock] = Builder.currentClock

  protected def clockWarning(sourceInfo: Option[SourceInfo], dir: MemPortDirection): Unit = {
    // Turn into pretty String if possible, if not, Builder.deprecated will find one via stack trace
    val infoStr = sourceInfo.collect { case s => s.makeMessage() }
    Builder.deprecated(
      "The clock used to initialize the memory is different than the one used to initialize the port. " +
        "If this is intentional, please pass the clock explicitly when creating the port. This behavior will be an error in 3.6.0",
      infoStr
    )
  }
  // REVIEW TODO: make accessors (static/dynamic, read/write) combinations consistent.

  protected def _applyImpl(idx: BigInt)(implicit sourceInfo: SourceInfo): T = {
    require(idx >= 0 && idx < length)
    _applyImpl(idx.asUInt, Builder.forcedClock)
  }

  protected def _applyImpl(idx: Int)(implicit sourceInfo: SourceInfo): T =
    _applyImpl(BigInt(idx))(sourceInfo)

  protected def _applyImpl(idx: UInt)(implicit sourceInfo: SourceInfo): T =
    _applyImpl(idx, Builder.forcedClock, MemPortDirection.INFER, true)

  protected def _applyImpl(idx: UInt, clock: Clock)(implicit sourceInfo: SourceInfo): T =
    _applyImpl(idx, clock, MemPortDirection.INFER, false)

  protected def _readImpl(idx: UInt)(implicit sourceInfo: SourceInfo): T =
    _applyImpl(idx, Builder.forcedClock, MemPortDirection.READ, true)

  protected def _readImpl(idx: UInt, clock: Clock)(implicit sourceInfo: SourceInfo): T =
    _applyImpl(idx, clock, MemPortDirection.READ, false)

  protected def _applyImpl(
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

  protected def _writeImpl(idx: UInt, data: T)(implicit sourceInfo: SourceInfo): Unit =
    _writeImpl(idx, data, Builder.forcedClock, true)

  protected def _writeImpl(idx: UInt, data: T, clock: Clock)(implicit sourceInfo: SourceInfo): Unit =
    _writeImpl(idx, data, clock, false)

  private def _writeImpl(
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

  protected def _writeImpl(
    idx:  UInt,
    data: T,
    mask: Seq[Bool]
  )(
    implicit evidence: T <:< Vec[_],
    sourceInfo:        SourceInfo
  ): Unit =
    _maskedWriteImpl(idx, data, mask, Builder.forcedClock, true)

  protected def _writeImpl(
    idx:   UInt,
    data:  T,
    mask:  Seq[Bool],
    clock: Clock
  )(
    implicit evidence: T <:< Vec[_],
    sourceInfo:        SourceInfo
  ): Unit =
    _maskedWriteImpl(idx, data, mask, clock, false)

  private def _maskedWriteImpl(
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
private[chisel3] trait MemImpl[T <: Data] extends MemBaseImpl[T] {
  override protected def clockWarning(sourceInfo: Option[SourceInfo], dir: MemPortDirection): Unit = {
    // Do not issue clock warnings on reads, since they are not clocked
    if (dir != MemPortDirection.READ)
      super.clockWarning(sourceInfo, dir)
  }
}

private[chisel3] trait ObjectSyncReadMemImpl {

  type ReadUnderWrite = fir.ReadUnderWrite.Value
  val Undefined = fir.ReadUnderWrite.Undefined
  val ReadFirst = fir.ReadUnderWrite.Old
  val WriteFirst = fir.ReadUnderWrite.New

  protected def _applyImpl[T <: Data](
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

  // Alternate signatures can't use default parameter values
  protected def _applyImpl[T <: Data](
    size: Int,
    t:    T
  )(
    implicit sourceInfo: SourceInfo
  ): SyncReadMem[T] = _applyImpl(BigInt(size), t)

  // Alternate signatures can't use default parameter values
  protected def _applyImpl[T <: Data](
    size: Int,
    t:    T,
    ruw:  ReadUnderWrite
  )(
    implicit sourceInfo: SourceInfo
  ): SyncReadMem[T] = _applyImpl(BigInt(size), t, ruw)
}

private[chisel3] trait SyncReadMemImpl[T <: Data] extends MemBaseImpl[T] {

  def readUnderWrite: SyncReadMem.ReadUnderWrite

  override protected def _readImpl(idx: UInt)(implicit sourceInfo: SourceInfo): T =
    _readImpl(idx = idx, en = true.B)

  protected def _readImpl(idx: UInt, en: Bool)(implicit sourceInfo: SourceInfo): T =
    _readImpl(idx, en, Builder.forcedClock, true)

  protected def _readImpl(idx: UInt, en: Bool, clock: Clock)(implicit sourceInfo: SourceInfo): T =
    _readImpl(idx, en, clock, false)

  private def _readImpl(
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
      _port = Some(super._applyImpl(_a, clock, MemPortDirection.READ, warn))
    }
    _port.get
  }
  // note: we implement do_read(addr) for SyncReadMem in terms of do_read(addr, en) in order to ensure that
  //       `mem.read(addr)` will always behave the same as `mem.read(addr, true.B)`

  protected def _readWriteImpl(idx: UInt, writeData: T, en: Bool, isWrite: Bool)(implicit sourceInfo: SourceInfo): T =
    _readWriteImpl(idx, writeData, en, isWrite, Builder.forcedClock, true)

  protected def _readWriteImpl(
    idx:     UInt,
    data:    T,
    en:      Bool,
    isWrite: Bool,
    clock:   Clock
  )(
    implicit sourceInfo: SourceInfo
  ): T =
    _readWriteImpl(idx, data, en, isWrite, clock, false)

  private def _readWriteImpl(
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
      _port = Some(super._applyImpl(_a, clock, MemPortDirection.RDWR, warn))

      when(isWrite) {
        _port.get := data
      }
    }
    _port.get
  }

  protected def _readWriteImpl(
    idx:       UInt,
    writeData: T,
    mask:      Seq[Bool],
    en:        Bool,
    isWrite:   Bool
  )(
    implicit evidence: T <:< Vec[_],
    sourceInfo:        SourceInfo
  ): T = _maskedReadWriteImpl(idx, writeData, mask, en, isWrite, Builder.forcedClock, true)

  protected def _readWriteImpl(
    idx:       UInt,
    writeData: T,
    mask:      Seq[Bool],
    en:        Bool,
    isWrite:   Bool,
    clock:     Clock
  )(
    implicit evidence: T <:< Vec[_],
    sourceInfo:        SourceInfo
  ) = _maskedReadWriteImpl(idx, writeData, mask, en, isWrite, clock, false)

  private def _maskedReadWriteImpl(
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
      _port = Some(super._applyImpl(_a, clock, MemPortDirection.RDWR, warn))
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
