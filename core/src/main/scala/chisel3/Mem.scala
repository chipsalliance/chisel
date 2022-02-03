// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros

import firrtl.{ir => fir}

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{MemTransform, SourceInfo, SourceInfoTransform, SourceLine, UnlocatableSourceInfo}

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
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Mem[T] = {
    if (compileOptions.declaredTypeMustBeUnbound) {
      requireIsChiselType(t, "memory type")
    }
    val mt = t.cloneTypeFull
    val mem = new Mem(mt, size)
    mt.bind(MemTypeBinding(mem))
    pushCommand(DefMemory(sourceInfo, mem, mt, size))
    mem
  }

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](size: Int, t: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Mem[T] =
    do_apply(BigInt(size), t)(sourceInfo, compileOptions)
}

sealed abstract class MemBase[T <: Data](val t: T, val length: BigInt)
    extends HasId
    with NamedComponent
    with SourceInfoDoc {
  _parent.foreach(_.addId(this))

  private val clockInst: Clock = Builder.forcedClock

  protected def clockWarning(sourceInfo: Option[SourceInfo]): Unit = {
    // Turn into pretty String if possible, if not, Builder.deprecated will find one via stack trace
    val infoStr = sourceInfo.collect { case SourceLine(file, line, col) => s"$file:$line:$col" }
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
  def do_apply(idx: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    require(idx >= 0 && idx < length)
    apply(idx.asUInt, Builder.forcedClock)
  }

  /** Creates a read accessor into the memory with static addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def apply(x: Int): T = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  def do_apply(idx: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    do_apply(BigInt(idx))(sourceInfo, compileOptions)

  /** Creates a read/write accessor into the memory with dynamic addressing.
    * See the class documentation of the memory for more detailed information.
    */
  def apply(x: UInt): T = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  def do_apply(idx: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    do_apply_impl(idx, Builder.forcedClock, MemPortDirection.INFER, true)

  def apply(x: UInt, y: Clock): T = macro SourceInfoTransform.xyArg

  def do_apply(idx: UInt, clock: Clock)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    do_apply_impl(idx, clock, MemPortDirection.INFER, false)

  /** Creates a read accessor into the memory with dynamic addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def read(x: UInt): T = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  def do_read(idx: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    do_apply_impl(idx, Builder.forcedClock, MemPortDirection.READ, true)

  /** Creates a read accessor into the memory with dynamic addressing.
    * Takes a clock parameter to bind a clock that may be different
    * from the implicit clock. See the class documentation of the memory
    * for more detailed information.
    */
  def read(x: UInt, y: Clock): T = macro SourceInfoTransform.xyArg

  /** @group SourceInfoTransformMacro */
  def do_read(idx: UInt, clock: Clock)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    do_apply_impl(idx, clock, MemPortDirection.READ, false)

  protected def do_apply_impl(
    idx:   UInt,
    clock: Clock,
    dir:   MemPortDirection,
    warn:  Boolean
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): T = {
    if (warn && clock != clockInst) {
      clockWarning(Some(sourceInfo))
    }
    makePort(sourceInfo, idx, dir, clock)
  }

  /** Creates a write accessor into the memory.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    */
  def write(idx: UInt, data: T)(implicit compileOptions: CompileOptions): Unit =
    write_impl(idx, data, Builder.forcedClock, true)

  /** Creates a write accessor into the memory with a clock
    * that may be different from the implicit clock.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    * @param clock clock to bind to this accessor
    */
  def write(idx: UInt, data: T, clock: Clock)(implicit compileOptions: CompileOptions): Unit =
    write_impl(idx, data, clock, false)

  private def write_impl(
    idx:   UInt,
    data:  T,
    clock: Clock,
    warn:  Boolean
  )(
    implicit compileOptions: CompileOptions
  ): Unit = {
    if (warn && clock != clockInst) {
      clockWarning(None)
    }
    implicit val sourceInfo = UnlocatableSourceInfo
    makePort(UnlocatableSourceInfo, idx, MemPortDirection.WRITE, clock) := data
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
    idx:  UInt,
    data: T,
    mask: Seq[Bool]
  )(
    implicit evidence: T <:< Vec[_],
    compileOptions:    CompileOptions
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
    idx:   UInt,
    data:  T,
    mask:  Seq[Bool],
    clock: Clock
  )(
    implicit evidence: T <:< Vec[_],
    compileOptions:    CompileOptions
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
    compileOptions:    CompileOptions
  ): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo
    if (warn && clock != clockInst) {
      clockWarning(None)
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
  )(
    implicit compileOptions: CompileOptions
  ): T = {
    requireIsHardware(idx, "memory port index")
    val i = Vec.truncateIndex(idx, length)(sourceInfo, compileOptions)

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
sealed class Mem[T <: Data] private (t: T, length: BigInt) extends MemBase(t, length)

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
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): SyncReadMem[T] = {
    if (compileOptions.declaredTypeMustBeUnbound) {
      requireIsChiselType(t, "memory type")
    }
    val mt = t.cloneTypeFull
    val mem = new SyncReadMem(mt, size, ruw)
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
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): SyncReadMem[T] =
    do_apply(BigInt(size), t)(sourceInfo, compileOptions)

  /** @group SourceInfoTransformMacro */
  // Alternate signatures can't use default parameter values
  def do_apply[T <: Data](
    size: Int,
    t:    T,
    ruw:  ReadUnderWrite
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): SyncReadMem[T] =
    do_apply(BigInt(size), t, ruw)(sourceInfo, compileOptions)
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
sealed class SyncReadMem[T <: Data] private (t: T, n: BigInt, val readUnderWrite: SyncReadMem.ReadUnderWrite)
    extends MemBase[T](t, n) {

  override def read(x: UInt): T = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  override def do_read(idx: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    do_read(idx = idx, en = true.B)

  def read(x: UInt, en: Bool): T = macro SourceInfoTransform.xEnArg

  /** @group SourceInfoTransformMacro */
  def do_read(idx: UInt, en: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    _read_impl(idx, en, Builder.forcedClock, true)

  def read(idx: UInt, en: Bool, clock: Clock): T = macro SourceInfoTransform.xyzArg

  /** @group SourceInfoTransformMacro */
  def do_read(idx: UInt, en: Bool, clock: Clock)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    _read_impl(idx, en, clock, false)

  /** @group SourceInfoTransformMacro */
  private def _read_impl(
    addr:   UInt,
    enable: Bool,
    clock:  Clock,
    warn:   Boolean
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): T = {
    val a = Wire(UInt())
    a := DontCare
    var port: Option[T] = None
    when(enable) {
      a := addr
      port = Some(super.do_apply_impl(a, clock, MemPortDirection.READ, warn))
    }
    port.get
  }
  // note: we implement do_read(addr) for SyncReadMem in terms of do_read(addr, en) in order to ensure that
  //       `mem.read(addr)` will always behave the same as `mem.read(addr, true.B)`
}
