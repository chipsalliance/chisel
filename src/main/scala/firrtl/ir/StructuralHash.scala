// See LICENSE for license details.

package firrtl.ir
import firrtl.PrimOps

import java.security.MessageDigest
import scala.collection.mutable

/** This object can performs a "structural" hash over any firrtl Module.
  * It ignores:
  * - [firrtl.ir.Expression Expression] types
  * - Any [firrtl.ir.Info Info] fields
  * - Description on DescribedStmt
  * - each identifier name is replaced by a unique integer which only depends on the order of declaration
  *   and is thus deterministic
  * - Module names are ignored.
  *
  * Because of the way we "agnostify" bundle types, all SubField access nodes need to have a known
  * bundle type. Thus - in a lot of cases, like after reading firrtl from a file - you need to run
  * the firrtl type inference before hashing.
  *
  * Please note that module hashes don't include any submodules.
  * Two structurally equivalent modules are only functionally equivalent if they are part
  * of the same circuit and thus all modules referred to in DefInstance are the same.
  *
  * @author Kevin Laeufer <laeufer@cs.berkeley.edu>
  * */
object StructuralHash {
  def sha256(node: DefModule, moduleRename: String => String = identity): HashCode = {
    val m = MessageDigest.getInstance(SHA256)
    new StructuralHash(new MessageDigestHasher(m), moduleRename).hash(node)
    new MDHashCode(m.digest())
  }

  /** This includes the names of ports and any port bundle field names in the hash. */
  def sha256WithSignificantPortNames(module: DefModule, moduleRename: String => String = identity): HashCode = {
    val m = MessageDigest.getInstance(SHA256)
    hashModuleAndPortNames(module, new MessageDigestHasher(m), moduleRename)
    new MDHashCode(m.digest())
  }

  private[firrtl] def sha256(str: String): HashCode = {
    val m = MessageDigest.getInstance(SHA256)
    m.update(str.getBytes())
    new MDHashCode(m.digest())
  }

  /** Using this to hash arbitrary nodes can have unexpected results like:
    *   hash(`a <= 1`) == hash(`b <= 1`).
    * This method is package private to allow for unit testing but should not be exposed to the user.
    */
  private[firrtl] def sha256Node(node: FirrtlNode): HashCode = {
    val m = MessageDigest.getInstance(SHA256)
    hash(node, new MessageDigestHasher(m), identity)
    new MDHashCode(m.digest())
  }

  // see: https://docs.oracle.com/javase/7/docs/api/java/security/MessageDigest.html
  private val SHA256 = "SHA-256"

  //scalastyle:off cyclomatic.complexity
  private def hash(node: FirrtlNode, h: Hasher, rename: String => String): Unit = node match {
    case n : Expression => new StructuralHash(h, rename).hash(n)
    case n : Statement => new StructuralHash(h, rename).hash(n)
    case n : Type => new StructuralHash(h, rename).hash(n)
    case n : Width => new StructuralHash(h, rename).hash(n)
    case n : Orientation => new StructuralHash(h, rename).hash(n)
    case n : Field => new StructuralHash(h, rename).hash(n)
    case n : Direction => new StructuralHash(h, rename).hash(n)
    case n : Port => new StructuralHash(h, rename).hash(n)
    case n : Param => new StructuralHash(h, rename).hash(n)
    case _ : Info => throw new RuntimeException("The structural hash of Info is meaningless.")
    case n : DefModule => new StructuralHash(h, rename).hash(n)
    case n : Circuit => hashCircuit(n, h, rename)
    case n : StringLit => h.update(n.toString)
  }
  //scalastyle:on cyclomatic.complexity

  private def hashModuleAndPortNames(m: DefModule, h: Hasher, rename: String => String): Unit = {
    val sh = new StructuralHash(h, rename)
    sh.hash(m)
    // hash port names
    m.ports.foreach { p =>
      h.update(p.name)
      hashPortTypeName(p.tpe, h.update)
    }
  }

  private def hashPortTypeName(tpe: Type, h: String => Unit): Unit = tpe match {
    case BundleType(fields) => fields.foreach{ f => h(f.name) ; hashPortTypeName(f.tpe, h) }
    case VectorType(vt, _) => hashPortTypeName(vt, h)
    case _ => // ignore ground types since they do not have field names nor sub-types
  }

  private def hashCircuit(c: Circuit, h: Hasher, rename: String => String): Unit = {
    h.update(127)
    h.update(c.main)
    // sort modules to make hash more useful
    val mods = c.modules.sortBy(_.name)
    // we create a new StructuralHash for each module since each module has its own namespace
    mods.foreach { m =>
      new StructuralHash(h, rename).hash(m)
    }
  }

  private val primOpToId = PrimOps.builtinPrimOps.zipWithIndex.map{ case (op, i) => op -> (-i -1).toByte }.toMap
  assert(primOpToId.values.max == -1,  "PrimOp nodes use ids -1 ... -50")
  assert(primOpToId.values.min >= -50, "PrimOp nodes use ids -1 ... -50")
  private def primOp(p: PrimOp): Byte = primOpToId(p)

  // verification ops are not firrtl nodes and thus not part of the same id namespace
  private def verificationOp(op: Formal.Value): Byte = op match {
    case Formal.Assert => 0
    case Formal.Assume => 1
    case Formal.Cover => 2
  }
}

trait HashCode {
  protected val str: String
  override def hashCode(): Int = str.hashCode
  override def equals(obj: Any): Boolean = obj match {
    case hashCode: HashCode => this.str.equals(hashCode.str)
    case _ => false
  }
}

private class MDHashCode(code: Array[Byte]) extends HashCode {
  protected override val str: String = code.map(b => f"${b.toInt & 0xff}%02x").mkString("")
}

/** Generic hashing interface which allows us to use different backends to trade of speed and collision resistance */
private trait Hasher {
  def update(b: Byte): Unit
  def update(i: Int): Unit
  def update(l: Long): Unit
  def update(s: String): Unit
  def update(b: Array[Byte]): Unit
  def update(d: Double): Unit = update(java.lang.Double.doubleToRawLongBits(d))
  def update(i: BigInt): Unit = update(i.toByteArray)
  def update(b: Boolean): Unit = if(b) update(1.toByte) else update(0.toByte)
  def update(i: BigDecimal): Unit = {
    // this might be broken, tried to borrow some code from BigDecimal.computeHashCode
    val temp = i.bigDecimal.stripTrailingZeros()
    val bigInt = temp.scaleByPowerOfTen(temp.scale).toBigInteger
    update(bigInt)
    update(temp.scale)
  }
}

private class MessageDigestHasher(m: MessageDigest) extends Hasher {
  override def update(b: Byte): Unit = m.update(b)
  override def update(i: Int): Unit = {
    m.update(((i >>  0) & 0xff).toByte)
    m.update(((i >>  8) & 0xff).toByte)
    m.update(((i >> 16) & 0xff).toByte)
    m.update(((i >> 24) & 0xff).toByte)
  }
  override def update(l: Long): Unit = {
    m.update(((l >>  0) & 0xff).toByte)
    m.update(((l >>  8) & 0xff).toByte)
    m.update(((l >> 16) & 0xff).toByte)
    m.update(((l >> 24) & 0xff).toByte)
    m.update(((l >> 32) & 0xff).toByte)
    m.update(((l >> 40) & 0xff).toByte)
    m.update(((l >> 48) & 0xff).toByte)
    m.update(((l >> 56) & 0xff).toByte)
  }
  // the encoding of the bytes should not matter as long as we are on the same platform
  override def update(s: String): Unit = m.update(s.getBytes())
  override def update(b: Array[Byte]): Unit = m.update(b)
}

class StructuralHash private(h: Hasher, renameModule: String => String) {
  // replace identifiers with incrementing integers
  private val nameToInt = mutable.HashMap[String, Int]()
  private var nameCounter: Int = 0
  @inline private def n(name: String): Unit = hash(nameToInt.getOrElseUpdate(name, {
    val ii = nameCounter
    nameCounter = nameCounter + 1
    ii
  }))

  // internal convenience methods
  @inline private def id(b: Byte): Unit = h.update(b)
  @inline private def hash(i: Int): Unit = h.update(i)
  @inline private def hash(b: Boolean): Unit = h.update(b)
  @inline private def hash(d: Double): Unit = h.update(d)
  @inline private def hash(i: BigInt): Unit = h.update(i)
  @inline private def hash(i: BigDecimal): Unit = h.update(i)
  @inline private def hash(s: String): Unit = h.update(s)

  //scalastyle:off magic.number
  //scalastyle:off cyclomatic.complexity
  private def hash(node: Expression): Unit = node match {
    case Reference(name, _, _, _) => id(0) ; n(name)
    case DoPrim(op, args, consts, _) =>
      // no need to hash the number of arguments or constants since that is implied by the op
      id(1) ; h.update(StructuralHash.primOp(op)) ; args.foreach(hash) ; consts.foreach(hash)
    case UIntLiteral(value, width) => id(2) ; hash(value) ; hash(width)
    // We hash bundles as if fields are accessed by their index.
    // Thus we need to also hash field accesses that way.
    // This has the side-effect that `x.y` might hash to the same value as `z.r`, for example if the
    // types are `x: {y: UInt<1>, ...}` and `z: {r: UInt<1>, ...}` respectively.
    // They do not hash to the same value if the type of `z` is e.g., `z: {..., r: UInt<1>, ...}`
    // as that would have the `r` field at a different index.
    case SubField(expr, name, _, _) => id(3) ; hash(expr)
      // find field index and hash that instead of the field name
      val fields = expr.tpe match {
        case b: BundleType => b.fields
        case other =>
          throw new RuntimeException(s"Unexpected type $other for SubField access. Did you run the type checker?")
      }
      val index = fields.zipWithIndex.find(_._1.name == name).map(_._2).get
      hash(index)
    case SubIndex(expr, value, _, _) => id(4) ; hash(expr) ; hash(value)
    case SubAccess(expr, index, _, _) => id(5) ; hash(expr) ; hash(index)
    case Mux(cond, tval, fval, _) => id(6) ; hash(cond) ; hash(tval) ; hash(fval)
    case ValidIf(cond, value, _) => id(7) ; hash(cond) ; hash(value)
    case SIntLiteral(value, width) => id(8) ; hash(value) ; hash(width)
    case FixedLiteral(value, width, point) => id(9) ; hash(value) ; hash(width) ; hash(point)
    // WIR
    case firrtl.WVoid => id(10)
    case firrtl.WInvalid => id(11)
    case firrtl.EmptyExpression => id(12)
    // VRandom is used in the Emitter
    case firrtl.VRandom(width) => id(13) ;  hash(width)
    // ids 14 ... 19 are reserved for future Expression nodes
  }
  //scalastyle:on cyclomatic.complexity

  //scalastyle:off cyclomatic.complexity method.length
  private def hash(node: Statement): Unit = node match {
    // all info fields are ignore
    case DefNode(_, name, value) => id(20) ; n(name) ; hash(value)
    case Connect(_, loc, expr) => id(21) ; hash(loc) ; hash(expr)
    // we place the unique id 23 between conseq and alt to distinguish between them in case conseq is empty
    // we place the unique id 24 after alt to distinguish between alt and the next statement in case alt is empty
    case Conditionally(_, pred, conseq, alt) => id(22) ; hash(pred) ; hash(conseq) ; id(23) ; hash(alt) ; id(24)
    case EmptyStmt => // empty statements are ignored
    case Block(stmts) => stmts.foreach(hash) // block structure is ignored
    case Stop(_, ret, clk, en) => id(25) ; hash(ret) ; hash(clk) ; hash(en)
    case Print(_, string, args, clk, en) =>
      // the string is part of the side effect and thus part of the circuit behavior
      id(26) ; hash(string.string) ; hash(args.length) ; args.foreach(hash) ; hash(clk) ; hash(en)
    case IsInvalid(_, expr) => id(27) ; hash(expr)
    case DefWire(_, name, tpe) => id(28) ; n(name) ; hash(tpe)
    case DefRegister(_, name, tpe, clock, reset, init) =>
      id(29) ; n(name) ; hash(tpe) ; hash(clock) ; hash(reset) ; hash(init)
    case DefInstance(_, name, module, _) =>
      // Module is in the global namespace which is why we cannot replace it with a numeric id.
      // However, it might have been renamed as part of the dedup consolidation.
      id(30) ; n(name) ; hash(renameModule(module))
    // descriptions on statements are ignores
    case firrtl.DescribedStmt(_, stmt) => hash(stmt)
    case DefMemory(_, name, dataType, depth, writeLatency, readLatency, readers, writers,
    readwriters, readUnderWrite) =>
      id(30) ; n(name) ;  hash(dataType) ;  hash(depth) ;  hash(writeLatency) ;  hash(readLatency)
      hash(readers.length) ; readers.foreach(hash)
      hash(writers.length) ; writers.foreach(hash)
      hash(readwriters.length) ; readwriters.foreach(hash)
      hash(readUnderWrite)
    case PartialConnect(_, loc, expr) => id(31) ; hash(loc) ; hash(expr)
    case Attach(_, exprs) => id(32) ; hash(exprs.length) ; exprs.foreach(hash)
    // WIR
    case firrtl.CDefMemory(_, name, tpe, size, seq, readUnderWrite) =>
      id(33) ; n(name) ; hash(tpe); hash(size) ; hash(seq) ; hash(readUnderWrite)
    case firrtl.CDefMPort(_, name, _, mem, exps, direction) =>
      // the type of the MPort depends only on the memory (in well types firrtl) and can thus be ignored
      id(34) ; n(name) ; n(mem) ; hash(exps.length) ; exps.foreach(hash) ; hash(direction)
    // DefAnnotatedMemory from MemIR.scala
    case firrtl.passes.memlib.DefAnnotatedMemory(_, name, dataType, depth, writeLatency, readLatency, readers, writers,
    readwriters, readUnderWrite, maskGran, memRef) =>
      id(35) ;  n(name) ;  hash(dataType) ;  hash(depth) ;  hash(writeLatency) ;  hash(readLatency)
      hash(readers.length) ; readers.foreach(hash)
      hash(writers.length) ; writers.foreach(hash)
      hash(readwriters.length) ; readwriters.foreach(hash)
      hash(readUnderWrite.toString)
      hash(maskGran.size) ; maskGran.foreach(hash)
      hash(memRef.size) ; memRef.foreach{ case (a, b) =>  hash(a) ;  hash(b) }
    case Verification(op, _, clk, pred, en, msg) =>
      id(36) ; hash(StructuralHash.verificationOp(op)) ; hash(clk) ; hash(pred) ; hash(en) ; hash(msg.string)
    // ids 37 ... 39 are reserved for future Statement nodes
  }
  //scalastyle:on cyclomatic.complexity method.length

  // ReadUnderWrite is never used in place of a FirrtlNode and thus we can start a new id namespace
  private def hash(ruw: ReadUnderWrite.Value): Unit = ruw match {
    case ReadUnderWrite.New => id(0)
    case ReadUnderWrite.Old => id(1)
    case ReadUnderWrite.Undefined => id(2)
  }

  private def hash(node: Width): Unit = node match {
    case IntWidth(width) => id(40) ; hash(width)
    case UnknownWidth => id(41)
    case CalcWidth(arg) => id(42) ; hash(arg)
    // we are hashing the name of the `VarWidth` instead of using `n` since these Vars exist in a different namespace
    case VarWidth(name) => id(43) ; hash(name)
    // ids 44 + 45 are reserved for future Width nodes
  }

  private def hash(node: Orientation): Unit = node match {
    case Default => id(46)
    case Flip => id(47)
  }

  private def hash(node: Field): Unit = {
    // since we are only interested in a structural hash, we ignore field names
    // this means that: hash(`{x : UInt<1>, y: UInt<2>}`) == hash(`{y : UInt<1>, x: UInt<2>}`)
    // but:             hash(`{x : UInt<1>, y: UInt<2>}`) != hash(`{y : UInt<2>, x: UInt<1>}`)
    // which seems strange, since the connect semantics rely on field names, but it is the behavior that
    // has been used in the Dedup pass for a long time.
    // This position-based notion of equality requires us to replace field names with field indexes when hashing
    // SubField accesses.
    id(48) ; hash(node.flip) ; hash(node.tpe)
  }

  //scalastyle:off cyclomatic.complexity
  private def hash(node: Type): Unit = node match {
    // Types
    case UIntType(width: Width) => id(50) ; hash(width)
    case SIntType(width: Width) => id(51) ; hash(width)
    case FixedType(width, point) => id(52) ; hash(width) ; hash(point)
    case BundleType(fields) => id(53) ; hash(fields.length) ; fields.foreach(hash)
    case VectorType(tpe, size) => id(54) ; hash(tpe) ; hash(size)
    case ClockType => id(55)
    case ResetType => id(56)
    case AsyncResetType => id(57)
    case AnalogType(width) => id(58) ; hash(width)
    case UnknownType => id(59)
    case IntervalType(lower, upper, point) => id(60) ; hash(lower) ;  hash(upper) ;  hash(point)
    // ids 61 ... 65 are reserved for future Type nodes
  }
  //scalastyle:on cyclomatic.complexity

  private def hash(node: Direction): Unit = node match {
    case Input => id(66)
    case Output => id(67)
  }

  private def hash(node: Port): Unit = {
    id(68) ; n(node.name) ; hash(node.direction) ; hash(node.tpe)
  }

  private def hash(node: Param): Unit = node match {
    case IntParam(name, value) => id(70) ; n(name) ; hash(value)
    case DoubleParam(name, value) => id(71) ; n(name) ; hash(value)
    case StringParam(name, value) => id(72) ; n(name) ; hash(value.string)
    case RawStringParam(name, value) => id(73) ; n(name) ; hash(value)
    // id 74 is reserved for future use
  }

  private def hash(node: DefModule): Unit = node match {
    // the module name is ignored since it does not affect module functionality
    case Module(_, _name, ports, body) =>
      id(75) ; hash(ports.length) ; ports.foreach(hash) ; hash(body)
    // the module name is ignored since it does not affect module functionality
    case ExtModule(_, name, ports, defname, params) =>
      id(76) ; hash(ports.length) ; ports.foreach(hash) ; hash(defname)
      hash(params.length) ; params.foreach(hash)
  }

  // id 127 is reserved for Circuit nodes

  private def hash(d: firrtl.MPortDir): Unit = d match {
    case firrtl.MInfer => id(-70)
    case firrtl.MRead => id(-71)
    case firrtl.MWrite => id(-72)
    case firrtl.MReadWrite => id(-73)
  }

  private def hash(c: firrtl.constraint.Constraint): Unit = c match {
    case b: Bound => hash(b) /* uses ids -80 ... -84 */
    case firrtl.constraint.IsAdd(known, maxs, mins, others) =>
      id(-85) ; hash(known.nonEmpty) ; known.foreach(hash)
      hash(maxs.length) ; maxs.foreach(hash)
      hash(mins.length) ; mins.foreach(hash)
      hash(others.length) ; others.foreach(hash)
    case firrtl.constraint.IsFloor(child, dummyArg) => id(-86) ; hash(child) ; hash(dummyArg)
    case firrtl.constraint.IsKnown(decimal) => id(-87) ; hash(decimal)
    case firrtl.constraint.IsNeg(child, dummyArg) => id(-88) ; hash(child) ; hash(dummyArg)
    case firrtl.constraint.IsPow(child, dummyArg) => id(-89) ; hash(child) ; hash(dummyArg)
    case firrtl.constraint.IsVar(str) => id(-90) ; n(str)
  }

  private def hash(b: Bound): Unit = b match {
    case UnknownBound => id(-80)
    case CalcBound(arg) => id(-81) ; hash(arg)
    // we are hashing the name of the `VarBound` instead of using `n` since these Vars exist in a different namespace
    case VarBound(name) => id(-82) ; hash(name)
    case Open(value) => id(-83) ; hash(value)
    case Closed(value) => id(-84) ; hash(value)
  }
}