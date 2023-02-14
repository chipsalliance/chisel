// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.ir._
import firrtl.PrimOps._
import firrtl.Mappers._
import firrtl.traversals.Foreachers._
import firrtl.WrappedExpression._

import scala.collection.mutable
import scala.util.matching.Regex

import Implicits.{constraint2width, width2constraint}
import firrtl.constraint.{IsMax, IsMin}
import firrtl.annotations.{ReferenceTarget, TargetToken}
import _root_.logger.LazyLogging

object seqCat {
  def apply(args: Seq[Expression]): Expression = args.length match {
    case 0 => Utils.error("Empty Seq passed to seqcat")
    case 1 => args.head
    case 2 => DoPrim(PrimOps.Cat, args, Nil, UIntType(UnknownWidth))
    case _ =>
      val (high, low) = args.splitAt(args.length / 2)
      DoPrim(PrimOps.Cat, Seq(seqCat(high), seqCat(low)), Nil, UIntType(UnknownWidth))
  }
}

object getWidth {
  def apply(t: Type): Width = t match {
    case t: GroundType => t.width
    case _ => Utils.error(s"No width: $t")
  }
  def apply(e: Expression): Width = apply(e.tpe)
}

/**
  * Helper object for computing the width of a firrtl type.
  */
object bitWidth {

  /**
    * Compute the width of a firrtl type.
    * For example, a Vec of 4 UInts of width 8 should have a width of 32.
    *
    * @param dt firrtl type
    * @return Width of the given type
    */
  def apply(dt:           Type): BigInt = widthOf(dt)
  private def widthOf(dt: Type): BigInt = dt match {
    case t: VectorType => t.size * bitWidth(t.tpe)
    case t: BundleType => t.fields.map(f => bitWidth(f.tpe)).foldLeft(BigInt(0))(_ + _)
    case GroundType(IntWidth(width)) => width
    case t                           => Utils.error(s"Unknown type encountered in bitWidth: $dt")
  }
}

object Utils extends LazyLogging {

  /** Unwind the causal chain until we hit the initial exception (which may be the first).
    *
    * @param maybeException - possible exception triggering the error,
    * @param first - true if we want the first (eldest) exception in the chain,
    * @return first or last Throwable in the chain.
    */
  def getThrowable(maybeException: Option[Throwable], first: Boolean): Throwable = {
    maybeException match {
      case Some(e: Throwable) => {
        val t = e.getCause
        if (t != null) {
          if (first) {
            getThrowable(Some(t), first)
          } else {
            t
          }
        } else {
          e
        }
      }
      case None | null => null
    }
  }

  /** Throw an internal error, possibly due to an exception.
    *
    * @param message - possible string to emit,
    * @param exception - possible exception triggering the error.
    */
  def throwInternalError(message: String = "", exception: Option[Throwable] = None) = {
    // We'll get the first exception in the chain, keeping it intact.
    val first = true
    val throwable = getThrowable(exception, true)
    val string = if (message.nonEmpty) message + "\n" else message
    error(
      "Internal Error! %sPlease file an issue at https://github.com/ucb-bar/firrtl/issues".format(string),
      throwable
    )
  }

  def time[R](block: => R): (Double, R) = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    val timeMillis = (t1 - t0) / 1000000.0
    (timeMillis, result)
  }

  /** Removes all [[firrtl.ir.EmptyStmt]] statements and condenses
    * [[firrtl.ir.Block]] statements.
    */
  def squashEmpty(s: Statement): Statement = s.map(squashEmpty) match {
    case Block(stmts) =>
      val newStmts = stmts.filter(_ != EmptyStmt)
      newStmts.size match {
        case 0 => EmptyStmt
        case 1 => newStmts.head
        case _ => Block(newStmts)
      }
    case sx => sx
  }

  /** Returns true if PrimOp is a BitExtraction, false otherwise */
  def isBitExtract(op: PrimOp): Boolean = op match {
    case Bits | Head | Tail | Shr => true
    case _                        => false
  }

  /** Returns true if Expression is a Bits PrimOp, false otherwise */
  def isBitExtract(expr: Expression): Boolean = expr match {
    case DoPrim(op, _, _, UIntType(_)) if isBitExtract(op) => true
    case _                                                 => false
  }

  /** Selects all the elements of this list ignoring the duplicates as determined by == after
    * applying the transforming function f
    *
    * @note In Scala Standard Library starting in 2.13
    */
  def distinctBy[A, B](xs: List[A])(f: A => B): List[A] = {
    val buf = new mutable.ListBuffer[A]
    val seen = new mutable.HashSet[B]
    for (x <- xs) {
      val y = f(x)
      if (!seen(y)) {
        buf += x
        seen += y
      }
    }
    buf.toList
  }

  /** Provide a nice name to create a temporary * */
  def niceName(e: Expression): String = niceName(1)(e)
  def niceName(depth: Int)(e: Expression): String = {
    e match {
      case Reference(name, _, _, _) if name(0) == '_' => name
      case Reference(name, _, _, _)                   => "_" + name
      case SubAccess(expr, index, _, _) if depth <= 0 => niceName(depth)(expr)
      case SubAccess(expr, index, _, _)               => niceName(depth)(expr) + niceName(depth - 1)(index)
      case SubField(expr, field, _, _)                => niceName(depth)(expr) + "_" + field
      case SubIndex(expr, index, _, _)                => niceName(depth)(expr) + "_" + index
      case DoPrim(op, args, consts, _) if depth <= 0  => "_" + op
      case DoPrim(op, args, consts, _)                => "_" + op + (args.map(niceName(depth - 1)) ++ consts.map("_" + _)).mkString("")
      case Mux(cond, tval, fval, _) if depth <= 0     => "_mux"
      case Mux(cond, tval, fval, _)                   => "_mux" + Seq(cond, tval, fval).map(niceName(depth - 1)).mkString("")
      case UIntLiteral(value, _)                      => "_" + value
      case SIntLiteral(value, _)                      => "_" + value
    }
  }

  /** Maps node name to value */
  type NodeMap = mutable.HashMap[String, Expression]

  def isTemp(str: String): Boolean = str.head == '_'

  implicit def toWrappedExpression(x: Expression): WrappedExpression = new WrappedExpression(x)
  def getSIntWidth(s:                 BigInt):     Int = s.bitLength + 1
  def getUIntWidth(u:                 BigInt):     Int = u.bitLength
  def dec2string(v:                   BigDecimal): String = v.underlying().stripTrailingZeros().toPlainString
  def trim(v:                         BigDecimal): BigDecimal = BigDecimal(dec2string(v))
  def max(a:                          BigInt, b: BigInt): BigInt = if (a >= b) a else b
  def min(a:                          BigInt, b:   BigInt): BigInt = if (a >= b) b else a
  def pow_minus_one(a:                BigInt, b:   BigInt): BigInt = a.pow(b.toInt) - 1
  val BoolType = UIntType(IntWidth(1))
  val one = UIntLiteral(1)
  val zero = UIntLiteral(0)

  private val ClockZero = DoPrim(PrimOps.AsClock, Seq(zero), Seq.empty, ClockType)
  private val AsyncZero = DoPrim(PrimOps.AsAsyncReset, Seq(zero), Nil, AsyncResetType)

  /** Returns an [[firrtl.ir.Expression Expression]] equal to zero for a given [[firrtl.ir.GroundType GroundType]]
    * @note Does not support [[firrtl.ir.AnalogType AnalogType]].
    */
  def getGroundZero(tpe: GroundType): Expression = tpe match {
    case u: UIntType => UIntLiteral(0, u.width)
    case s: SIntType => SIntLiteral(0, s.width)
    // Default reset type is Bool
    case ResetType      => Utils.zero
    case ClockType      => ClockZero
    case AsyncResetType => AsyncZero
    case other          => throwInternalError(s"Unexpected type $other")
  }

  def create_exps(n: String, t: Type): Seq[Expression] =
    create_exps(WRef(n, t, ExpKind, UnknownFlow))
  def create_exps(e: Expression): Seq[Expression] = e match {
    case ex: Mux =>
      val e1s = create_exps(ex.tval)
      val e2s = create_exps(ex.fval)
      e1s.zip(e2s).map {
        case (e1, e2) =>
          Mux(ex.cond, e1, e2, mux_type_and_widths(e1, e2))
      }
    case ex: ValidIf => create_exps(ex.value).map(e1 => ValidIf(ex.cond, e1, e1.tpe))
    case ex =>
      ex.tpe match {
        case (_: GroundType) => Seq(ex)
        case t:  BundleType =>
          t.fields.flatMap(f => create_exps(WSubField(ex, f.name, f.tpe, times(flow(ex), f.flip))))
        case t: VectorType => (0 until t.size).flatMap(i => create_exps(WSubIndex(ex, i, t.tpe, flow(ex))))
      }
  }

  /** Like create_exps, but returns intermediate Expressions as well
    * @param e
    * @return
    */
  def expandRef(e: Expression): Seq[Expression] = e match {
    case ex: Mux =>
      val e1s = expandRef(ex.tval)
      val e2s = expandRef(ex.fval)
      e1s.zip(e2s).map {
        case (e1, e2) =>
          Mux(ex.cond, e1, e2, mux_type_and_widths(e1, e2))
      }
    case ex: ValidIf => expandRef(ex.value).map(e1 => ValidIf(ex.cond, e1, e1.tpe))
    case ex =>
      ex.tpe match {
        case (_: GroundType) => Seq(ex)
        case (t: BundleType) =>
          ex +: t.fields.flatMap(f => expandRef(WSubField(ex, f.name, f.tpe, times(flow(ex), f.flip))))
        case (t: VectorType) =>
          ex +: (0 until t.size).flatMap(i => expandRef(WSubIndex(ex, i, t.tpe, flow(ex))))
      }
  }
  def toTarget(main: String, module: String)(expression: Expression): ReferenceTarget = {
    val tokens = mutable.ArrayBuffer[TargetToken]()
    var ref = "???"
    def onExp(expr: Expression): Expression = {
      expr.map(onExp) match {
        case e: Reference => ref = e.name
        case e: SubField  => tokens += TargetToken.Field(e.name)
        case e: SubIndex  => tokens += TargetToken.Index(e.value)
        case other => throwInternalError("Cannot call Utils.toTarget on non-referencing expression")
      }
      expr
    }
    onExp(expression)
    ReferenceTarget(main, module, Nil, ref, tokens.toSeq)
  }

  def get_point(e: Expression): Int = e match {
    case (e: WRef) => 0
    case (e: WSubField) =>
      e.expr.tpe match {
        case b: BundleType =>
          (b.fields.takeWhile(_.name != e.name).foldLeft(0))((point, f) => point + get_size(f.tpe))
      }
    case (e: WSubIndex)  => e.value * get_size(e.tpe)
    case (e: WSubAccess) => get_point(e.expr)
  }

  /** Returns true if t, or any subtype, contains a flipped field
    *
    * @param t type [[firrtl.ir.Type]] to be checked
    * @return if t contains [[firrtl.ir.Flip]]
    */
  def hasFlip(t: Type): Boolean = t match {
    case t: BundleType =>
      (t.fields.exists(_.flip == Flip)) ||
        (t.fields.exists(f => hasFlip(f.tpe)))
    case t: VectorType => hasFlip(t.tpe)
    case _ => false
  }

  /** Returns children Expressions of e */
  def getKids(e: Expression): Seq[Expression] = {
    val kids = mutable.ArrayBuffer[Expression]()
    def addKids(e: Expression): Expression = {
      kids += e
      e
    }
    e.map(addKids)
    kids.toSeq
  }

  /** Walks two expression trees and returns a sequence of tuples of where they differ */
  def diff(e1: Expression, e2: Expression): Seq[(Expression, Expression)] = {
    if (weq(e1, e2)) Nil
    else {
      val (e1Kids, e2Kids) = (getKids(e1), getKids(e2))

      if (e1Kids == Nil || e2Kids == Nil || e1Kids.size != e2Kids.size) Seq((e1, e2))
      else {
        e1Kids.zip(e2Kids).flatMap { case (e1k, e2k) => diff(e1k, e2k) }
      }
    }
  }

  /** Returns an inlined expression (replacing node references with values),
    * stopping on a stopping condition or until the reference is not a node
    */
  def inline(nodeMap: NodeMap, stop: String => Boolean = { x: String => false })(e: Expression): Expression = {
    def onExp(e: Expression): Expression = e.map(onExp) match {
      case Reference(name, _, _, _) if nodeMap.contains(name) && !stop(name) => onExp(nodeMap(name))
      case other                                                             => other
    }
    onExp(e)
  }

  def mux_type(e1: Expression, e2: Expression): Type = mux_type(e1.tpe, e2.tpe)
  def mux_type(t1: Type, t2:       Type): Type = (t1, t2) match {
    case (ClockType, ClockType)           => ClockType
    case (AsyncResetType, AsyncResetType) => AsyncResetType
    case (t1: UIntType, t2: UIntType) => UIntType(UnknownWidth)
    case (t1: SIntType, t2: SIntType) => SIntType(UnknownWidth)
    case (t1: VectorType, t2: VectorType) => VectorType(mux_type(t1.tpe, t2.tpe), t1.size)
    case (t1: BundleType, t2: BundleType) =>
      BundleType(t1.fields.zip(t2.fields).map {
        case (f1, f2) => Field(f1.name, f1.flip, mux_type(f1.tpe, f2.tpe))
      })
    case _ => UnknownType
  }
  def mux_type_and_widths(e1: Expression, e2: Expression): Type =
    mux_type_and_widths(e1.tpe, e2.tpe)
  def mux_type_and_widths(t1: Type, t2: Type): Type = {
    def wmax(w1: Width, w2: Width): Width = (w1, w2) match {
      case (w1x: IntWidth, w2x: IntWidth) => IntWidth(w1x.width.max(w2x.width))
      case (w1x, w2x) => IsMax(w1x, w2x)
    }
    (t1, t2) match {
      case (ClockType, ClockType)           => ClockType
      case (AsyncResetType, AsyncResetType) => AsyncResetType
      case (t1x: UIntType, t2x: UIntType) => UIntType(IsMax(t1x.width, t2x.width))
      case (t1x: SIntType, t2x: SIntType) => SIntType(IsMax(t1x.width, t2x.width))
      case (t1x: VectorType, t2x: VectorType) => VectorType(mux_type_and_widths(t1x.tpe, t2x.tpe), t1x.size)
      case (t1x: BundleType, t2x: BundleType) =>
        BundleType(t1x.fields.zip(t2x.fields).map {
          case (f1, f2) => Field(f1.name, f1.flip, mux_type_and_widths(f1.tpe, f2.tpe))
        })
      case _ => UnknownType
    }
  }

  def module_type(m: DefModule): BundleType = BundleType(m.ports.map {
    case Port(_, name, dir, tpe) => Field(name, to_flip(dir), tpe)
  })
  def sub_type(v: Type): Type = v match {
    case vx: VectorType => vx.tpe
    case vx => UnknownType
  }
  def field_type(v: Type, s: String): Type = v match {
    case vx: BundleType =>
      vx.fields.find(_.name == s) match {
        case Some(f) => f.tpe
        case None    => UnknownType
      }
    case vx => UnknownType
  }

// =================================
  def error(str: String, cause: Throwable = null) = throw new FirrtlInternalException(str, cause)

//// =============== EXPANSION FUNCTIONS ================
  def get_size(t: Type): Int = t match {
    case tx: BundleType => (tx.fields.foldLeft(0))((sum, f) => sum + get_size(f.tpe))
    case tx: VectorType => tx.size * get_size(tx.tpe)
    case tx => 1
  }

  def get_valid_points(t1: Type, t2: Type, flip1: Orientation, flip2: Orientation): Seq[(Int, Int)] = {
    import passes.CheckTypes.legalResetType
    //;println_all(["Inside with t1:" t1 ",t2:" t2 ",f1:" flip1 ",f2:" flip2])
    (t1, t2) match {
      case (_: UIntType, _: UIntType) => if (flip1 == flip2) Seq((0, 0)) else Nil
      case (_: SIntType, _: SIntType) => if (flip1 == flip2) Seq((0, 0)) else Nil
      case (_: AnalogType, _: AnalogType) => if (flip1 == flip2) Seq((0, 0)) else Nil
      case (t1x: BundleType, t2x: BundleType) =>
        def emptyMap = Map[String, (Type, Orientation, Int)]()
        val t1_fields = t1x.fields
          .foldLeft((emptyMap, 0)) {
            case ((map, ilen), f1) =>
              (map + (f1.name -> ((f1.tpe, f1.flip, ilen))), ilen + get_size(f1.tpe))
          }
          ._1
        t2x.fields
          .foldLeft((Seq[(Int, Int)](), 0)) {
            case ((points, jlen), f2) =>
              t1_fields.get(f2.name) match {
                case None => (points, jlen + get_size(f2.tpe))
                case Some((f1_tpe, f1_flip, ilen)) =>
                  val f1_times = times(flip1, f1_flip)
                  val f2_times = times(flip2, f2.flip)
                  val ls = get_valid_points(f1_tpe, f2.tpe, f1_times, f2_times)
                  (points ++ (ls.map { case (x, y) => (x + ilen, y + jlen) }), jlen + get_size(f2.tpe))
              }
          }
          ._1
      case (t1x: VectorType, t2x: VectorType) =>
        val size = math.min(t1x.size, t2x.size)
        (0 until size)
          .foldLeft((Seq[(Int, Int)](), 0, 0)) {
            case ((points, ilen, jlen), _) =>
              val ls = get_valid_points(t1x.tpe, t2x.tpe, flip1, flip2)
              (
                points ++ (ls.map { case (x, y) => (x + ilen, y + jlen) }),
                ilen + get_size(t1x.tpe),
                jlen + get_size(t2x.tpe)
              )
          }
          ._1
      case (ClockType, ClockType)           => if (flip1 == flip2) Seq((0, 0)) else Nil
      case (AsyncResetType, AsyncResetType) => if (flip1 == flip2) Seq((0, 0)) else Nil
      // The following two cases handle driving ResetType from other legal reset types
      // Flippedness is important here because ResetType can be driven by other reset types, but it
      //   cannot *drive* other reset types
      case (ResetType, other) =>
        if (legalResetType(other) && flip1 == Default && flip1 == flip2) Seq((0, 0)) else Nil
      case (other, ResetType) =>
        if (legalResetType(other) && flip1 == Flip && flip1 == flip2) Seq((0, 0)) else Nil
      case _ => throwInternalError(s"get_valid_points: shouldn't be here - ($t1, $t2)")
    }
  }

// =========== FLOW/FLIP UTILS ============
  def swap(g: Flow): Flow = g match {
    case UnknownFlow => UnknownFlow
    case SourceFlow  => SinkFlow
    case SinkFlow    => SourceFlow
    case DuplexFlow  => DuplexFlow
  }
  def swap(d: Direction): Direction = d match {
    case Output => Input
    case Input  => Output
  }
  def swap(f: Orientation): Orientation = f match {
    case Default => Flip
    case Flip    => Default
  }
  // Input  <-> SourceFlow <-> Flip
  // Output <-> SinkFlow   <-> Default
  def to_dir(g: Flow): Direction = g match {
    case SourceFlow => Input
    case SinkFlow   => Output
  }
  def to_dir(o: Orientation): Direction = o match {
    case Flip    => Input
    case Default => Output
  }
  def to_flow(d: Direction): Flow = d match {
    case Input  => SourceFlow
    case Output => SinkFlow
  }
  def to_flip(d: Direction): Orientation = d match {
    case Input  => Flip
    case Output => Default
  }
  def to_flip(g: Flow): Orientation = g match {
    case SourceFlow => Flip
    case SinkFlow   => Default
  }

  def field_flip(v: Type, s: String): Orientation = v match {
    case vx: BundleType =>
      vx.fields.find(_.name == s) match {
        case Some(ft) => ft.flip
        case None     => Default
      }
    case vx => Default
  }
  def get_field(v: Type, s: String): Field = v match {
    case vx: BundleType =>
      vx.fields.find(_.name == s) match {
        case Some(ft) => ft
        case None     => throwInternalError(s"get_field: shouldn't be here - $v.$s")
      }
    case vx => throwInternalError(s"get_field: shouldn't be here - $v")
  }

  def times(d: Direction, flip: Orientation): Direction = flip match {
    case Default => d
    case Flip    => swap(d)
  }
  def times(g: Flow, d:      Direction): Direction = times(d, g)
  def times(d: Direction, g: Flow): Direction = g match {
    case SinkFlow   => d
    case SourceFlow => swap(d) // SourceFlow == INPUT == REVERSE
  }

  def times(g:    Flow, flip:     Orientation): Flow = times(flip, g)
  def times(flip: Orientation, g: Flow): Flow = flip match {
    case Default => g
    case Flip    => swap(g)
  }
  def times(f1: Orientation, f2: Orientation): Orientation = f2 match {
    case Default => f1
    case Flip    => swap(f1)
  }

// =========== ACCESSORS =========
  def kind(e: Expression): Kind = e match {
    case ex: WRef       => ex.kind
    case ex: WSubField  => kind(ex.expr)
    case ex: WSubIndex  => kind(ex.expr)
    case ex: WSubAccess => kind(ex.expr)
    case ex => ExpKind
  }
  def flow(e: Expression): Flow = e match {
    case ex: WRef        => ex.flow
    case ex: WSubField   => ex.flow
    case ex: WSubIndex   => ex.flow
    case ex: WSubAccess  => ex.flow
    case ex: DoPrim      => SourceFlow
    case ex: UIntLiteral => SourceFlow
    case ex: SIntLiteral => SourceFlow
    case ex: Mux         => SourceFlow
    case ex: ValidIf     => SourceFlow
    case WInvalid => SourceFlow
    case ex       => throwInternalError(s"flow: shouldn't be here - $e")
  }
  def get_flow(s: Statement): Flow = s match {
    case sx: DefWire        => DuplexFlow
    case sx: DefRegister    => DuplexFlow
    case sx: DefNode        => SourceFlow
    case sx: DefInstance    => SourceFlow
    case sx: DefMemory      => SourceFlow
    case sx: Block          => UnknownFlow
    case sx: Connect        => UnknownFlow
    case sx: PartialConnect => UnknownFlow
    case sx: Stop           => UnknownFlow
    case sx: Print          => UnknownFlow
    case sx: IsInvalid      => UnknownFlow
    case EmptyStmt => UnknownFlow
  }
  def get_flow(p: Port): Flow = if (p.direction == Input) SourceFlow else SinkFlow
  def get_info(s: Statement): Info = s match {
    case s: HasInfo => s.info
    case _ => NoInfo
  }

  /** Finds all root References in a nested Expression */
  def getAllRefs(expr: Expression): Seq[Reference] = {
    val refs = mutable.ListBuffer.empty[Reference]
    def rec(e: Expression): Unit = {
      e match {
        case ref: Reference => refs += ref
        case other => other.foreach(rec)
      }
    }
    rec(expr)
    refs.toList
  }

  /** Splits an Expression into root Ref and tail
    *
    * @example
    *   Given:   SubField(SubIndex(SubField(Ref("a", UIntType(IntWidth(32))), "b"), 2), "c")
    *   Returns: (Ref("a"), SubField(SubIndex(Ref("b"), 2), "c"))
    *   a.b[2].c -> (a, b[2].c)
    * @example
    *   Given:   SubField(SubIndex(Ref("b"), 2), "c")
    *   Returns: (Ref("b"), SubField(SubIndex(EmptyExpression, 2), "c"))
    *   b[2].c -> (b, EMPTY[2].c)
    * @note This function only supports [[firrtl.ir.RefLikeExpression RefLikeExpression]]s: [[firrtl.ir.Reference
    * Reference]], [[firrtl.ir.SubField SubField]], [[firrtl.ir.SubIndex SubIndex]], and [[firrtl.ir.SubAccess
    * SubAccess]]
    */
  def splitRef(e: Expression): (WRef, Expression) = e match {
    case e: Reference => (e, EmptyExpression)
    case e: SubIndex =>
      val (root, tail) = splitRef(e.expr)
      (root, SubIndex(tail, e.value, e.tpe, e.flow))
    case e: SubField =>
      val (root, tail) = splitRef(e.expr)
      tail match {
        case EmptyExpression => (root, Reference(e.name, e.tpe, root.kind, e.flow))
        case exp             => (root, SubField(tail, e.name, e.tpe, e.flow))
      }
    case e: SubAccess =>
      val (root, tail) = splitRef(e.expr)
      (root, SubAccess(tail, e.index, e.tpe, e.flow))
  }

  /** Adds a root reference to some SubField/SubIndex chain */
  def mergeRef(root: Expression, body: Expression): Expression = body match {
    case e: WRef =>
      WSubField(root, e.name, e.tpe, e.flow)
    case e: WSubIndex =>
      WSubIndex(mergeRef(root, e.expr), e.value, e.tpe, e.flow)
    case e: WSubField =>
      WSubField(mergeRef(root, e.expr), e.name, e.tpe, e.flow)
    case EmptyExpression => root
  }

  case class DeclarationNotFoundException(msg: String) extends FirrtlUserException(msg)

  /** Gets the root declaration of an expression
    *
    * @param m    the [[firrtl.ir.Module]] to search
    * @param expr the [[firrtl.ir.Expression]] that refers to some declaration
    * @return the [[firrtl.ir.IsDeclaration]] of `expr`
    * @throws DeclarationNotFoundException if no declaration of `expr` is found
    */
  def getDeclaration(m: Module, expr: Expression): IsDeclaration = {
    def getRootDecl(name: String)(s: Statement): Option[IsDeclaration] = s match {
      case decl: IsDeclaration => if (decl.name == name) Some(decl) else None
      case c:    Conditionally =>
        val m = (getRootDecl(name)(c.conseq), getRootDecl(name)(c.alt))
        (m: @unchecked) match {
          case (Some(decl), None) => Some(decl)
          case (None, Some(decl)) => Some(decl)
          case (None, None)       => None
        }
      case begin: Block =>
        val stmts = begin.stmts.flatMap(getRootDecl(name)) // can we short circuit?
        if (stmts.nonEmpty) Some(stmts.head) else None
      case _ => None
    }
    expr match {
      case (_: WRef | _: WSubIndex | _: WSubField) =>
        val (root, tail) = splitRef(expr)
        val rootDecl = m.ports.find(_.name == root.name) match {
          case Some(decl) => decl
          case None =>
            getRootDecl(root.name)(m.body) match {
              case Some(decl) => decl
              case None =>
                throw new DeclarationNotFoundException(s"[module ${m.name}]  Reference ${expr.serialize} not declared!")
            }
        }
        rootDecl
      case e => Utils.error(s"getDeclaration does not support Expressions of type ${e.getClass}")
    }
  }

  /** Similar to Seq.groupBy except that it preserves ordering of elements within each group */
  def groupByIntoSeq[A, K](xs: Iterable[A])(f: A => K): Seq[(K, Seq[A])] = {
    val map = mutable.LinkedHashMap.empty[K, mutable.ListBuffer[A]]
    for (x <- xs) {
      val key = f(x)
      val l = map.getOrElseUpdate(key, mutable.ListBuffer.empty[A])
      l += x
    }
    map.view.map({ case (k, vs) => k -> vs.toList }).toList
  }

  // For a given module, returns a Seq of all instantiated modules inside of it
  private[firrtl] def collectInstantiatedModules(mod: Module, map: Map[String, DefModule]): Seq[DefModule] = {
    // Use list instead of set to maintain order
    val modules = mutable.ArrayBuffer.empty[DefModule]
    def onStmt(stmt: Statement): Unit = stmt match {
      case DefInstance(_, _, name, _) => modules += map(name)
      case _: WDefInstanceConnector => throwInternalError(s"unrecognized statement: $stmt")
      case other => other.foreach(onStmt)
    }
    onStmt(mod.body)
    modules.distinct.toSeq
  }

  /** Checks if two circuits are equal regardless of their ordering of module definitions */
  def orderAgnosticEquality(a: Circuit, b: Circuit): Boolean =
    a.copy(modules = a.modules.sortBy(_.name)) == b.copy(modules = b.modules.sortBy(_.name))

  /** Combines several separate circuit modules (typically emitted by -e or -p compiler options) into a single circuit */
  def combine(circuits: Seq[Circuit]): Circuit = {
    def dedup(modules: Seq[DefModule]): Seq[Either[Module, DefModule]] = {
      // Left means module with no ExtModules, Right means child modules or lone ExtModules
      val module: Option[Module] = {
        val found: Seq[Module] = modules.collect { case m: Module => m }
        assert(
          found.size <= 1,
          s"Module definitions should have unique names, found ${found.size} definitions named ${found.head.name}"
        )
        found.headOption
      }
      val extModules: Seq[ExtModule] = modules.collect { case e: ExtModule => e }.distinct

      // If the module is a lone module (no extmodule references in any other file)
      if (extModules.isEmpty && !module.isEmpty)
        Seq(Left(module.get))
      // If a module has extmodules, but no other file contains the implementation
      else if (!extModules.isEmpty && module.isEmpty)
        extModules.map(Right(_))
      // Otherwise there is a module implementation with extmodule references
      else
        Seq(Right(module.get))
    }

    // 1. Combine modules
    val grouped: Seq[(String, Seq[DefModule])] = groupByIntoSeq(circuits.flatMap(_.modules))({
      case mod: Module    => mod.name
      case ext: ExtModule => ext.defname
    })
    val deduped: Iterable[Either[Module, DefModule]] = grouped.flatMap { case (_, insts) => dedup(insts) }

    // 2. Determine top
    val top = {
      val found = deduped.collect { case Left(m) => m }
      assert(found.size == 1, s"There should only be 1 top module, got: ${found.map(_.name).mkString(", ")}")
      found.head
    }
    val res = deduped.collect { case Right(m) => m }
    ir.Circuit(NoInfo, top +: res.toSeq, top.name)
  }

}
