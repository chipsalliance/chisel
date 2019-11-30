// See LICENSE for license details.

package firrtl

import firrtl.ir._
import firrtl.PrimOps._
import firrtl.Mappers._
import firrtl.WrappedExpression._

import scala.collection.mutable
import scala.util.matching.Regex

import Implicits.{constraint2bound, constraint2width, width2constraint}
import firrtl.constraint.{IsMax, IsMin}
import firrtl.annotations.{ReferenceTarget, TargetToken}
import _root_.logger.LazyLogging

object seqCat {
  def apply(args: Seq[Expression]): Expression = args.length match {
    case 0 => Utils.error("Empty Seq passed to seqcat")
    case 1 => args.head
    case 2 => DoPrim(PrimOps.Cat, args, Nil, UIntType(UnknownWidth))
    case _ =>
      val (high, low) = args splitAt (args.length / 2)
      DoPrim(PrimOps.Cat, Seq(seqCat(high), seqCat(low)), Nil, UIntType(UnknownWidth))
  }
}

/** Given an expression, return an expression consisting of all sub-expressions
 * concatenated (or flattened).
 */
object toBits {
  def apply(e: Expression): Expression = e match {
    case ex @ (_: WRef | _: WSubField | _: WSubIndex) => hiercat(ex)
    case t => Utils.error(s"Invalid operand expression for toBits: $e")
  }
  private def hiercat(e: Expression): Expression = e.tpe match {
    case t: VectorType => seqCat((0 until t.size).reverse map (i =>
      hiercat(WSubIndex(e, i, t.tpe, UnknownFlow))))
    case t: BundleType => seqCat(t.fields map (f =>
      hiercat(WSubField(e, f.name, f.tpe, UnknownFlow))))
    case t: GroundType => DoPrim(AsUInt, Seq(e), Seq.empty, UnknownType)
    case t => Utils.error(s"Unknown type encountered in toBits: $e")
  }
}

object getWidth {
  def apply(t: Type): Width = t match {
    case t: GroundType => t.width
    case _ => Utils.error(s"No width: $t")
  }
  def apply(e: Expression): Width = apply(e.tpe)
}

object bitWidth {
  def apply(dt: Type): BigInt = widthOf(dt)
  private def widthOf(dt: Type): BigInt = dt match {
    case t: VectorType => t.size * bitWidth(t.tpe)
    case t: BundleType => t.fields.map(f => bitWidth(f.tpe)).foldLeft(BigInt(0))(_+_)
    case GroundType(IntWidth(width)) => width
    case t => Utils.error(s"Unknown type encountered in bitWidth: $dt")
  }
}

object castRhs {
  def apply(lhst: Type, rhs: Expression) = {
    (lhst, rhs.tpe) match {
      case (x: GroundType, y: GroundType) if WrappedType(x) == WrappedType(y) =>
        rhs
      case (_: SIntType, _) =>
        DoPrim(AsSInt, Seq(rhs), Seq.empty, lhst)
      case (FixedType(_, IntWidth(p)), _) =>
        DoPrim(AsFixedPoint, Seq(rhs), Seq(p), lhst)
      case (ClockType, _) =>
        DoPrim(AsClock, Seq(rhs), Seq.empty, lhst)
      case (_: UIntType, _) =>
        DoPrim(AsUInt, Seq(rhs), Seq.empty, lhst)
      case (_, _) => Utils.error("castRhs lhst, rhs type combination is invalid")
    }
  }
}

object fromBits {
  def apply(lhs: Expression, rhs: Expression): Statement = {
    val fbits = lhs match {
      case ex @ (_: WRef | _: WSubField | _: WSubIndex) => getPart(ex, ex.tpe, rhs, 0)
      case _ => Utils.error("Invalid LHS expression for fromBits!")
    }
    Block(fbits._2)
  }
  private def getPartGround(lhs: Expression,
                            lhst: Type,
                            rhs: Expression,
                            offset: BigInt): (BigInt, Seq[Statement]) = {
    val intWidth = bitWidth(lhst)
    val sel = DoPrim(PrimOps.Bits, Seq(rhs), Seq(offset + intWidth - 1, offset), UnknownType)
    val rhsConnect = castRhs(lhst, sel)
    (offset + intWidth, Seq(Connect(NoInfo, lhs, rhsConnect)))
  }
  private def getPart(lhs: Expression,
                      lhst: Type,
                      rhs: Expression,
                      offset: BigInt): (BigInt, Seq[Statement]) =
    lhst match {
      case t: VectorType => (0 until t.size foldLeft( (offset, Seq[Statement]()) )) {
        case ((curOffset, stmts), i) =>
          val subidx = WSubIndex(lhs, i, t.tpe, UnknownFlow)
          val (tmpOffset, substmts) = getPart(subidx, t.tpe, rhs, curOffset)
          (tmpOffset, stmts ++ substmts)
      }
      case t: BundleType => (t.fields foldRight( (offset, Seq[Statement]()) )) {
        case (f, (curOffset, stmts)) =>
          val subfield = WSubField(lhs, f.name, f.tpe, UnknownFlow)
          val (tmpOffset, substmts) = getPart(subfield, f.tpe, rhs, curOffset)
          (tmpOffset, stmts ++ substmts)
      }
      case t: GroundType => getPartGround(lhs, t, rhs, offset)
      case t => Utils.error(s"Unknown type encountered in fromBits: $lhst")
    }
}

object connectFields {
  def apply(lref: Expression, lname: String, rref: Expression, rname: String): Connect =
    Connect(NoInfo, WSubField(lref, lname), WSubField(rref, rname))
}

object flattenType {
  def apply(t: Type) = UIntType(IntWidth(bitWidth(t)))
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
  def throwInternalError(message: String = "", exception: Option[Exception] = None) = {
    // We'll get the first exception in the chain, keeping it intact.
    val first = true
    val throwable = getThrowable(exception, true)
    val string = if (message.nonEmpty) message + "\n" else message
    error("Internal Error! %sPlease file an issue at https://github.com/ucb-bar/firrtl/issues".format(string), throwable)
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
  def squashEmpty(s: Statement): Statement = s map squashEmpty match {
    case Block(stmts) =>
      val newStmts = stmts filter (_ != EmptyStmt)
      newStmts.size match {
        case 0 => EmptyStmt
        case 1 => newStmts.head
        case _ => Block(newStmts)
      }
    case sx => sx
  }

  /** Returns true if PrimOp is a cast, false otherwise */
  def isCast(op: PrimOp): Boolean = op match {
    case AsUInt | AsSInt | AsClock | AsAsyncReset | AsFixedPoint => true
    case _ => false
  }
  /** Returns true if Expression is a casting PrimOp, false otherwise */
  def isCast(expr: Expression): Boolean = expr match {
    case DoPrim(op, _,_,_) if isCast(op) => true
    case _ => false
  }

  /** Provide a nice name to create a temporary **/
  def niceName(e: Expression): String = niceName(1)(e)
  def niceName(depth: Int)(e: Expression): String = {
    e match {
      case WRef(name, _, _, _) if name(0) == '_' => name
      case WRef(name, _, _, _) => "_" + name
      case WSubAccess(expr, index, _, _) if depth <= 0 => niceName(depth)(expr)
      case WSubAccess(expr, index, _, _) => niceName(depth)(expr) + niceName(depth - 1)(index)
      case WSubField(expr, field, _, _) => niceName(depth)(expr) + "_" + field
      case WSubIndex(expr, index, _, _) => niceName(depth)(expr) + "_" + index
      case Reference(name, _) if name(0) == '_' => name
      case Reference(name, _) => "_" + name
      case SubAccess(expr, index,  _) if depth <= 0 => niceName(depth)(expr)
      case SubAccess(expr, index,  _) => niceName(depth)(expr) + niceName(depth - 1)(index)
      case SubField(expr, field, _) => niceName(depth)(expr) + "_" + field
      case SubIndex(expr, index, _) => niceName(depth)(expr) + "_" + index
      case DoPrim(op, args, consts, _) if depth <= 0 => "_" + op
      case DoPrim(op, args, consts, _) => "_" + op + (args.map(niceName(depth - 1)) ++ consts.map("_" + _)).mkString("")
      case Mux(cond, tval, fval, _) if depth <= 0 => "_mux"
      case Mux(cond, tval, fval, _) => "_mux" + Seq(cond, tval, fval).map(niceName(depth - 1)).mkString("")
      case UIntLiteral(value, _) => "_" + value
      case SIntLiteral(value, _) => "_" + value
    }
  }
  /** Maps node name to value */
  type NodeMap = mutable.HashMap[String, Expression]

  def isTemp(str: String): Boolean = str.head == '_'

  /** Indent the results of [[ir.FirrtlNode.serialize]] */
  def indent(str: String) = str replaceAllLiterally ("\n", "\n  ")

  implicit def toWrappedExpression (x:Expression): WrappedExpression = new WrappedExpression(x)
  def getSIntWidth(s: BigInt): Int = s.bitLength + 1
  def getUIntWidth(u: BigInt): Int = u.bitLength
  def dec2string(v: BigDecimal): String = v.underlying().stripTrailingZeros().toPlainString
  def trim(v: BigDecimal): BigDecimal = BigDecimal(dec2string(v))
  def max(a: BigInt, b: BigInt): BigInt = if (a >= b) a else b
  def min(a: BigInt, b: BigInt): BigInt = if (a >= b) b else a
  def pow_minus_one(a: BigInt, b: BigInt): BigInt = a.pow(b.toInt) - 1
  val BoolType = UIntType(IntWidth(1))
  val one  = UIntLiteral(1)
  val zero = UIntLiteral(0)

  def create_exps(n: String, t: Type): Seq[Expression] =
    create_exps(WRef(n, t, ExpKind, UnknownFlow))
  def create_exps(e: Expression): Seq[Expression] = e match {
    case ex: Mux =>
      val e1s = create_exps(ex.tval)
      val e2s = create_exps(ex.fval)
      e1s zip e2s map {case (e1, e2) =>
        Mux(ex.cond, e1, e2, mux_type_and_widths(e1, e2))
      }
    case ex: ValidIf => create_exps(ex.value) map (e1 => ValidIf(ex.cond, e1, e1.tpe))
    case ex => ex.tpe match {
      case (_: GroundType) => Seq(ex)
      case t: BundleType =>
        t.fields.flatMap(f => create_exps(WSubField(ex, f.name, f.tpe,times(flow(ex), f.flip))))
      case t: VectorType => (0 until t.size).flatMap(i => create_exps(WSubIndex(ex, i, t.tpe,flow(ex))))
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
      e1s zip e2s map {case (e1, e2) =>
        Mux(ex.cond, e1, e2, mux_type_and_widths(e1, e2))
      }
    case ex: ValidIf => expandRef(ex.value) map (e1 => ValidIf(ex.cond, e1, e1.tpe))
    case ex => ex.tpe match {
      case (_: GroundType) => Seq(ex)
      case (t: BundleType) => (t.fields foldLeft Seq[Expression](ex))((exps, f) =>
        exps ++ create_exps(WSubField(ex, f.name, f.tpe,times(flow(ex), f.flip))))
      case (t: VectorType) => (0 until t.size foldLeft Seq[Expression](ex))((exps, i) =>
        exps ++ create_exps(WSubIndex(ex, i, t.tpe,flow(ex))))
    }
  }
  def toTarget(main: String, module: String)(expression: Expression): ReferenceTarget = {
    val tokens = mutable.ArrayBuffer[TargetToken]()
    var ref = "???"
    def onExp(expr: Expression): Expression = {
      expr map onExp match {
        case e: WRef => ref = e.name
        case e: Reference => tokens += TargetToken.Ref(e.name)
        case e: WSubField => tokens += TargetToken.Field(e.name)
        case e: SubField => tokens += TargetToken.Field(e.name)
        case e: WSubIndex => tokens += TargetToken.Index(e.value)
        case e: SubIndex => tokens += TargetToken.Index(e.value)
        case other => throwInternalError("Cannot call Utils.toTarget on non-referencing expression")
      }
      expr
    }
    onExp(expression)
    ReferenceTarget(main, module, Nil, ref, tokens)
  }
   @deprecated("get_flip is fundamentally slow, use to_flip(flow(expr))", "1.2")
   def get_flip(t: Type, i: Int, f: Orientation): Orientation = {
     if (i >= get_size(t)) throwInternalError(s"get_flip: shouldn't be here - $i >= get_size($t)")
     t match {
       case (_: GroundType) => f
       case (tx: BundleType) =>
         val (_, flip) = tx.fields.foldLeft( (i, None: Option[Orientation]) ) {
           case ((n, ret), x) if n < get_size(x.tpe) => ret match {
             case None => (n, Some(get_flip(x.tpe, n, times(x.flip, f))))
             case Some(_) => (n, ret)
           }
           case ((n, ret), x) => (n - get_size(x.tpe), ret)
         }
         flip.get
       case (tx: VectorType) =>
         val (_, flip) = (0 until tx.size).foldLeft( (i, None: Option[Orientation]) ) {
           case ((n, ret), x) if n < get_size(tx.tpe) => ret match {
             case None => (n, Some(get_flip(tx.tpe, n, f)))
             case Some(_) => (n, ret)
           }
           case ((n, ret), x) => (n - get_size(tx.tpe), ret)
         }
         flip.get
     }
   }

   def get_point (e:Expression) : Int = e match {
     case (e: WRef) => 0
     case (e: WSubField) => e.expr.tpe match {case b: BundleType =>
       (b.fields takeWhile (_.name != e.name) foldLeft 0)(
         (point, f) => point + get_size(f.tpe))
    }
    case (e: WSubIndex) => e.value * get_size(e.tpe)
    case (e: WSubAccess) => get_point(e.expr)
  }

  /** Returns true if t, or any subtype, contains a flipped field
    *
    * @param t type [[firrtl.ir.Type]] to be checked
    * @return if t contains [[firrtl.ir.Flip]]
    */
  def hasFlip(t: Type): Boolean = t match {
    case t: BundleType =>
      (t.fields exists (_.flip == Flip)) ||
      (t.fields exists (f => hasFlip(f.tpe)))
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
    e map addKids
    kids
  }

  /** Walks two expression trees and returns a sequence of tuples of where they differ */
  def diff(e1: Expression, e2: Expression): Seq[(Expression, Expression)] = {
    if(weq(e1, e2)) Nil
    else {
      val (e1Kids, e2Kids) = (getKids(e1), getKids(e2))

      if(e1Kids == Nil || e2Kids == Nil || e1Kids.size != e2Kids.size) Seq((e1, e2))
      else {
        e1Kids.zip(e2Kids).flatMap { case (e1k, e2k) => diff(e1k, e2k) }
      }
    }
  }

  /** Returns an inlined expression (replacing node references with values),
    * stopping on a stopping condition or until the reference is not a node
    */
  def inline(nodeMap: NodeMap, stop: String => Boolean = {x: String => false})(e: Expression): Expression = {
    def onExp(e: Expression): Expression = e map onExp match {
      case Reference(name, _) if nodeMap.contains(name) && !stop(name) => onExp(nodeMap(name))
      case WRef(name, _, _, _) if nodeMap.contains(name) && !stop(name) => onExp(nodeMap(name))
      case other => other
    }
    onExp(e)
  }

  def mux_type(e1: Expression, e2: Expression): Type = mux_type(e1.tpe, e2.tpe)
  def mux_type(t1: Type, t2: Type): Type = (t1, t2) match {
    case (ClockType, ClockType) => ClockType
    case (AsyncResetType, AsyncResetType) => AsyncResetType
    case (t1: UIntType, t2: UIntType) => UIntType(UnknownWidth)
    case (t1: SIntType, t2: SIntType) => SIntType(UnknownWidth)
    case (t1: FixedType, t2: FixedType) => FixedType(UnknownWidth, UnknownWidth)
    case (t1: IntervalType, t2: IntervalType) => IntervalType(UnknownBound, UnknownBound, UnknownWidth)
    case (t1: VectorType, t2: VectorType) => VectorType(mux_type(t1.tpe, t2.tpe), t1.size)
    case (t1: BundleType, t2: BundleType) => BundleType(t1.fields zip t2.fields map {
      case (f1, f2) => Field(f1.name, f1.flip, mux_type(f1.tpe, f2.tpe))
    })
    case _ => UnknownType
  }
  def mux_type_and_widths(e1: Expression,e2: Expression): Type =
    mux_type_and_widths(e1.tpe, e2.tpe)
  def mux_type_and_widths(t1: Type, t2: Type): Type = {
    def wmax(w1: Width, w2: Width): Width = (w1, w2) match {
      case (w1x: IntWidth, w2x: IntWidth) => IntWidth(w1x.width max w2x.width)
      case (w1x, w2x) => IsMax(w1x, w2x)
    }
    (t1, t2) match {
      case (ClockType, ClockType) => ClockType
      case (AsyncResetType, AsyncResetType) => AsyncResetType
      case (t1x: UIntType, t2x: UIntType) => UIntType(IsMax(t1x.width, t2x.width))
      case (t1x: SIntType, t2x: SIntType) => SIntType(IsMax(t1x.width, t2x.width))
      case (FixedType(w1, p1), FixedType(w2, p2)) =>
        FixedType(PLUS(MAX(p1, p2),MAX(MINUS(w1, p1), MINUS(w2, p2))), MAX(p1, p2))
      case (IntervalType(l1, u1, p1), IntervalType(l2, u2, p2)) =>
        IntervalType(IsMin(l1, l2), constraint.IsMax(u1, u2), MAX(p1, p2))
      case (t1x: VectorType, t2x: VectorType) => VectorType(
        mux_type_and_widths(t1x.tpe, t2x.tpe), t1x.size)
      case (t1x: BundleType, t2x: BundleType) => BundleType(t1x.fields zip t2x.fields map {
        case (f1, f2) => Field(f1.name, f1.flip, mux_type_and_widths(f1.tpe, f2.tpe))
      })
      case _ => UnknownType
    }
  }

  def module_type(m: DefModule): BundleType = BundleType(m.ports map {
    case Port(_, name, dir, tpe) => Field(name, to_flip(dir), tpe)
  })
  def sub_type(v: Type): Type = v match {
    case vx: VectorType => vx.tpe
    case vx => UnknownType
  }
  def field_type(v: Type, s: String) : Type = v match {
    case vx: BundleType => vx.fields find (_.name == s) match {
      case Some(f) => f.tpe
      case None => UnknownType
    }
    case vx => UnknownType
  }

// =================================
  def error(str: String, cause: Throwable = null) = throw new FirrtlInternalException(str, cause)

//// =============== EXPANSION FUNCTIONS ================
  def get_size(t: Type): Int = t match {
    case tx: BundleType => (tx.fields foldLeft 0)(
      (sum, f) => sum + get_size(f.tpe))
    case tx: VectorType => tx.size * get_size(tx.tpe)
    case tx => 1
  }

  def get_valid_points(t1: Type, t2: Type, flip1: Orientation, flip2: Orientation): Seq[(Int,Int)] = {
    import passes.CheckTypes.legalResetType
    //;println_all(["Inside with t1:" t1 ",t2:" t2 ",f1:" flip1 ",f2:" flip2])
    (t1, t2) match {
      case (_: UIntType, _: UIntType) => if (flip1 == flip2) Seq((0, 0)) else Nil
      case (_: SIntType, _: SIntType) => if (flip1 == flip2) Seq((0, 0)) else Nil
      case (_: FixedType, _: FixedType) => if (flip1 == flip2) Seq((0, 0)) else Nil
      case (_: AnalogType, _: AnalogType) => if (flip1 == flip2) Seq((0, 0)) else Nil
      case (t1x: BundleType, t2x: BundleType) =>
        def emptyMap = Map[String, (Type, Orientation, Int)]()
        val t1_fields = t1x.fields.foldLeft( (emptyMap, 0) ) { case ((map, ilen), f1) =>
          (map + (f1.name ->( (f1.tpe, f1.flip, ilen) )), ilen + get_size(f1.tpe))
        }._1
        t2x.fields.foldLeft( (Seq[(Int, Int)](), 0) ) { case ((points, jlen), f2) =>
          t1_fields get f2.name match {
            case None => (points, jlen + get_size(f2.tpe))
            case Some((f1_tpe, f1_flip, ilen)) =>
              val f1_times = times(flip1, f1_flip)
              val f2_times = times(flip2, f2.flip)
              val ls = get_valid_points(f1_tpe, f2.tpe, f1_times, f2_times)
              (points ++ (ls map { case (x, y) => (x + ilen, y + jlen) }), jlen + get_size(f2.tpe))
          }
        }._1
      case (t1x: VectorType, t2x: VectorType) =>
        val size = math.min(t1x.size, t2x.size)
        (0 until size).foldLeft( (Seq[(Int, Int)](), 0, 0) ) { case ((points, ilen, jlen), _) =>
          val ls = get_valid_points(t1x.tpe, t2x.tpe, flip1, flip2)
          (points ++ (ls map { case (x, y) => (x + ilen, y + jlen) }),
            ilen + get_size(t1x.tpe), jlen + get_size(t2x.tpe))
        }._1
      case (ClockType, ClockType) => if (flip1 == flip2) Seq((0, 0)) else Nil
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
  def swap(g: Flow) : Flow = g match {
    case UnknownFlow => UnknownFlow
    case SourceFlow => SinkFlow
    case SinkFlow => SourceFlow
    case DuplexFlow => DuplexFlow
  }
  def swap(d: Direction) : Direction = d match {
    case Output => Input
    case Input => Output
  }
  def swap(f: Orientation) : Orientation = f match {
    case Default => Flip
    case Flip => Default
  }
  def to_dir(g: Flow): Direction = g match {
    case SourceFlow => Input
    case SinkFlow => Output
  }
  @deprecated("Migrate from 'Gender' to 'Flow. This method will be removed in 1.3", "1.2")
  def to_gender(d: Direction): Gender = d match {
    case Input => MALE
    case Output => FEMALE
  }
  def to_flow(d: Direction): Flow = d match {
    case Input => SourceFlow
    case Output => SinkFlow
  }
  def to_flip(d: Direction): Orientation = d match {
    case Input => Flip
    case Output => Default
  }
  def to_flip(g: Flow): Orientation = g match {
    case SourceFlow => Flip
    case SinkFlow => Default
  }

  def field_flip(v: Type, s: String): Orientation = v match {
    case vx: BundleType => vx.fields find (_.name == s) match {
      case Some(ft) => ft.flip
      case None => Default
    }
    case vx => Default
  }
  def get_field(v: Type, s: String): Field = v match {
    case vx: BundleType => vx.fields find (_.name == s) match {
      case Some(ft) => ft
      case None => throwInternalError(s"get_field: shouldn't be here - $v.$s")
    }
    case vx => throwInternalError(s"get_field: shouldn't be here - $v")
  }

  def times(d: Direction,flip: Orientation): Direction = flip match {
    case Default => d
    case Flip => swap(d)
  }
  def times(g: Flow, d: Direction): Direction = times(d, g)
  def times(d: Direction, g: Flow): Direction = g match {
    case SinkFlow => d
    case SourceFlow => swap(d) // SourceFlow == INPUT == REVERSE
  }

  def times(g: Flow, flip: Orientation): Flow = times(flip, g)
  def times(flip: Orientation, g: Flow): Flow = flip match {
    case Default => g
    case Flip => swap(g)
  }
  def times(f1: Orientation, f2: Orientation): Orientation = f2 match {
    case Default => f1
    case Flip => swap(f1)
  }

// =========== ACCESSORS =========
  def kind(e: Expression): Kind = e match {
    case ex: WRef => ex.kind
    case ex: WSubField => kind(ex.expr)
    case ex: WSubIndex => kind(ex.expr)
    case ex: WSubAccess => kind(ex.expr)
    case ex => ExpKind
  }
  @deprecated("Migrate from 'Gender' to 'Flow'. This method will be removed in 1.3", "1.2")
  def gender(e: Expression): Gender = e match {
    case ex: WRef => ex.gender
    case ex: WSubField => ex.gender
    case ex: WSubIndex => ex.gender
    case ex: WSubAccess => ex.gender
    case ex: DoPrim => MALE
    case ex: UIntLiteral => MALE
    case ex: SIntLiteral => MALE
    case ex: Mux => MALE
    case ex: ValidIf => MALE
    case WInvalid => MALE
    case ex => throwInternalError(s"gender: shouldn't be here - $e")
  }
  @deprecated("Migrate from 'Gender' to 'Flow'. This method will be removed in 1.3", "1.2")
  def get_gender(s: Statement): Gender = s match {
    case sx: DefWire => BIGENDER
    case sx: DefRegister => BIGENDER
    case sx: WDefInstance => MALE
    case sx: DefNode => MALE
    case sx: DefInstance => MALE
    case sx: DefMemory => MALE
    case sx: Block => UNKNOWNGENDER
    case sx: Connect => UNKNOWNGENDER
    case sx: PartialConnect => UNKNOWNGENDER
    case sx: Stop => UNKNOWNGENDER
    case sx: Print => UNKNOWNGENDER
    case sx: IsInvalid => UNKNOWNGENDER
    case EmptyStmt => UNKNOWNGENDER
  }
  @deprecated("Migrate from 'Gender' to 'Flow'. This method will be removed in 1.3", "1.2")
  def get_gender(p: Port): Gender = if (p.direction == Input) MALE else FEMALE
  def flow(e: Expression): Flow = e match {
    case ex: WRef => ex.flow
    case ex: WSubField => ex.flow
    case ex: WSubIndex => ex.flow
    case ex: WSubAccess => ex.flow
    case ex: DoPrim => SourceFlow
    case ex: UIntLiteral => SourceFlow
    case ex: SIntLiteral => SourceFlow
    case ex: Mux => SourceFlow
    case ex: ValidIf => SourceFlow
    case WInvalid => SourceFlow
    case ex => throwInternalError(s"flow: shouldn't be here - $e")
  }
  def get_flow(s: Statement): Flow = s match {
    case sx: DefWire => DuplexFlow
    case sx: DefRegister => DuplexFlow
    case sx: WDefInstance => SourceFlow
    case sx: DefNode => SourceFlow
    case sx: DefInstance => SourceFlow
    case sx: DefMemory => SourceFlow
    case sx: Block => UnknownFlow
    case sx: Connect => UnknownFlow
    case sx: PartialConnect => UnknownFlow
    case sx: Stop => UnknownFlow
    case sx: Print => UnknownFlow
    case sx: IsInvalid => UnknownFlow
    case EmptyStmt => UnknownFlow
  }
  def get_flow(p: Port): Flow = if (p.direction == Input) SourceFlow else SinkFlow
  def get_info(s: Statement): Info = s match {
    case s: HasInfo => s.info
    case _ => NoInfo
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
    * @note This function only supports WRef, WSubField, and WSubIndex
    */
  def splitRef(e: Expression): (WRef, Expression) = e match {
    case e: WRef => (e, EmptyExpression)
    case e: WSubIndex =>
      val (root, tail) = splitRef(e.expr)
      (root, WSubIndex(tail, e.value, e.tpe, e.flow))
    case e: WSubField =>
      val (root, tail) = splitRef(e.expr)
      tail match {
        case EmptyExpression => (root, WRef(e.name, e.tpe, root.kind, e.flow))
        case exp => (root, WSubField(tail, e.name, e.tpe, e.flow))
      }
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
      case c: Conditionally =>
        val m = (getRootDecl(name)(c.conseq), getRootDecl(name)(c.alt))
        (m: @unchecked) match {
          case (Some(decl), None) => Some(decl)
          case (None, Some(decl)) => Some(decl)
          case (None, None) => None
        }
      case begin: Block =>
        val stmts = begin.stmts flatMap getRootDecl(name) // can we short circuit?
        if (stmts.nonEmpty) Some(stmts.head) else None
      case _ => None
    }
    expr match {
      case (_: WRef | _: WSubIndex | _: WSubField) =>
        val (root, tail) = splitRef(expr)
        val rootDecl = m.ports find (_.name == root.name) match {
          case Some(decl) => decl
          case None =>
            getRootDecl(root.name)(m.body) match {
              case Some(decl) => decl
              case None => throw new DeclarationNotFoundException(
                s"[module ${m.name}]  Reference ${expr.serialize} not declared!")
            }
        }
        rootDecl
      case e => Utils.error(s"getDeclaration does not support Expressions of type ${e.getClass}")
    }
  }

  val v_keywords = Set(
    "alias", "always", "always_comb", "always_ff", "always_latch",
    "and", "assert", "assign", "assume", "attribute", "automatic",

    "before", "begin", "bind", "bins", "binsof", "bit", "break",
    "buf", "bufif0", "bufif1", "byte",

    "case", "casex", "casez", "cell", "chandle", "checker", "class", "clocking",
    "cmos", "config", "const", "constraint", "context", "continue",
    "cover", "covergroup", "coverpoint", "cross",

    "deassign", "default", "defparam", "design", "disable", "dist", "do",

    "edge", "else", "end", "endattribute", "endcase", "endclass",
    "endclocking", "endconfig", "endfunction", "endgenerate",
    "endgroup", "endinterface", "endmodule", "endpackage",
    "endprimitive", "endprogram", "endproperty", "endspecify",
    "endsequence", "endtable", "endtask",
    "enum", "event", "expect", "export", "extends", "extern",

    "final", "first_match", "for", "force", "foreach", "forever",
    "fork", "forkjoin", "function",
    "generate", "genvar",
    "highz0", "highz1",
    "if", "iff", "ifnone", "ignore_bins", "illegal_bins", "import",
    "incdir", "include", "initial", "initvar", "inout", "input",
    "inside", "instance", "int", "integer", "interconnect",
    "interface", "intersect",

    "join", "join_any", "join_none", "large", "liblist", "library",
    "local", "localparam", "logic", "longint",

    "macromodule", "matches", "medium", "modport", "module",

    "nand", "negedge", "new", "nmos", "nor", "noshowcancelled",
    "not", "notif0", "notif1", "null",

    "or", "output",

    "package", "packed", "parameter", "pmos", "posedge",
    "primitive", "priority", "program", "property", "protected",
    "pull0", "pull1", "pulldown", "pullup",
    "pulsestyle_onevent", "pulsestyle_ondetect", "pure",

    "rand", "randc", "randcase", "randsequence", "rcmos",
    "real", "realtime", "ref", "reg", "release", "repeat",
    "return", "rnmos", "rpmos", "rtran", "rtranif0", "rtranif1",

    "scalared", "sequence", "shortint", "shortreal", "showcancelled",
    "signed", "small", "solve", "specify", "specparam", "static",
    "strength", "string", "strong0", "strong1", "struct", "super",
    "supply0", "supply1",

    "table", "tagged", "task", "this", "throughout", "time", "timeprecision",
    "timeunit", "tran", "tranif0", "tranif1", "tri", "tri0", "tri1", "triand",
    "trior", "trireg", "type","typedef",

    "union", "unique", "unsigned", "use",

    "var", "vectored", "virtual", "void",

    "wait", "wait_order", "wand", "weak0", "weak1", "while",
    "wildcard", "wire", "with", "within", "wor",

    "xnor", "xor",

    "SYNTHESIS",
    "PRINTF_COND",
    "VCS")

  /** Expand a name into its prefixes, e.g., 'foo_bar__baz' becomes 'Seq[foo_, foo_bar__, foo_bar__baz]'. This can be used
    * to produce better names when generating prefix unique names.
    * @param name a signal name
    * @param prefixDelim a prefix delimiter (default is "_")
    * @return the signal name and any prefixes
    */
  def expandPrefixes(name: String, prefixDelim: String = "_"): Seq[String] = {
    val regex = ("(" + Regex.quote(prefixDelim) + ")+[A-Za-z0-9$]").r

    name +: regex
      .findAllMatchIn(name)
      .map(_.end - 1)
      .toSeq
      .foldLeft(Seq[String]()){ case (seq, id) => seq :+ name.splitAt(id)._1 }
  }
}

object MemoizedHash {
  implicit def convertTo[T](e: T): MemoizedHash[T] = new MemoizedHash(e)
  implicit def convertFrom[T](f: MemoizedHash[T]): T = f.t
}

class MemoizedHash[T](val t: T) {
  override lazy val hashCode = t.hashCode
  override def equals(that: Any) = that match {
    case x: MemoizedHash[_] => t equals x.t
    case _ => false
  }
}

/**
  * Maintains a one to many graph of each modules instantiated child module.
  * This graph can be searched for a path from a child module back to one of
  * it's parents.  If one is found a recursive loop has happened
  * The graph is a map between the name of a node to set of names of that nodes children
  */
class ModuleGraph {
  val nodes = mutable.HashMap[String, mutable.HashSet[String]]()

  /**
    * Add a child to a parent node
    * A parent node is created if it does not already exist
    *
    * @param parent module that instantiates another module
    * @param child  module instantiated by parent
    * @return a list indicating a path from child to parent, empty if no such path
    */
  def add(parent: String, child: String): List[String] = {
    val childSet = nodes.getOrElseUpdate(parent, new mutable.HashSet[String])
    childSet += child
    pathExists(child, parent, List(child, parent))
  }

  /**
    * Starting at the name of a given child explore the tree of all children in depth first manner.
    * Return the first path (a list of strings) that goes from child to parent,
    * or an empty list of no such path is found.
    *
    * @param child  starting name
    * @param parent name to find in children (recursively)
    * @param path   path being investigated as possible route
    * @return
    */
  def pathExists(child: String, parent: String, path: List[String] = Nil): List[String] = {
    nodes.get(child) match {
      case Some(children) =>
        if(children(parent)) {
          parent :: path
        }
        else {
          children.foreach { grandchild =>
            val newPath = pathExists(grandchild, parent, grandchild :: path)
            if(newPath.nonEmpty) {
              return newPath
            }
          }
          Nil
        }
      case _ => Nil
    }
  }
}
