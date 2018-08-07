// See LICENSE for license details.

package firrtl

import scala.collection.Seq
import Utils._
import firrtl.ir._
import WrappedExpression._
import WrappedWidth._

trait Kind
case object WireKind extends Kind
case object PoisonKind extends Kind
case object RegKind extends Kind
case object InstanceKind extends Kind
case object PortKind extends Kind
case object NodeKind extends Kind
case object MemKind extends Kind
case object ExpKind extends Kind

trait Gender
case object MALE extends Gender
case object FEMALE extends Gender
case object BIGENDER extends Gender
case object UNKNOWNGENDER extends Gender

case class WRef(name: String, tpe: Type, kind: Kind, gender: Gender) extends Expression {
  def serialize: String = name
  def mapExpr(f: Expression => Expression): Expression = this
  def mapType(f: Type => Type): Expression = this.copy(tpe = f(tpe))
  def mapWidth(f: Width => Width): Expression = this
}
object WRef {
  /** Creates a WRef from a Wire */
  def apply(wire: DefWire): WRef = new WRef(wire.name, wire.tpe, WireKind, UNKNOWNGENDER)
  /** Creates a WRef from a Register */
  def apply(reg: DefRegister): WRef = new WRef(reg.name, reg.tpe, RegKind, UNKNOWNGENDER)
  /** Creates a WRef from a Node */
  def apply(node: DefNode): WRef = new WRef(node.name, node.value.tpe, NodeKind, MALE)
  def apply(n: String, t: Type = UnknownType, k: Kind = ExpKind): WRef = new WRef(n, t, k, UNKNOWNGENDER)
}
case class WSubField(expr: Expression, name: String, tpe: Type, gender: Gender) extends Expression {
  def serialize: String = s"${expr.serialize}.$name"
  def mapExpr(f: Expression => Expression): Expression = this.copy(expr = f(expr))
  def mapType(f: Type => Type): Expression = this.copy(tpe = f(tpe))
  def mapWidth(f: Width => Width): Expression = this
}
object WSubField {
  def apply(expr: Expression, n: String): WSubField = new WSubField(expr, n, field_type(expr.tpe, n), UNKNOWNGENDER)
  def apply(expr: Expression, name: String, tpe: Type): WSubField = new WSubField(expr, name, tpe, UNKNOWNGENDER)
}
case class WSubIndex(expr: Expression, value: Int, tpe: Type, gender: Gender) extends Expression {
  def serialize: String = s"${expr.serialize}[$value]"
  def mapExpr(f: Expression => Expression): Expression = this.copy(expr = f(expr))
  def mapType(f: Type => Type): Expression = this.copy(tpe = f(tpe))
  def mapWidth(f: Width => Width): Expression = this
}
case class WSubAccess(expr: Expression, index: Expression, tpe: Type, gender: Gender) extends Expression {
  def serialize: String = s"${expr.serialize}[${index.serialize}]"
  def mapExpr(f: Expression => Expression): Expression = this.copy(expr = f(expr), index = f(index))
  def mapType(f: Type => Type): Expression = this.copy(tpe = f(tpe))
  def mapWidth(f: Width => Width): Expression = this
}
case object WVoid extends Expression {
  def tpe = UnknownType
  def serialize: String = "VOID"
  def mapExpr(f: Expression => Expression): Expression = this
  def mapType(f: Type => Type): Expression = this
  def mapWidth(f: Width => Width): Expression = this
}
case object WInvalid extends Expression {
  def tpe = UnknownType
  def serialize: String = "INVALID"
  def mapExpr(f: Expression => Expression): Expression = this
  def mapType(f: Type => Type): Expression = this
  def mapWidth(f: Width => Width): Expression = this
}
// Useful for splitting then remerging references
case object EmptyExpression extends Expression {
  def tpe = UnknownType
  def serialize: String = "EMPTY"
  def mapExpr(f: Expression => Expression): Expression = this
  def mapType(f: Type => Type): Expression = this
  def mapWidth(f: Width => Width): Expression = this
}
case class WDefInstance(info: Info, name: String, module: String, tpe: Type) extends Statement with IsDeclaration {
  def serialize: String = s"inst $name of $module" + info.serialize
  def mapExpr(f: Expression => Expression): Statement = this
  def mapStmt(f: Statement => Statement): Statement = this
  def mapType(f: Type => Type): Statement = this.copy(tpe = f(tpe))
  def mapString(f: String => String): Statement = this.copy(name = f(name))
  def mapInfo(f: Info => Info): Statement = this.copy(f(info))
}
object WDefInstance {
  def apply(name: String, module: String): WDefInstance = new WDefInstance(NoInfo, name, module, UnknownType)
}
case class WDefInstanceConnector(
    info: Info,
    name: String,
    module: String,
    tpe: Type,
    portCons: Seq[(Expression, Expression)]) extends Statement with IsDeclaration {
  def serialize: String = s"inst $name of $module with ${tpe.serialize} connected to " +
                          portCons.map(_._2.serialize).mkString("(", ", ", ")") + info.serialize
  def mapExpr(f: Expression => Expression): Statement =
    this.copy(portCons = portCons map { case (e1, e2) => (f(e1), f(e2)) })
  def mapStmt(f: Statement => Statement): Statement = this
  def mapType(f: Type => Type): Statement = this.copy(tpe = f(tpe))
  def mapString(f: String => String): Statement = this.copy(name = f(name))
  def mapInfo(f: Info => Info): Statement = this.copy(f(info))
}

// Resultant width is the same as the maximum input width
case object Addw extends PrimOp { override def toString = "addw" }
// Resultant width is the same as the maximum input width
case object Subw extends PrimOp { override def toString = "subw" }
// Resultant width is the same as input argument width
case object Dshlw extends PrimOp { override def toString = "dshlw" }
// Resultant width is the same as input argument width
case object Shlw extends PrimOp { override def toString = "shlw" }

object WrappedExpression {
  def apply(e: Expression) = new WrappedExpression(e)
  def we(e: Expression) = new WrappedExpression(e)
  def weq(e1: Expression, e2: Expression) = we(e1) == we(e2)
}
class WrappedExpression(val e1: Expression) {
  override def equals(we: Any) = we match {
    case (we: WrappedExpression) => (e1,we.e1) match {
      case (e1x: UIntLiteral, e2x: UIntLiteral) => e1x.value == e2x.value && eqw(e1x.width, e2x.width)
      case (e1x: SIntLiteral, e2x: SIntLiteral) => e1x.value == e2x.value && eqw(e1x.width, e2x.width)
      case (e1x: WRef, e2x: WRef) => e1x.name equals e2x.name
      case (e1x: WSubField, e2x: WSubField) => (e1x.name equals e2x.name) && weq(e1x.expr,e2x.expr)
      case (e1x: WSubIndex, e2x: WSubIndex) => (e1x.value == e2x.value) && weq(e1x.expr,e2x.expr)
      case (e1x: WSubAccess, e2x: WSubAccess) => weq(e1x.index,e2x.index) && weq(e1x.expr,e2x.expr)
      case (WVoid, WVoid) => true
      case (WInvalid, WInvalid) => true
      case (e1x: DoPrim, e2x: DoPrim) => e1x.op == e2x.op &&
         ((e1x.consts zip e2x.consts) forall {case (x, y) => x == y}) &&
         ((e1x.args zip e2x.args) forall {case (x, y) => weq(x, y)})
      case (e1x: Mux, e2x: Mux) => weq(e1x.cond,e2x.cond) && weq(e1x.tval,e2x.tval) && weq(e1x.fval,e2x.fval)
      case (e1x: ValidIf, e2x: ValidIf) => weq(e1x.cond,e2x.cond) && weq(e1x.value,e2x.value)
      case (e1x, e2x) => false
    }
    case _ => false
  }
  override def hashCode = e1.serialize.hashCode
  override def toString = e1.serialize
}

private[firrtl] sealed trait HasMapWidth {
  def mapWidth(f: Width => Width): Width
}
case class VarWidth(name: String) extends Width with HasMapWidth {
  def serialize: String = name
  def mapWidth(f: Width => Width): Width = this
}
case class PlusWidth(arg1: Width, arg2: Width) extends Width with HasMapWidth {
  def serialize: String = "(" + arg1.serialize + " + " + arg2.serialize + ")"
  def mapWidth(f: Width => Width): Width = PlusWidth(f(arg1), f(arg2))
}
case class MinusWidth(arg1: Width, arg2: Width) extends Width with HasMapWidth {
  def serialize: String = "(" + arg1.serialize + " - " + arg2.serialize + ")"
  def mapWidth(f: Width => Width): Width = MinusWidth(f(arg1), f(arg2))
}
case class MaxWidth(args: Seq[Width]) extends Width with HasMapWidth {
  def serialize: String = args map (_.serialize) mkString ("max(", ", ", ")")
  def mapWidth(f: Width => Width): Width = MaxWidth(args map f)
}
case class MinWidth(args: Seq[Width]) extends Width with HasMapWidth {
  def serialize: String = args map (_.serialize) mkString ("min(", ", ", ")")
  def mapWidth(f: Width => Width): Width = MinWidth(args map f)
}
case class ExpWidth(arg1: Width) extends Width with HasMapWidth {
  def serialize: String = "exp(" + arg1.serialize + " )"
  def mapWidth(f: Width => Width): Width = ExpWidth(f(arg1))
}

object WrappedType {
  def apply(t: Type) = new WrappedType(t)
  def wt(t: Type) = apply(t)
}
class WrappedType(val t: Type) {
  def wt(tx: Type) = new WrappedType(tx)
  override def equals(o: Any): Boolean = o match {
    case (t2: WrappedType) => (t, t2.t) match {
      case (_: UIntType, _: UIntType) => true
      case (_: SIntType, _: SIntType) => true
      case (ClockType, ClockType) => true
      case (_: FixedType, _: FixedType) => true
      // Analog totally skips out of the Firrtl type system.
      // The only way Analog can play with another Analog component is through Attach.
      // Ohterwise, we'd need to special case it during ExpandWhens, Lowering,
      // ExpandConnects, etc.
      case (_: AnalogType, _: AnalogType) => false
      case (t1: VectorType, t2: VectorType) =>
        t1.size == t2.size && wt(t1.tpe) == wt(t2.tpe)
      case (t1: BundleType, t2: BundleType) =>
        t1.fields.size == t2.fields.size && (
        (t1.fields zip t2.fields) forall { case (f1, f2) =>
          f1.flip == f2.flip && f1.name == f2.name
        }) && ((t1.fields zip t2.fields) forall { case (f1, f2) =>
          wt(f1.tpe) == wt(f2.tpe)
        })
      case _ => false
    }
    case _ => false
  }
}

object WrappedWidth {
  def eqw(w1: Width, w2: Width): Boolean = new WrappedWidth(w1) == new WrappedWidth(w2)
}
   
class WrappedWidth (val w: Width) {
  def ww(w: Width): WrappedWidth = new WrappedWidth(w)
  override def toString = w match {
    case (w: VarWidth) => w.name
    case (w: MaxWidth) => s"max(${w.args.mkString})"
    case (w: MinWidth) => s"min(${w.args.mkString})"
    case (w: PlusWidth) => s"(${w.arg1} + ${w.arg2})"
    case (w: MinusWidth) => s"(${w.arg1} -${w.arg2})"
    case (w: ExpWidth) => s"exp(${w.arg1})"
    case (w: IntWidth) => w.width.toString
    case UnknownWidth => "?"
  }
  override def equals(o: Any): Boolean = o match {
    case (w2: WrappedWidth) => (w, w2.w) match {
      case (w1: VarWidth, w2: VarWidth) => w1.name.equals(w2.name)
      case (w1: MaxWidth, w2: MaxWidth) => w1.args.size == w2.args.size &&
        (w1.args forall (a1 => w2.args exists (a2 => eqw(a1, a2))))
      case (w1: MinWidth, w2: MinWidth) => w1.args.size == w2.args.size &&
        (w1.args forall (a1 => w2.args exists (a2 => eqw(a1, a2))))
      case (w1: IntWidth, w2: IntWidth) => w1.width == w2.width
      case (w1: PlusWidth, w2: PlusWidth) => 
        (ww(w1.arg1) == ww(w2.arg1) && ww(w1.arg2) == ww(w2.arg2)) ||
        (ww(w1.arg1) == ww(w2.arg2) && ww(w1.arg2) == ww(w2.arg1))
      case (w1: MinusWidth,w2: MinusWidth) => 
        (ww(w1.arg1) == ww(w2.arg1) && ww(w1.arg2) == ww(w2.arg2)) ||
        (ww(w1.arg1) == ww(w2.arg2) && ww(w1.arg2) == ww(w2.arg1))
      case (w1: ExpWidth, w2: ExpWidth) => ww(w1.arg1) == ww(w2.arg1)
      case (UnknownWidth, UnknownWidth) => true
      case _ => false
    }
    case _ => false
  }
}

trait Constraint
class WGeq(val loc: Width, val exp: Width) extends Constraint {
  override def toString = {
    val wloc = new WrappedWidth(loc)
    val wexp = new WrappedWidth(exp)
    wloc.toString + " >= " + wexp.toString
  }
}
object WGeq {
  def apply(loc: Width, exp: Width) = new WGeq(loc, exp)
}

abstract class MPortDir extends FirrtlNode
case object MInfer extends MPortDir {
  def serialize: String = "infer"
}
case object MRead extends MPortDir {
  def serialize: String = "read"
}
case object MWrite extends MPortDir {
  def serialize: String = "write"
}
case object MReadWrite extends MPortDir {
  def serialize: String = "rdwr"
}

case class CDefMemory(
    info: Info,
    name: String,
    tpe: Type,
    size: Int,
    seq: Boolean) extends Statement with HasInfo {
  def serialize: String = (if (seq) "smem" else "cmem") +
    s" $name : ${tpe.serialize} [$size]" + info.serialize
  def mapExpr(f: Expression => Expression): Statement = this
  def mapStmt(f: Statement => Statement): Statement = this
  def mapType(f: Type => Type): Statement = this.copy(tpe = f(tpe))
  def mapString(f: String => String): Statement = this.copy(name = f(name))
  def mapInfo(f: Info => Info): Statement = this.copy(f(info))
}
case class CDefMPort(info: Info,
    name: String,
    tpe: Type,
    mem: String,
    exps: Seq[Expression],
    direction: MPortDir) extends Statement with HasInfo {
  def serialize: String = {
    val dir = direction.serialize
    s"$dir mport $name = $mem[${exps.head.serialize}], ${exps(1).serialize}" + info.serialize
  }
  def mapExpr(f: Expression => Expression): Statement = this.copy(exps = exps map f)
  def mapStmt(f: Statement => Statement): Statement = this
  def mapType(f: Type => Type): Statement = this.copy(tpe = f(tpe))
  def mapString(f: String => String): Statement = this.copy(name = f(name))
  def mapInfo(f: Info => Info): Statement = this.copy(f(info))
}

