
package firrtl

import scala.collection.Seq

// Should this be defined elsewhere?
/*
Structure containing source locator information.
Member of most Stmt case classes.
*/
trait Info
case object NoInfo extends Info
case class FileInfo(file: String, line: Int, column: Int) extends Info {
  override def toString(): String = s"$file@$line.$column"
}

case class FIRRTLException(str:String) extends Exception

trait AST 

trait PrimOp extends AST
case object AddOp extends PrimOp 
case object SubOp extends PrimOp
case object MulOp extends PrimOp
case object DivOp extends PrimOp
case object RemOp extends PrimOp
case object LessOp extends PrimOp
case object LessEqOp extends PrimOp
case object GreaterOp extends PrimOp
case object GreaterEqOp extends PrimOp
case object EqualOp extends PrimOp
case object NEqualOp extends PrimOp
case object PadOp extends PrimOp
case object AsUIntOp extends PrimOp
case object AsSIntOp extends PrimOp
case object AsClockOp extends PrimOp
case object ShiftLeftOp extends PrimOp
case object ShiftRightOp extends PrimOp
case object DynShiftLeftOp extends PrimOp
case object DynShiftRightOp extends PrimOp
case object ConvertOp extends PrimOp
case object NegOp extends PrimOp
case object BitNotOp extends PrimOp
case object BitAndOp extends PrimOp
case object BitOrOp extends PrimOp
case object BitXorOp extends PrimOp
case object BitAndReduceOp extends PrimOp
case object BitOrReduceOp extends PrimOp
case object BitXorReduceOp extends PrimOp
case object ConcatOp extends PrimOp
case object BitsSelectOp extends PrimOp
case object HeadOp extends PrimOp
case object TailOp extends PrimOp

trait Expression extends AST
case class Ref(name: String, tpe: Type) extends Expression
case class SubField(exp: Expression, name: String, tpe: Type) extends Expression
case class SubIndex(exp: Expression, value: BigInt, tpe: Type) extends Expression
case class SubAccess(exp: Expression, index: Expression, tpe: Type) extends Expression
case class Mux(cond: Expression, tval: Expression, fval: Expression, tpe: Type) extends Expression
case class ValidIf(cond: Expression, value: Expression, tpe: Type) extends Expression
case class UIntValue(value: BigInt, width: Width) extends Expression
case class SIntValue(value: BigInt, width: Width) extends Expression
case class DoPrim(op: PrimOp, args: Seq[Expression], consts: Seq[BigInt], tpe: Type) extends Expression 

trait Stmt extends AST
case class DefWire(info: Info, name: String, tpe: Type) extends Stmt
case class DefPoison(info: Info, name: String, tpe: Type) extends Stmt
case class DefRegister(info: Info, name: String, tpe: Type, clock: Expression, reset: Expression, init: Expression) extends Stmt 
case class DefInstance(info: Info, name: String, module: String) extends Stmt 
case class DefMemory(info: Info, name: String, dataType: Type, depth: Int, writeLatency: Int, 
               readLatency: Int, readers: Seq[String], writers: Seq[String], readwriters: Seq[String]) extends Stmt
case class DefNode(info: Info, name: String, value: Expression) extends Stmt
case class Conditionally(info: Info, pred: Expression, conseq: Stmt, alt: Stmt) extends Stmt
case class Begin(stmts: Seq[Stmt]) extends Stmt
case class BulkConnect(info: Info, loc: Expression, exp: Expression) extends Stmt
case class Connect(info: Info, loc: Expression, exp: Expression) extends Stmt
case class IsInvalid(info: Info, exp: Expression) extends Stmt
case class Stop(info: Info, ret: Int, clk: Expression, en: Expression) extends Stmt
case class Print(info: Info, string: String, args: Seq[Expression], clk: Expression, en: Expression) extends Stmt
case class Empty() extends Stmt

trait Width extends AST 
case class IntWidth(width: BigInt) extends Width 
case object UnknownWidth extends Width

trait Flip extends AST
case object Default extends Flip
case object Reverse extends Flip

case class Field(name: String, flip: Flip, tpe: Type) extends AST

trait Type extends AST
case class UIntType(width: Width) extends Type
case class SIntType(width: Width) extends Type
case class BundleType(fields: Seq[Field]) extends Type
case class VectorType(tpe: Type, size: BigInt) extends Type
case class ClockType() extends Type
case class UnknownType() extends Type

trait Direction extends AST
case object Input extends Direction
case object Output extends Direction

case class Port(info: Info, name: String, dir: Direction, tpe: Type) extends AST

trait Module extends AST {
  val info : Info
  val name : String
  val ports : Seq[Port]
}
case class InModule(info: Info, name: String, ports: Seq[Port], body: Stmt) extends Module
case class ExModule(info: Info, name: String, ports: Seq[Port]) extends Module

case class Circuit(info: Info, modules: Seq[Module], main: String) extends AST


