
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
case object ADD_OP extends PrimOp 
case object SUB_OP extends PrimOp
case object MUL_OP extends PrimOp
case object DIV_OP extends PrimOp
case object REM_OP extends PrimOp
case object LESS_OP extends PrimOp
case object LESS_EQ_OP extends PrimOp
case object GREATER_OP extends PrimOp
case object GREATER_EQ_OP extends PrimOp
case object EQUAL_OP extends PrimOp
case object NEQUAL_OP extends PrimOp
case object PAD_OP extends PrimOp
case object AS_UINT_OP extends PrimOp
case object AS_SINT_OP extends PrimOp
case object AS_CLOCK_OP extends PrimOp
case object SHIFT_LEFT_OP extends PrimOp
case object SHIFT_RIGHT_OP extends PrimOp
case object DYN_SHIFT_LEFT_OP extends PrimOp
case object DYN_SHIFT_RIGHT_OP extends PrimOp
case object CONVERT_OP extends PrimOp
case object NEG_OP extends PrimOp
case object NOT_OP extends PrimOp
case object AND_OP extends PrimOp
case object OR_OP extends PrimOp
case object XOR_OP extends PrimOp
case object AND_REDUCE_OP extends PrimOp
case object OR_REDUCE_OP extends PrimOp
case object XOR_REDUCE_OP extends PrimOp
case object CONCAT_OP extends PrimOp
case object BITS_SELECT_OP extends PrimOp
case object HEAD_OP extends PrimOp
case object TAIL_OP extends PrimOp

trait Expression extends AST
case class Ref(name: String, tpe: Type) extends Expression
case class SubField(exp: Expression, name: String, tpe: Type) extends Expression
case class SubIndex(exp: Expression, value: Int, tpe: Type) extends Expression
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
case class DefMemory(info: Info, name: String, data_type: Type, depth: Int, write_latency: Int, 
               read_latency: Int, readers: Seq[String], writers: Seq[String], readwriters: Seq[String]) extends Stmt
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
case class UnknownWidth() extends Width

trait Flip extends AST
case object DEFAULT extends Flip
case object REVERSE extends Flip

case class Field(name: String, flip: Flip, tpe: Type) extends AST

trait Type extends AST
case class UIntType(width: Width) extends Type
case class SIntType(width: Width) extends Type
case class BundleType(fields: Seq[Field]) extends Type
case class VectorType(tpe: Type, size: Int) extends Type
case class ClockType() extends Type
case class UnknownType() extends Type

trait Direction extends AST
case object INPUT extends Direction
case object OUTPUT extends Direction

case class Port(info: Info, name: String, direction: Direction, tpe: Type) extends AST

trait Module extends AST {
  val info : Info
  val name : String
  val ports : Seq[Port]
}
case class InModule(info: Info, name: String, ports: Seq[Port], body: Stmt) extends Module
case class ExModule(info: Info, name: String, ports: Seq[Port]) extends Module

case class Circuit(info: Info, modules: Seq[Module], main: String) extends AST

