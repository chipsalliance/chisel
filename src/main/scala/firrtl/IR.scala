package firrtl

import scala.collection.Seq

// Should this be defined elsewhere?
case class FileInfo(file: String, line: Int, column: Int) {
  override def toString(): String = s"$file@$line.$column"
}

trait AST 

trait PrimOp extends AST
case object Add extends PrimOp 
case object Sub extends PrimOp
case object Addw extends PrimOp
case object Subw extends PrimOp
case object Mul extends PrimOp
case object Div extends PrimOp
case object Mod extends PrimOp
case object Quo extends PrimOp
case object Rem extends PrimOp
case object Lt extends PrimOp
case object Leq extends PrimOp
case object Gt extends PrimOp
case object Geq extends PrimOp
case object Eq extends PrimOp
case object Neq extends PrimOp
case object Mux extends PrimOp
case object Pad extends PrimOp
case object AsUInt extends PrimOp
case object AsSInt extends PrimOp
case object Shl extends PrimOp
case object Shr extends PrimOp
case object Dshl extends PrimOp
case object Dshr extends PrimOp
case object Cvt extends PrimOp
case object Neg extends PrimOp
case object Not extends PrimOp
case object And extends PrimOp
case object Or extends PrimOp
case object Xor extends PrimOp
case object Andr extends PrimOp
case object Orr extends PrimOp
case object Xorr extends PrimOp
case object Cat extends PrimOp
case object Bit extends PrimOp
case object Bits extends PrimOp

// TODO stanza ir has types on many of these, why? Is it the type of what we're referencing?
// Add types, default to UNKNOWN
// TODO add type
trait Exp extends AST
case class UIntValue(value: BigInt, width: Width) extends Exp
case class SIntValue(value: BigInt, width: Width) extends Exp
case class Ref(name: String, tpe: Type) extends Exp
case class Subfield(exp: Exp, name: String, tpe: Type) extends Exp
case class Subindex(exp: Exp, value: BigInt) extends Exp
case class DoPrimOp(op: PrimOp, args: Seq[Exp], consts: Seq[BigInt]) extends Exp 

trait AccessorDir extends AST
case object Infer extends AccessorDir
case object Read extends AccessorDir
case object Write extends AccessorDir
case object RdWr extends AccessorDir

trait Stmt extends AST
case class DefWire(info: FileInfo, name: String, tpe: Type) extends Stmt
case class DefReg(info: FileInfo, name: String, tpe: Type, clock: Exp, reset: Exp) extends Stmt
case class DefMemory(info: FileInfo, name: String, seq: Boolean, tpe: Type, clock: Exp) extends Stmt
case class DefInst(info: FileInfo, name: String, module: Exp) extends Stmt 
case class DefNode(info: FileInfo, name: String, value: Exp) extends Stmt
case class DefPoison(info: FileInfo, name: String, tpe: Type) extends Stmt
case class DefAccessor(info: FileInfo, name: String, dir: AccessorDir, source: Exp, index: Exp) extends Stmt
case class OnReset(info: FileInfo, lhs: Exp, rhs: Exp) extends Stmt
case class Connect(info: FileInfo, lhs: Exp, rhs: Exp) extends Stmt
case class BulkConnect(info: FileInfo, lhs: Exp, rhs: Exp) extends Stmt
case class When(info: FileInfo, pred: Exp, conseq: Stmt, alt: Stmt) extends Stmt
case class Assert(info: FileInfo, pred: Exp) extends Stmt
case class Block(stmts: Seq[Stmt]) extends Stmt
case object EmptyStmt extends Stmt

trait Width extends AST 
case class IntWidth(width: BigInt) extends Width 
case object UnknownWidth extends Width

trait FieldDir extends AST
case object Default extends FieldDir
case object Reverse extends FieldDir

case class Field(name: String, dir: FieldDir, tpe: Type) extends AST

trait Type extends AST
case class UIntType(width: Width) extends Type
case class SIntType(width: Width) extends Type
case object ClockType extends Type
case class BundleType(fields: Seq[Field]) extends Type
case class VectorType(tpe: Type, size: BigInt) extends Type
case object UnknownType extends Type

trait PortDir extends AST
case object Input extends PortDir
case object Output extends PortDir

case class Port(info: FileInfo, name: String, dir: PortDir, tpe: Type) extends AST

case class Module(info: FileInfo, name: String, ports: Seq[Port], stmt: Stmt) extends AST

case class Circuit(info: FileInfo, name: String, modules: Seq[Module]) extends AST


