
/* TODO
 *  - Should FileInfo be a FIRRTL node?
 *
 */

package firrtl

import scala.collection.Seq

// Should this be defined elsewhere?
case class FileInfo(file: String, line: Int, column: Int) {
  override def toString(): String = s"$file@$line.$column"
}

trait AST 

trait Primop extends AST
case object Add extends Primop 
case object Sub extends Primop
case object Addw extends Primop
case object Subw extends Primop
case object Mul extends Primop
case object Div extends Primop
case object Mod extends Primop
case object Quo extends Primop
case object Rem extends Primop
case object Lt extends Primop
case object Leq extends Primop
case object Gt extends Primop
case object Geq extends Primop
case object Eq extends Primop
case object Neq extends Primop
case object Eqv extends Primop
case object Neqv extends Primop
case object Mux extends Primop
case object Pad extends Primop
case object AsUInt extends Primop
case object AsSInt extends Primop
case object Shl extends Primop
case object Shr extends Primop
case object Dshl extends Primop
case object Dshr extends Primop
case object Cvt extends Primop
case object Neg extends Primop
case object Not extends Primop
case object And extends Primop
case object Or extends Primop
case object Xor extends Primop
case object Andr extends Primop
case object Orr extends Primop
case object Xorr extends Primop
case object Cat extends Primop
case object Bit extends Primop
case object Bits extends Primop

// TODO stanza ir has types on many of these, why? Is it the type of what we're referencing?
// Add types, default to UNKNOWN
// TODO add type
trait Exp extends AST
case class UIntValue(value: BigInt, width: Width) extends Exp
case class SIntValue(value: BigInt, width: Width) extends Exp
case class Ref(name: String, tpe: Type) extends Exp
case class Subfield(exp: Exp, name: String, tpe: Type) extends Exp
case class Index(exp: Exp, value: BigInt, tpe: Type) extends Exp
case class DoPrimop(op: Primop, args: Seq[Exp], consts: Seq[BigInt], tpe: Type) extends Exp 

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


