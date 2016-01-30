
package firrtl

import scala.collection.Seq
import Utils._


trait Kind
case class WireKind() extends Kind
case class PoisonKind() extends Kind 
case class RegKind() extends Kind
case class InstanceKind() extends Kind
case class PortKind() extends Kind
case class NodeKind() extends Kind
case class MemKind(ports:Seq[String]) extends Kind
case class ExpKind() extends Kind

trait Gender
case object MALE extends Gender
case object FEMALE extends Gender
case object BIGENDER extends Gender
case object UNKNOWNGENDER extends Gender

case class BoolType() extends Type { UIntType(IntWidth(1)) } 
case class WRef(name:String,tpe:Type,kind:Kind,gender:Gender) extends Expression
case class WSubField(exp:Expression,name:String,tpe:Type,gender:Gender) extends Expression
case class WSubIndex(exp:Expression,value:BigInt,tpe:Type,gender:Gender) extends Expression
case class WSubAccess(exp:Expression,index:Expression,tpe:Type,gender:Gender) extends Expression
case class WVoid() extends Expression
case class WInvalid() extends Expression

case class WDefInstance(info:Info,name:String,module:String,tpe:Type) extends Stmt 

case class VarWidth(name:String) extends Width
case class PlusWidth(arg1:Width,arg2:Width) extends Width
case class MinusWidth(arg1:Width,arg2:Width) extends Width
case class MaxWidth(args:Seq[Width]) extends Width
case class MinWidth(args:Seq[Width]) extends Width
case class ExpWidth(arg1:Width) extends Width
//case class IntWidth(width: BigInt) extends Width 
//case object UnknownWidth extends Width

