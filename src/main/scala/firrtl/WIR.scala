
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

case class WRef(name:String,tpe:Type,kind:Kind,gender:Gender) extends Expression
case class WSubField(exp:Expression,name:String,tpe:Type,gender:Gender) extends Expression
case class WSubIndex(exp:Expression,value:Int,tpe:Type,gender:Gender) extends Expression
case class WSubAccess(exp:Expression,index:Expression,tpe:Type,gender:Gender) extends Expression
case class WVoid() extends Expression
case class WInvalid() extends Expression
case class WDefInstance(info:Info,name:String,module:String,tpe:Type) extends Stmt 

class WrappedExpression (val e1:Expression) {
   override def equals (we:Any) = {
      we match {
         case (we:WrappedExpression) => {
            (e1,we.e1) match {
               case (e1:UIntValue,e2:UIntValue) => if (e1.value == e2.value) true else false
                  // TODO is this necessary? width(e1) == width(e2) 
               case (e1:SIntValue,e2:SIntValue) => if (e1.value == e2.value) true else false
                  // TODO is this necessary? width(e1) == width(e2) 
               case (e1:WRef,e2:WRef) => e1.name equals e2.name
               case (e1:WSubField,e2:WSubField) => (e1.name equals e2.name) && (e1.exp == e2.exp)
               case (e1:WSubIndex,e2:WSubIndex) => (e1.value == e2.value) && (e1.exp == e2.exp)
               case (e1:WSubAccess,e2:WSubAccess) => (e1.index == e2.index) && (e1.exp == e2.exp)
               case (e1:WVoid,e2:WVoid) => true
               case (e1:WInvalid,e2:WInvalid) => true
               case (e1:DoPrim,e2:DoPrim) => {
                  var are_equal = e1.op == e2.op
                  (e1.args,e2.args).zipped.foreach{ (x,y) => { if (x != y) are_equal = false }}
                  (e1.consts,e2.consts).zipped.foreach{ (x,y) => { if (x != y) are_equal = false }}
                  are_equal
               }
               case (e1:Mux,e2:Mux) => (e1.cond == e2.cond) && (e1.tval == e2.tval) && (e1.fval == e2.fval)
               case (e1:ValidIf,e2:ValidIf) => (e1.cond == e2.cond) && (e1.value == e2.value)
               case (e1,e2) => false
            }
         }
         case _ => false
      }
   }
   override def hashCode = e1.serialize().hashCode
   override def toString = e1.serialize()
}
      

case class VarWidth(name:String) extends Width
case class PlusWidth(arg1:Width,arg2:Width) extends Width
case class MinusWidth(arg1:Width,arg2:Width) extends Width
case class MaxWidth(args:Seq[Width]) extends Width
case class MinWidth(args:Seq[Width]) extends Width
case class ExpWidth(arg1:Width) extends Width
//case class IntWidth(width: BigInt) extends Width 
//case object UnknownWidth extends Width

