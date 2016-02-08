
package firrtl

import scala.collection.Seq
import Utils._
import WrappedExpression._

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

case object ADDW_OP extends PrimOp 
case object SUBW_OP extends PrimOp

object WrappedExpression {
   def apply (e:Expression) = new WrappedExpression(e)
   def we (e:Expression) = new WrappedExpression(e)
   def weq (e1:Expression,e2:Expression) = we(e1) == we(e2)
}
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
               case (e1:WSubField,e2:WSubField) => (e1.name equals e2.name) && weq(e1.exp,e2.exp)
               case (e1:WSubIndex,e2:WSubIndex) => (e1.value == e2.value) && weq(e1.exp,e2.exp)
               case (e1:WSubAccess,e2:WSubAccess) => weq(e1.index,e2.index) && weq(e1.exp,e2.exp)
               case (e1:WVoid,e2:WVoid) => true
               case (e1:WInvalid,e2:WInvalid) => true
               case (e1:DoPrim,e2:DoPrim) => {
                  var are_equal = e1.op == e2.op
                  (e1.args,e2.args).zipped.foreach{ (x,y) => { if (!weq(x,y)) are_equal = false }}
                  (e1.consts,e2.consts).zipped.foreach{ (x,y) => { if (x != y) are_equal = false }}
                  are_equal
               }
               case (e1:Mux,e2:Mux) => weq(e1.cond,e2.cond) && weq(e1.tval,e2.tval) && weq(e1.fval,e2.fval)
               case (e1:ValidIf,e2:ValidIf) => weq(e1.cond,e2.cond) && weq(e1.value,e2.value)
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

class WrappedType (val t:Type) {
   def wt (tx:Type) = new WrappedType(tx)
   override def equals (o:Any) : Boolean = {
      o match {
         case (t2:WrappedType) => {
            (t,t2.t) match {
               case (t1:UIntType,t2:UIntType) => true
               case (t1:SIntType,t2:SIntType) => true
               case (t1:ClockType,t2:ClockType) => true
               case (t1:VectorType,t2:VectorType) => (wt(t1.tpe) == wt(t2.tpe) && t1.size == t2.size)
               case (t1:BundleType,t2:BundleType) => {
                  var ret = true
                  (t1.fields,t2.fields).zipped.foreach{ (f1,f2) => {
                     if (f1.flip != f2.flip) ret = false
                     if (f1.name != f2.name) ret = false
                     if (wt(f1.tpe) != wt(f2.tpe)) ret = false
                  }}
                  ret
               }
               case (t1,t2) => false
            }
         }
         case _ => false
      }
   }
}
class WrappedWidth (val w:Width) {
   override def toString = {
      w match {
         case (w:VarWidth) => w.name
         case (w:MaxWidth) => "max(" + w.args.map(_.toString).reduce(_ + _) + ")"
         case (w:MinWidth) => "min(" + w.args.map(_.toString).reduce(_ + _) + ")"
         case (w:PlusWidth) => "(" + w.arg1 + " + " + w.arg2 + ")"
         case (w:MinusWidth) => "(" + w.arg1 + " - " + w.arg2 + ")"
         case (w:ExpWidth) => "exp(" + w.arg1 + ")"
         case (w:IntWidth) => w.width.toString
         case (w:UnknownWidth) => "?"
      }
   }
   def eq (w1:Width,w2:Width) : Boolean = {
      (new WrappedWidth(w1)) == (new WrappedWidth(w2))
   }
   def ww (w:Width) : WrappedWidth = new WrappedWidth(w)
   override def equals (o:Any) : Boolean = {
      o match {
         case (w2:WrappedWidth) => {
            (w,w2.w) match {
               case (w1:VarWidth,w2:VarWidth) => w1.name.equals(w2.name)
               case (w1:MaxWidth,w2:MaxWidth) => {
                  var ret = true
                  if (w1.args.size != w2.args.size) ret = false
                  else {
                     for (a1 <- w1.args) {
                        var found = false
                        for (a2 <- w2.args) { if (eq(a1,a2)) found = true }
                        if (found == false) ret = false
                     }
                  }
                  ret
               }
               case (w1:MinWidth,w2:MinWidth) => {
                  var ret = true
                  if (w1.args.size != w2.args.size) ret = false
                  else {
                     for (a1 <- w1.args) {
                        var found = false
                        for (a2 <- w2.args) { if (eq(a1,a2)) found = true }
                        if (found == false) ret = false
                     }
                  }
                  ret
               }
               case (w1:IntWidth,w2:IntWidth) => w1.width == w2.width
               case (w1:PlusWidth,w2:PlusWidth) => 
                  (ww(w1.arg1) == ww(w2.arg1) && ww(w1.arg2) == ww(w2.arg2)) || (ww(w1.arg1) == ww(w2.arg2) && ww(w1.arg2) == ww(w2.arg1))
               case (w1:MinusWidth,w2:MinusWidth) => 
                  (ww(w1.arg1) == ww(w2.arg1) && ww(w1.arg2) == ww(w2.arg2)) || (ww(w1.arg1) == ww(w2.arg2) && ww(w1.arg2) == ww(w2.arg1))
               case (w1:ExpWidth,w2:ExpWidth) => ww(w1.arg1) == ww(w2.arg1)
               case (w1:UnknownWidth,w2:UnknownWidth) => true
               case (w1,w2) => false
            }
         }
         case _ => false
      }
   }
}

trait Constraint
class WGeq(val loc:Width,val exp:Width) extends Constraint {
   override def toString = {
      val wloc = new WrappedWidth(loc)
      val wexp = new WrappedWidth(exp)
      wloc.toString + " >= " + wexp.toString
   }
}
object WGeq {
   def apply (loc:Width,exp:Width) = new WGeq(loc,exp)
}


