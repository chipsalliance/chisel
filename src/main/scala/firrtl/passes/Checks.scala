/*
Copyright (c) 2014 - 2016 The Regents of the University of
California (Regents). All Rights Reserved.  Redistribution and use in
source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
   * Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer in the documentation and/or other materials
     provided with the distribution.
   * Neither the name of the Regents nor the names of its contributors
     may be used to endorse or promote products derived from this
     software without specific prior written permission.
IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS.
*/

package firrtl.passes

import com.typesafe.scalalogging.LazyLogging

// Datastructures
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.Serialize._
import firrtl.PrimOps._
import firrtl.WrappedType._

object CheckHighForm extends Pass with LazyLogging {
  def name = "High Form Check"

  // Custom Exceptions
  class NotUniqueException(name: String) extends PassException(s"${sinfo}: [module ${mname}] Reference ${name} does not have a unique name.")
  class InvalidLOCException extends PassException(s"${sinfo}: [module ${mname}] Invalid connect to an expression that is not a reference or a WritePort.")
  class NegUIntException extends PassException(s"${sinfo}: [module ${mname}] UIntLiteral cannot be negative.")
  class UndeclaredReferenceException(name: String) extends PassException(s"${sinfo}: [module ${mname}] Reference ${name} is not declared.")
  class PoisonWithFlipException(name: String) extends PassException(s"${sinfo}: [module ${mname}] Poison ${name} cannot be a bundle type with flips.")
  class MemWithFlipException(name: String) extends PassException(s"${sinfo}: [module ${mname}] Memory ${name} cannot be a bundle type with flips.")
  class InvalidAccessException extends PassException(s"${sinfo}: [module ${mname}] Invalid access to non-reference.")
  class NoTopModuleException(name: String) extends PassException(s"${sinfo}: A single module must be named ${name}.")
  class ModuleNotDefinedException(name: String) extends PassException(s"${sinfo}: Module ${name} is not defined.")
  class IncorrectNumArgsException(op: String, n: Int) extends PassException(s"${sinfo}: [module ${mname}] Primop ${op} requires ${n} expression arguments.")
  class IncorrectNumConstsException(op: String, n: Int) extends PassException(s"${sinfo}: [module ${mname}] Primop ${op} requires ${n} integer arguments.")
  class NegWidthException extends PassException(s"${sinfo}: [module ${mname}] Width cannot be negative or zero.")
  class NegVecSizeException extends PassException(s"${sinfo}: [module ${mname}] Vector type size cannot be negative.")
  class NegMemSizeException extends PassException(s"${sinfo}: [module ${mname}] Memory size cannot be negative or zero.")
  class BadPrintfException(x: Char) extends PassException(s"${sinfo}: [module ${mname}] Bad printf format: " + "\"%" + x + "\"")
  class BadPrintfTrailingException extends PassException(s"${sinfo}: [module ${mname}] Bad printf format: trailing " + "\"%\"")
  class BadPrintfIncorrectNumException extends PassException(s"${sinfo}: [module ${mname}] Bad printf format: incorrect number of arguments")

  // Utility functions
  def hasFlip(t: Type): Boolean = {
    var has = false
    def findFlip(t: Type): Type = {
      t map (findFlip) match {
        case t: BundleType => {
          for (f <- t.fields) {
            if (f.flip == Flip) has = true
          }
          t
        }
        case t: Type => t
      }
    }
    findFlip(t)
    has
  }

  // TODO FIXME
  // - Do we need to check for uniquness on port names?
  // Global Variables
  private var mname: String = ""
  private var sinfo: Info = NoInfo
  def run (c:Circuit): Circuit = {
    val errors = new Errors()
    def checkHighFormPrimop(e: DoPrim) = {
      def correctNum(ne: Option[Int], nc: Int) = {
        ne match {
          case Some(i) => if(e.args.length != i) errors.append(new IncorrectNumArgsException(e.op.toString, i))
          case None => // Do Nothing
        }
        if (e.consts.length != nc) errors.append(new IncorrectNumConstsException(e.op.toString, nc))
      }

      e.op match {
        case Add => correctNum(Option(2),0)
        case Sub => correctNum(Option(2),0)
        case Mul => correctNum(Option(2),0)
        case Div => correctNum(Option(2),0)
        case Rem => correctNum(Option(2),0)
        case Lt => correctNum(Option(2),0)
        case Leq => correctNum(Option(2),0)
        case Gt => correctNum(Option(2),0)
        case Geq => correctNum(Option(2),0)
        case Eq => correctNum(Option(2),0)
        case Neq => correctNum(Option(2),0)
        case Pad => correctNum(Option(1),1)
        case AsUInt => correctNum(Option(1),0)
        case AsSInt => correctNum(Option(1),0)
        case AsClock => correctNum(Option(1),0)
        case Shl => correctNum(Option(1),1)
        case Shr => correctNum(Option(1),1)
        case Dshl => correctNum(Option(2),0)
        case Dshr => correctNum(Option(2),0)
        case Cvt => correctNum(Option(1),0)
        case Neg => correctNum(Option(1),0)
        case Not => correctNum(Option(1),0)
        case And => correctNum(Option(2),0)
        case Or => correctNum(Option(2),0)
        case Xor => correctNum(Option(2),0)
        case Andr => correctNum(None,0)
        case Orr => correctNum(None,0)
        case Xorr => correctNum(None,0)
        case Cat => correctNum(Option(2),0)
        case Bits => correctNum(Option(1),2)
        case Head => correctNum(Option(1),1)
        case Tail => correctNum(Option(1),1)
      }
    }

    def checkFstring(s: StringLit, i: Int) = {
      val validFormats = "bdxc"
      var percent = false
      var npercents = 0
      s.array.foreach { b =>
        if (percent) {
          if (validFormats.contains(b)) npercents += 1
          else if (b != '%') errors.append(new BadPrintfException(b.toChar))
        }
        percent = if (b == '%') !percent else false // %% -> percent = false
      }
      if (percent) errors.append(new BadPrintfTrailingException)
      if (npercents != i) errors.append(new BadPrintfIncorrectNumException)
    }
    def checkValidLoc(e: Expression) = {
      e match {
        case e @ (_: UIntLiteral | _: SIntLiteral | _: DoPrim ) => errors.append(new InvalidLOCException)
        case _ => // Do Nothing
      }
    }
    def checkHighFormW(w: Width): Width = {
      w match {
        case w: IntWidth => 
          if (w.width <= BigInt(0)) errors.append(new NegWidthException)
        case _ => // Do Nothing
      }
      w
    }
    def checkHighFormT(t: Type): Type = {
      t map (checkHighFormT) match {
        case t: VectorType => 
          if (t.size < 0) errors.append(new NegVecSizeException)
        case _ => // Do nothing
      }
      t map (checkHighFormW)
    }

    def checkHighFormM(m: DefModule): DefModule = {
      val names = HashMap[String, Boolean]()
      val mnames = HashMap[String, Boolean]()
      def checkHighFormE(e: Expression): Expression = {
        def validSubexp(e: Expression): Expression = {
          e match {
            case (_:WRef|_:WSubField|_:WSubIndex|_:WSubAccess|_:Mux|_:ValidIf) => {} // No error
            case _ => errors.append(new InvalidAccessException)
          }
          e
        }
        e map (checkHighFormE) match {
          case e: WRef => 
            if (!names.contains(e.name)) errors.append(new UndeclaredReferenceException(e.name))
          case e: DoPrim => checkHighFormPrimop(e)
          case (_:Mux|_:ValidIf) => {}
          case e: WSubAccess => {
            validSubexp(e.exp)
            e
          }
          case e: UIntLiteral =>
            if (e.value < 0) errors.append(new NegUIntException)
          case e => e map (validSubexp)
        }
        e map (checkHighFormW)
        e map (checkHighFormT)
        e
      }
      def checkHighFormS(s: Statement): Statement = {
        def checkName(name: String): String = {
          if (names.contains(name)) errors.append(new NotUniqueException(name))
          else names(name) = true
          name 
        }
        sinfo = s.getInfo

        s map (checkName)
        s map (checkHighFormT)
        s map (checkHighFormE)
        s match {
          case s: DefMemory => { 
            if (hasFlip(s.dataType)) errors.append(new MemWithFlipException(s.name))
            if (s.depth <= 0) errors.append(new NegMemSizeException)
          }
          case s: WDefInstance => { 
            if (!c.modules.map(_.name).contains(s.module))
              errors.append(new ModuleNotDefinedException(s.module))
          }
          case s: Connect => checkValidLoc(s.loc)
          case s: PartialConnect => checkValidLoc(s.loc)
          case s: Print => checkFstring(s.string, s.args.length)
          case _ => // Do Nothing
        }

        s map (checkHighFormS)
      }

      mname = m.name
      for (m <- c.modules) {
        mnames(m.name) = true
      }
      for (p <- m.ports) {
        // FIXME should we set sinfo here?
        names(p.name) = true
        val tpe = p.getType
        tpe map (checkHighFormT)
        tpe map (checkHighFormW)
      }

      m match {
        case m: Module => checkHighFormS(m.body)
        case m: ExtModule => // Do Nothing
      }
      m
    }
    
    var numTopM = 0
    for (m <- c.modules) {
      if (m.name == c.main) numTopM = numTopM + 1
      checkHighFormM(m)
    }
    sinfo = c.info
    if (numTopM != 1) errors.append(new NoTopModuleException(c.main))
    errors.trigger
    c
  }
}

object CheckTypes extends Pass with LazyLogging {
   def name = "Check Types"
   var mname = ""

  // Custom Exceptions
   class SubfieldNotInBundle(info:Info, name:String) extends PassException(s"${info}: [module ${mname} ]  Subfield ${name} is not in bundle.")
   class SubfieldOnNonBundle(info:Info, name:String) extends PassException(s"${info}: [module ${mname}]  Subfield ${name} is accessed on a non-bundle.")
   class IndexTooLarge(info:Info, value:Int) extends PassException(s"${info}: [module ${mname}]  Index with value ${value} is too large.")
   class IndexOnNonVector(info:Info) extends PassException(s"${info}: [module ${mname}]  Index illegal on non-vector type.")
   class AccessIndexNotUInt(info:Info) extends PassException(s"${info}: [module ${mname}]  Access index must be a UInt type.")
   class IndexNotUInt(info:Info) extends PassException(s"${info}: [module ${mname}]  Index is not of UIntType.")
   class EnableNotUInt(info:Info) extends PassException(s"${info}: [module ${mname}]  Enable is not of UIntType.")
   class InvalidConnect(info:Info, lhs:String, rhs:String) extends PassException(s"${info}: [module ${mname}]  Type mismatch. Cannot connect ${lhs} to ${rhs}.")
   class InvalidRegInit(info:Info) extends PassException(s"${info}: [module ${mname}]  Type of init must match type of DefRegister.")
   class PrintfArgNotGround(info:Info) extends PassException(s"${info}: [module ${mname}]  Printf arguments must be either UIntType or SIntType.")
   class ReqClk(info:Info) extends PassException(s"${info}: [module ${mname}]  Requires a clock typed signal.")
   class EnNotUInt(info:Info) extends PassException(s"${info}: [module ${mname}]  Enable must be a UIntType typed signal.")
   class PredNotUInt(info:Info) extends PassException(s"${info}: [module ${mname}]  Predicate not a UIntType.")
   class OpNotGround(info:Info, op:String) extends PassException(s"${info}: [module ${mname}]  Primop ${op} cannot operate on non-ground types.")
   class OpNotUInt(info:Info, op:String,e:String) extends PassException(s"${info}: [module ${mname}]  Primop ${op} requires argument ${e} to be a UInt type.")
   class OpNotAllUInt(info:Info, op:String) extends PassException(s"${info}: [module ${mname}]  Primop ${op} requires all arguments to be UInt type.")
   class OpNotAllSameType(info:Info, op:String) extends PassException(s"${info}: [module ${mname}]  Primop ${op} requires all operands to have the same type.")
   class NodePassiveType(info:Info) extends PassException(s"${info}: [module ${mname}]  Node must be a passive type.")
   class MuxSameType(info:Info) extends PassException(s"${info}: [module ${mname}]  Must mux between equivalent types.")
   class MuxPassiveTypes(info:Info) extends PassException(s"${info}: [module ${mname}]  Must mux between passive types.")
   class MuxCondUInt(info:Info) extends PassException(s"${info}: [module ${mname}]  A mux condition must be of type UInt.")
   class ValidIfPassiveTypes(info:Info) extends PassException(s"${info}: [module ${mname}]  Must validif a passive type.")
   class ValidIfCondUInt(info:Info) extends PassException(s"${info}: [module ${mname}]  A validif condition must be of type UInt.")
   //;---------------- Helper Functions --------------
   def ut () : UIntType = UIntType(UnknownWidth)
   def st () : SIntType = SIntType(UnknownWidth)
   
   def check_types_primop (e:DoPrim, errors:Errors, info:Info) : Unit = {
      def all_same_type (ls:Seq[Expression]) : Unit = {
         var error = false
         for (x <- ls) {
            if (wt(tpe(ls.head)) != wt(tpe(x))) error = true
         }
         if (error) errors.append(new OpNotAllSameType(info,e.op.serialize))
      }
      def all_ground (ls:Seq[Expression]) : Unit = {
         var error = false
         for (x <- ls ) {
            if (!(tpe(x).typeof[UIntType] || tpe(x).typeof[SIntType])) error = true
         }
         if (error) errors.append(new OpNotGround(info,e.op.serialize))
      }
      def all_uint (ls:Seq[Expression]) : Unit = {
         var error = false
         for (x <- ls ) {
            if (!(tpe(x).typeof[UIntType])) error = true
         }
         if (error) errors.append(new OpNotAllUInt(info,e.op.serialize))
      }
      def is_uint (x:Expression) : Unit = {
         var error = false
         if (!(tpe(x).typeof[UIntType])) error = true
         if (error) errors.append(new OpNotUInt(info,e.op.serialize,x.serialize))
      }
      
      e.op match {
         case AsUInt =>
         case AsSInt =>
         case AsClock =>
         case Dshl => is_uint(e.args(1)); all_ground(e.args)
         case Dshr => is_uint(e.args(1)); all_ground(e.args)
         case Add => all_ground(e.args)
         case Sub => all_ground(e.args)
         case Mul => all_ground(e.args)
         case Div => all_ground(e.args)
         case Rem => all_ground(e.args)
         case Lt => all_ground(e.args)
         case Leq => all_ground(e.args)
         case Gt => all_ground(e.args)
         case Geq => all_ground(e.args)
         case Eq => all_ground(e.args)
         case Neq => all_ground(e.args)
         case Pad => all_ground(e.args)
         case Shl => all_ground(e.args)
         case Shr => all_ground(e.args)
         case Cvt => all_ground(e.args)
         case Neg => all_ground(e.args)
         case Not => all_ground(e.args)
         case And => all_ground(e.args)
         case Or => all_ground(e.args)
         case Xor => all_ground(e.args)
         case Andr => all_ground(e.args)
         case Orr => all_ground(e.args)
         case Xorr => all_ground(e.args)
         case Cat => all_ground(e.args)
         case Bits => all_ground(e.args)
         case Head => all_ground(e.args)
         case Tail => all_ground(e.args)
      }
   }
      
   def run (c:Circuit) : Circuit = {
      val errors = new Errors()
      def passive (t:Type) : Boolean = {
         (t) match { 
            case (_:UIntType|_:SIntType) => true
            case (t:VectorType) => passive(t.tpe)
            case (t:BundleType) => {
               var p = true
               for (x <- t.fields ) {
                  if (x.flip == Flip) p = false
                  if (!passive(x.tpe)) p = false
               }
               p
            }
            case (t) => true
         }
      }
      def check_types_e (info:Info)(e:Expression) : Expression = {
         (e map (check_types_e(info))) match { 
            case (e:WRef) => e
            case (e:WSubField) => {
               (tpe(e.exp)) match  { 
                  case (t:BundleType) => {
                     val ft = t.fields.find(p => p.name == e.name)
                     if (ft == None) errors.append(new SubfieldNotInBundle(info,e.name))
                  }
                  case (t) => errors.append(new SubfieldOnNonBundle(info,e.name))
               }
            }
            case (e:WSubIndex) => {
               (tpe(e.exp)) match { 
                  case (t:VectorType) => {
                     if (e.value >= t.size) errors.append(new IndexTooLarge(info,e.value))
                  }
                  case (t) => errors.append(new IndexOnNonVector(info))
               }
            }
            case (e:WSubAccess) => {
               (tpe(e.exp)) match { 
                  case (t:VectorType) => false
                  case (t) => errors.append(new IndexOnNonVector(info))
               }
               (tpe(e.index)) match { 
                  case (t:UIntType) => false
                  case (t) => errors.append(new AccessIndexNotUInt(info))
               }
            }
            case (e:DoPrim) => check_types_primop(e,errors,info)
            case (e:Mux) => {
               if (wt(tpe(e.tval)) != wt(tpe(e.fval))) errors.append(new MuxSameType(info))
               if (!passive(tpe(e))) errors.append(new MuxPassiveTypes(info))
               if (!passive(tpe(e))) errors.append(new MuxPassiveTypes(info))
               if (!(tpe(e.cond).typeof[UIntType])) errors.append(new MuxCondUInt(info))
            }
            case (e:ValidIf) => {
               if (!passive(tpe(e))) errors.append(new ValidIfPassiveTypes(info))
               if (!(tpe(e.cond).typeof[UIntType])) errors.append(new ValidIfCondUInt(info))
            }
            case (_:UIntLiteral | _:SIntLiteral) => false
         }
         e
      }
   
      def bulk_equals (t1: Type, t2: Type, flip1: Orientation, flip2: Orientation): Boolean = {
         //;println_all(["Inside with t1:" t1 ",t2:" t2 ",f1:" flip1 ",f2:" flip2])
         (t1,t2) match {
            case (ClockType, ClockType) => flip1 == flip2
            case (t1:UIntType,t2:UIntType) => flip1 == flip2
            case (t1:SIntType,t2:SIntType) => flip1 == flip2
            case (t1:BundleType,t2:BundleType) => {
               var isEqual = true
               for (i <- 0 until t1.fields.size) {
                  for (j <- 0 until t2.fields.size) {
                     val f1 = t1.fields(i)
                     val f2 = t2.fields(j)
                     if (f1.name == f2.name) {
                        val field_equal = bulk_equals(f1.tpe,f2.tpe,times(flip1, f1.flip),times(flip2, f2.flip))
                        if (!field_equal) isEqual = false
                     }
                  }
               }
               isEqual
            }
            case (t1:VectorType,t2:VectorType) => bulk_equals(t1.tpe,t2.tpe,flip1,flip2)
         }
      }

      def check_types_s (s:Statement) : Statement = {
         s map (check_types_e(get_info(s))) match { 
            case (s:Connect) => if (wt(tpe(s.loc)) != wt(tpe(s.expr))) errors.append(new InvalidConnect(s.info, s.loc.serialize, s.expr.serialize))
            case (s:DefRegister) => if (wt(s.tpe) != wt(tpe(s.init))) errors.append(new InvalidRegInit(s.info))
            case (s:PartialConnect) => if (!bulk_equals(tpe(s.loc),tpe(s.expr),Default,Default) ) errors.append(new InvalidConnect(s.info, s.loc.serialize, s.expr.serialize))
            case (s:Stop) => {
               if (wt(tpe(s.clk)) != wt(ClockType) ) errors.append(new ReqClk(s.info))
               if (wt(tpe(s.en)) != wt(ut()) ) errors.append(new EnNotUInt(s.info))
            }
            case (s:Print)=> {
               for (x <- s.args ) {
                  if (wt(tpe(x)) != wt(ut()) && wt(tpe(x)) != wt(st()) ) errors.append(new PrintfArgNotGround(s.info))
               }
               if (wt(tpe(s.clk)) != wt(ClockType) ) errors.append(new ReqClk(s.info))
               if (wt(tpe(s.en)) != wt(ut()) ) errors.append(new EnNotUInt(s.info))
            }
            case (s:Conditionally) => if (wt(tpe(s.pred)) != wt(ut()) ) errors.append(new PredNotUInt(s.info))
            case (s:DefNode) => if (!passive(tpe(s.value)) ) errors.append(new NodePassiveType(s.info))
            case (s) => false
         }
         s map (check_types_s)
      }
      
      for (m <- c.modules ) {
         mname = m.name
         (m) match { 
            case (m:ExtModule) => false
            case (m:Module) => check_types_s(m.body)
         }
      }
      errors.trigger
      c
   }
}

object CheckGenders extends Pass {
   def name = "Check Genders"
   var mname = ""
   class WrongGender (info:Info,expr:String,wrong:String,right:String) extends PassException(s"${info}: [module ${mname}]  Expression ${expr} is used as a ${wrong} but can only be used as a ${right}.")
   
   def dir_to_gender (d:Direction) : Gender = {
      d match {
         case Input => MALE
         case Output => FEMALE //BI-GENDER
      }
   }
   
   def as_srcsnk (g:Gender) : String = {
      g match {
         case MALE => "source"
         case FEMALE => "sink"
         case UNKNOWNGENDER => "unknown"
         case BIGENDER => "sourceOrSink"
      }
   }
   
   def run (c:Circuit): Circuit = {
      val errors = new Errors()
      def get_kind (e:Expression) : Kind = {
         (e) match { 
            case (e:WRef) => e.kind
            case (e:WSubField) => get_kind(e.exp)
            case (e:WSubIndex) => get_kind(e.exp)
            case (e:WSubAccess) => get_kind(e.exp)
            case (e) => NodeKind()
         }
      }
   
      def check_gender (info:Info,genders:HashMap[String,Gender],desired:Gender)(e:Expression) : Expression = {
         val gender = get_gender(e,genders)
         val kindx = get_kind(e)
         def flipQ (t:Type) : Boolean = {
            var fQ = false
            def flip_rec (t:Type,f:Orientation) : Type = {
               (t) match { 
                  case (t:BundleType) => {
                     for (field <- t.fields) {
                        flip_rec(field.tpe,times(f, field.flip))
                     }
                  }
                  case (t:VectorType) => flip_rec(t.tpe,f)
                  case (t) => if (f == Flip) fQ = true
               }
               t
            }
            flip_rec(t,Default)
            fQ
         }
            
         val has_flipQ = flipQ(tpe(e))
         //println(e)
         //println(gender)
         //println(desired)
         //println(kindx)
         //println(desired == gender)
         //if gender != desired and gender != BI-GENDER:
         (gender,desired) match {
            case (MALE, FEMALE) => errors.append(new WrongGender(info,e.serialize,as_srcsnk(desired),as_srcsnk(gender)))
            case (FEMALE, MALE) =>
               if ((kindx == PortKind() || kindx == InstanceKind()) && has_flipQ == false) {
                  //; OK!
                  false
               } else {
                  //; Not Ok!
                  errors.append(new WrongGender(info,e.serialize,as_srcsnk(desired),as_srcsnk(gender)))
               }
            case _ => false
         }
         e
      }
   
      def get_gender (e:Expression,genders:HashMap[String,Gender]) : Gender = {
         (e) match { 
            case (e:WRef) => genders(e.name)
            case (e:WSubField) => 
               val f = tpe(e.exp).as[BundleType].get.fields.find(f => f.name == e.name).get
               times(get_gender(e.exp,genders),f.flip)
            case (e:WSubIndex) => get_gender(e.exp,genders)
            case (e:WSubAccess) => get_gender(e.exp,genders)
            case (e:DoPrim) => MALE
            case (e:UIntLiteral) => MALE
            case (e:SIntLiteral) => MALE
            case (e:Mux) => MALE
            case (e:ValidIf) => MALE
         }
      }
   
      def check_genders_e (info:Info,genders:HashMap[String,Gender])(e:Expression) : Expression = {
         e map (check_genders_e(info,genders))
         (e) match { 
            case (e:WRef) => false
            case (e:WSubField) => false
            case (e:WSubIndex) => false
            case (e:WSubAccess) => false
            case (e:DoPrim) => for (e <- e.args ) { check_gender(info,genders,MALE)(e) }
            case (e:Mux) => e map (check_gender(info,genders,MALE))
            case (e:ValidIf) => e map (check_gender(info,genders,MALE))
            case (e:UIntLiteral) => false
            case (e:SIntLiteral) => false
         }
         e
      }
        
      def check_genders_s (genders:HashMap[String,Gender])(s:Statement) : Statement = {
         s map (check_genders_e(get_info(s),genders))
         s map (check_genders_s(genders))
         (s) match { 
            case (s:DefWire) => genders(s.name) = BIGENDER
            case (s:DefRegister) => genders(s.name) = BIGENDER
            case (s:DefNode) => {
               check_gender(s.info,genders,MALE)(s.value)
               genders(s.name) = MALE
            }
            case (s:DefMemory) => genders(s.name) = MALE
            case (s:WDefInstance) => genders(s.name) = MALE
            case (s:Connect) => {
               check_gender(s.info,genders,FEMALE)(s.loc)
               check_gender(s.info,genders,MALE)(s.expr)
            }
            case (s:Print) => {
               for (x <- s.args ) {
                  check_gender(s.info,genders,MALE)(x)
               }
               check_gender(s.info,genders,MALE)(s.en)
               check_gender(s.info,genders,MALE)(s.clk)
            }
            case (s:PartialConnect) => {
               check_gender(s.info,genders,FEMALE)(s.loc)
               check_gender(s.info,genders,MALE)(s.expr)
            }
            case (s:Conditionally) => {
               check_gender(s.info,genders,MALE)(s.pred)
            }
            case EmptyStmt => false
            case (s:Stop) => {
               check_gender(s.info,genders,MALE)(s.en)
               check_gender(s.info,genders,MALE)(s.clk)
            }
            case (_:Begin|_:IsInvalid) => false
         }
         s
      }
   
      for (m <- c.modules ) {
         mname = m.name
         val genders = HashMap[String,Gender]()
         for (p <- m.ports) {
            genders(p.name) = dir_to_gender(p.direction)
         }
         (m) match { 
            case (m:ExtModule) => false
            case (m:Module) => check_genders_s(genders)(m.body)
         }
      }
      errors.trigger
      c
   }
}

object CheckWidths extends Pass {
   def name = "Width Check"
   var mname = ""
   class UninferredWidth (info:Info) extends PassException(s"${info} : [module ${mname}]  Uninferred width.")
   class WidthTooSmall(info: Info, b: BigInt) extends PassException(
         s"$info : [module $mname]  Width too small for constant " +
         Serialize().serialize(b) + ".")
   class NegWidthException(info:Info) extends PassException(s"${info}: [module ${mname}] Width cannot be negative or zero.")
   def run (c:Circuit): Circuit = {
      val errors = new Errors()
      def check_width_m (m:DefModule) : Unit = {
         def check_width_w (info:Info)(w:Width) : Width = {
            (w) match { 
               case (w:IntWidth)=> if (w.width <= 0) errors.append(new NegWidthException(info))
               case (w) => errors.append(new UninferredWidth(info))
            }
            w
         }
         def check_width_e (info:Info)(e:Expression) : Expression = {
            (e map (check_width_e(info))) match { 
               case (e:UIntLiteral) => {
                  (e.width) match { 
                     case (w:IntWidth) => 
                        if (scala.math.max(1,e.value.bitLength) > w.width) {
                           errors.append(new WidthTooSmall(info, e.value))
                        }
                     case (w) => errors.append(new UninferredWidth(info))
                  }
                  check_width_w(info)(e.width)
               }
               case (e:SIntLiteral) => {
                  (e.width) match { 
                     case (w:IntWidth) => 
                        if (e.value.bitLength + 1 > w.width) errors.append(new WidthTooSmall(info, e.value))
                     case (w) => errors.append(new UninferredWidth(info))
                  }
                  check_width_w(info)(e.width)
               }
               case (e:DoPrim) => false
               case (e) => false
            }
            e
         }
         def check_width_s (s:Statement) : Statement = {
            s map (check_width_s) map (check_width_e(get_info(s)))
            def tm (t:Type) : Type = mapr(check_width_w(info(s)) _,t)
            s map (tm)
         }
      
         for (p <- m.ports) {
            mapr(check_width_w(p.info) _,p.tpe)
         }
   
         (m) match { 
            case (m:ExtModule) => {}
            case (m:Module) => check_width_s(m.body)
         }
      }
      
      for (m <- c.modules) {
         mname = m.name
         check_width_m(m)
      }
      errors.trigger
      c
   }
}
