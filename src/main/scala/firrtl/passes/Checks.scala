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
import firrtl.Utils._
import firrtl.Serialize._
import firrtl.PrimOps._
import firrtl.WrappedType._

object CheckHighForm extends Pass with LazyLogging {
  def name = "High Form Check"

  // Custom Exceptions
  class NotUniqueException(name: String) extends PassException(s"${sinfo}: [module ${mname}] Reference ${name} does not have a unique name.")
  class IsPrefixException(prefix: String) extends PassException(s"${sinfo}: [module ${mname}] Symbol ${prefix} is a prefix.")
  class InvalidLOCException extends PassException(s"${sinfo}: [module ${mname}] Invalid connect to an expression that is not a reference or a WritePort.")
  class NegUIntException extends PassException(s"${sinfo}: [module ${mname}] UIntValue cannot be negative.")
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
  // Note the following awkward strings are due to an issue with Scala string interpolation and escaped double quotes
  class BadPrintfException(x: Char) extends PassException(s"${sinfo}: [module ${mname}] Bad printf format: " + "\"%" + x + "\"")
  class BadPrintfTrailingException extends PassException(s"${sinfo}: [module ${mname}] Bad printf format: trailing " + "\"%\"")
  class BadPrintfIncorrectNumException extends PassException(s"${sinfo}: [module ${mname}] Bad printf format: incorrect number of arguments")

  // Trie Datastructure for prefix checking
  case class Trie(var children: HashMap[String, Trie], var end: Boolean) {
    def empty: Boolean = children.isEmpty
    def add(ls: Seq[String]): Boolean = {
      var t: Trie = this
      var sawEnd = false
      for (x <- ls) {
        if (t.end) sawEnd = true
        if (t.contains(x)) t = t.children(x)
        else {
          val temp = new Trie(HashMap[String,Trie](),false)
          t.children(x) = temp
          t = temp
        }
      }
      t.end = true
      sawEnd | !t.empty
    }
    def contains(s: String): Boolean = children.contains(s)
    def contains(ls: Seq[String]): Boolean = {
      var t: Trie = this
      for (x <- ls) {
        if (t.contains(x)) t = t.children(x)
        else return false
      }
      t.end
    }
  }

  // Utility functions
  def hasFlip(t: Type): Boolean = {
    var has = false
    def findFlip(t: Type): Type = {
      t match {
        case t: BundleType => {
          for (f <- t.fields) {
            if (f.flip == REVERSE) has = true
          }
          t
        }
        case t: Type => t
      }
    }
    findFlip(t)
    tMap(findFlip _, t)
    has
  }

  // TODO FIXME
  // - Do we need to check for uniquness on port names?
  // Global Variables
  private var mname: String = ""
  private var sinfo: Info = NoInfo
  def run (c:Circuit): Circuit = {
    val errors = ArrayBuffer[PassException]()
    def checkHighFormPrimop(e: DoPrim) = {
      def correctNum(ne: Option[Int], nc: Int) = {
        ne match {
          case Some(i) => if(e.args.length != i) errors.append(new IncorrectNumArgsException(e.op.getString, i))
          case None => // Do Nothing
        }
        if (e.consts.length != nc) errors.append(new IncorrectNumConstsException(e.op.getString, nc))
      }

      e.op match {
        case ADD_OP             => correctNum(Option(2),0)
        case SUB_OP             => correctNum(Option(2),0)
        case MUL_OP             => correctNum(Option(2),0)
        case DIV_OP             => correctNum(Option(2),0)
        case REM_OP             => correctNum(Option(2),0)
        case LESS_OP            => correctNum(Option(2),0)
        case LESS_EQ_OP         => correctNum(Option(2),0)
        case GREATER_OP         => correctNum(Option(2),0)
        case GREATER_EQ_OP      => correctNum(Option(2),0)
        case EQUAL_OP           => correctNum(Option(2),0)
        case NEQUAL_OP          => correctNum(Option(2),0)
        case PAD_OP             => correctNum(Option(1),1)
        case AS_UINT_OP         => correctNum(Option(1),0)
        case AS_SINT_OP         => correctNum(Option(1),0)
        case AS_CLOCK_OP        => correctNum(Option(1),0)
        case SHIFT_LEFT_OP      => correctNum(Option(1),1)
        case SHIFT_RIGHT_OP     => correctNum(Option(1),1)
        case DYN_SHIFT_LEFT_OP  => correctNum(Option(2),0)
        case DYN_SHIFT_RIGHT_OP => correctNum(Option(2),0)
        case CONVERT_OP         => correctNum(Option(1),0)
        case NEG_OP             => correctNum(Option(1),0)
        case NOT_OP             => correctNum(Option(1),0)
        case AND_OP             => correctNum(Option(2),0)
        case OR_OP              => correctNum(Option(2),0)
        case XOR_OP             => correctNum(Option(2),0)
        case AND_REDUCE_OP      => correctNum(None,0)
        case OR_REDUCE_OP       => correctNum(None,0)
        case XOR_REDUCE_OP      => correctNum(None,0)
        case CONCAT_OP          => correctNum(Option(2),0)
        case BITS_SELECT_OP     => correctNum(Option(1),2)
        case HEAD_OP            => correctNum(Option(1),1)
        case TAIL_OP            => correctNum(Option(1),1)
      }
    }

    def checkFstring(s: String, i: Int) = {
      val validFormats = "bedxs"
      var percent = false
      var ret = true
      var npercents = 0
      for (x <- s) { 
        if (!validFormats.contains(x) && percent)
          errors.append(new BadPrintfException(x))
        if (x == '%') npercents = npercents + 1
        percent = (x == '%')
      }
      if (percent) errors.append(new BadPrintfTrailingException)
      if (npercents != i) errors.append(new BadPrintfIncorrectNumException)
    }
    def checkValidLoc(e: Expression) = {
      e match {
        case e @ ( _: UIntValue | _: SIntValue | _: DoPrim ) => errors.append(new InvalidLOCException)
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
      tMap(checkHighFormT _, t) match {
        case t: VectorType => 
          if (t.size < 0) errors.append(new NegVecSizeException)
        case _ => // Do nothing
      }
      wMap(checkHighFormW _, t)
    }

    def checkHighFormM(m: Module): Module = {
      val names = HashMap[String, Boolean]()
      val mnames = HashMap[String, Boolean]()
      val tries = Trie(HashMap[String, Trie](),false)
      def checkHighFormE(e: Expression): Expression = {
        def validSubexp(e: Expression): Expression = {
          e match {
            case (_:WRef|_:WSubField|_:WSubIndex|_:WSubAccess|_:Mux|_:ValidIf) => {} // No error
            case _ => errors.append(new InvalidAccessException)
          }
          e
        }
        eMap(checkHighFormE _, e) match {
          case e: WRef => 
            if (!names.contains(e.name)) errors.append(new UndeclaredReferenceException(e.name))
          case e: DoPrim => checkHighFormPrimop(e)
          case (_:Mux|_:ValidIf) => {}
          case e: WSubAccess => {
            validSubexp(e.exp)
            e
          }
          case e: UIntValue => 
            if (e.value < 0) errors.append(new NegUIntException)
          case e => eMap(validSubexp _, e)
        }
        wMap(checkHighFormW _, e)
        tMap(checkHighFormT _, e)
        e
      }
      def checkHighFormS(s: Stmt): Stmt = {
        def checkName(name: String): String = {
          if (names.contains(name)) errors.append(new NotUniqueException(name))
          else names(name) = true
          val ls: Seq[String] = name.split('$')
          if (tries.add(ls)) errors.append(new IsPrefixException(name))
          name 
        }
        sinfo = s.getInfo

        stMap(checkName _, s)
        tMap(checkHighFormT _, s)
        eMap(checkHighFormE _, s)
        s match {
          case s: DefPoison => {
             if (hasFlip(s.tpe)) errors.append(new PoisonWithFlipException(s.name))
             checkHighFormT(s.tpe)
          }
          case s: DefMemory => { 
            if (hasFlip(s.data_type)) errors.append(new MemWithFlipException(s.name))
            if (s.depth <= 0) errors.append(new NegMemSizeException)
          }
          case s: WDefInstance => { 
            if (!c.modules.map(_.name).contains(s.module))
              errors.append(new ModuleNotDefinedException(s.module))
          }
          case s: Connect => checkValidLoc(s.loc)
          case s: BulkConnect => checkValidLoc(s.loc)
          case s: Print => checkFstring(s.string, s.args.length)
          case _ => // Do Nothing
        }

        sMap(checkHighFormS _, s)
      }

      mname = m.name
      for (m <- c.modules) {
        mnames(m.name) = true
      }
      for (p <- m.ports) {
        // FIXME should we set sinfo here?
        names(p.name) = true
        val tpe = p.getType
        tMap(checkHighFormT _, tpe)
        wMap(checkHighFormW _, tpe)
      }

      m match {
        case m: InModule => checkHighFormS(m.body)
        case m: ExModule => // Do Nothing
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
    if (errors.nonEmpty) throw new PassExceptions(errors)
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
   class InvalidConnect(info:Info) extends PassException(s"${info}: [module ${mname}]  Type mismatch.")
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
   def ut () : UIntType = UIntType(UnknownWidth())
   def st () : SIntType = SIntType(UnknownWidth())
   
   def check_types_primop (e:DoPrim, errors:ArrayBuffer[PassException],info:Info) : Unit = {
      def all_same_type (ls:Seq[Expression]) : Unit = {
         var error = false
         for (x <- ls) {
            if (wt(tpe(ls.head)) != wt(tpe(x))) error = true
         }
         if (error) errors += new OpNotAllSameType(info,e.op.serialize)
      }
      def all_ground (ls:Seq[Expression]) : Unit = {
         var error = false
         for (x <- ls ) {
            if (!(tpe(x).typeof[UIntType] || tpe(x).typeof[SIntType])) error = true
         }
         if (error) errors += new OpNotGround(info,e.op.serialize)
      }
      def all_uint (ls:Seq[Expression]) : Unit = {
         var error = false
         for (x <- ls ) {
            if (!(tpe(x).typeof[UIntType])) error = true
         }
         if (error) errors += new OpNotAllUInt(info,e.op.serialize)
      }
      def is_uint (x:Expression) : Unit = {
         var error = false
         if (!(tpe(x).typeof[UIntType])) error = true
         if (error) errors += new OpNotUInt(info,e.op.serialize,x.serialize)
      }
      
      e.op match {
         case AS_UINT_OP =>  {}
         case AS_SINT_OP =>  {}
         case AS_CLOCK_OP =>  {}
         case DYN_SHIFT_LEFT_OP =>  is_uint(e.args(1)); all_ground(e.args)
         case DYN_SHIFT_RIGHT_OP => is_uint(e.args(1)); all_ground(e.args)
         case ADD_OP =>  all_ground(e.args)
         case SUB_OP =>  all_ground(e.args)
         case MUL_OP =>  all_ground(e.args)
         case DIV_OP =>  all_ground(e.args)
         case REM_OP =>  all_ground(e.args)
         case LESS_OP =>  all_ground(e.args)
         case LESS_EQ_OP =>  all_ground(e.args)
         case GREATER_OP =>  all_ground(e.args)
         case GREATER_EQ_OP =>  all_ground(e.args)
         case EQUAL_OP =>  all_ground(e.args)
         case NEQUAL_OP =>  all_ground(e.args)
         case PAD_OP =>  all_ground(e.args)
         case SHIFT_LEFT_OP =>  all_ground(e.args)
         case SHIFT_RIGHT_OP =>  all_ground(e.args)
         case CONVERT_OP =>  all_ground(e.args)
         case NEG_OP =>  all_ground(e.args)
         case NOT_OP =>  all_ground(e.args)
         case AND_OP =>  all_ground(e.args)
         case OR_OP =>  all_ground(e.args)
         case XOR_OP =>  all_ground(e.args)
         case AND_REDUCE_OP =>  all_ground(e.args)
         case OR_REDUCE_OP =>  all_ground(e.args)
         case XOR_REDUCE_OP =>  all_ground(e.args)
         case CONCAT_OP =>  all_ground(e.args)
         case BITS_SELECT_OP =>  all_ground(e.args)
         case HEAD_OP =>  all_ground(e.args)
         case TAIL_OP =>  all_ground(e.args)
      }
   }
      
   def run (c:Circuit) : Circuit = {
      val errors = ArrayBuffer[PassException]()
      def passive (t:Type) : Boolean = {
         (t) match { 
            case (_:UIntType|_:SIntType) => true
            case (t:VectorType) => passive(t.tpe)
            case (t:BundleType) => {
               var p = true
               for (x <- t.fields ) {
                  if (x.flip == REVERSE) p = false
                  if (!passive(x.tpe)) p = false
               }
               p
            }
            case (t) => true
         }
      }
      def check_types_e (info:Info)(e:Expression) : Expression = {
         (eMap(check_types_e(info) _,e)) match { 
            case (e:WRef) => e
            case (e:WSubField) => {
               (tpe(e.exp)) match  { 
                  case (t:BundleType) => {
                     val ft = t.fields.find(p => p.name == e.name)
                     if (ft == None) errors += new SubfieldNotInBundle(info,e.name)
                  }
                  case (t) => errors += new SubfieldOnNonBundle(info,e.name)
               }
            }
            case (e:WSubIndex) => {
               (tpe(e.exp)) match { 
                  case (t:VectorType) => {
                     if (e.value >= t.size) errors += new IndexTooLarge(info,e.value)
                  }
                  case (t) => errors += new IndexOnNonVector(info)
               }
            }
            case (e:WSubAccess) => {
               (tpe(e.exp)) match { 
                  case (t:VectorType) => false
                  case (t) => errors += new IndexOnNonVector(info)
               }
               (tpe(e.index)) match { 
                  case (t:UIntType) => false
                  case (t) => errors += new AccessIndexNotUInt(info)
               }
            }
            case (e:DoPrim) => check_types_primop(e,errors,info)
            case (e:Mux) => {
               if (wt(tpe(e.tval)) != wt(tpe(e.fval))) errors += new MuxSameType(info)
               if (!passive(tpe(e))) errors += new MuxPassiveTypes(info)
               if (!passive(tpe(e))) errors += new MuxPassiveTypes(info)
               if (!(tpe(e.cond).typeof[UIntType])) errors += new MuxCondUInt(info)
            }
            case (e:ValidIf) => {
               if (!passive(tpe(e))) errors += new ValidIfPassiveTypes(info)
               if (!(tpe(e.cond).typeof[UIntType])) errors += new ValidIfCondUInt(info)
            }
            case (_:UIntValue|_:SIntValue) => false
         }
         e
      }
   
      def bulk_equals (t1:Type,t2:Type) : Boolean = {
         (t1,t2) match { 
            case (t1:BundleType,t2:BundleType) => {
               var same = true
               (t1.fields, t2.fields).zipped.map{ (f1,f2) => {
                  if (f1.name == f2.name) {
                     if (f1.flip != f2.flip) same = false
                     if (!bulk_equals(f1.tpe,f2.tpe)) same = false
                  }
               }}
               same
            }
            case (t1:ClockType,t2:ClockType) => true
            case (t1:UIntType,t2:UIntType) => true
            case (t1:SIntType,t2:SIntType) => true
            case (t1:VectorType,t2:VectorType) => {
               if (bulk_equals(t1.tpe,t2.tpe)) true
               else false
            }
            case (t1,t2) => false
         }
      }
           
      def check_types_s (s:Stmt) : Stmt = {
         eMap(check_types_e(get_info(s)) _,s) match { 
            case (s:Connect) => if (wt(tpe(s.loc)) != wt(tpe(s.exp))) errors += new InvalidConnect(s.info)
            case (s:BulkConnect) => if (!bulk_equals(tpe(s.loc),tpe(s.exp)) ) errors += new InvalidConnect(s.info)
            case (s:Stop) => {
               if (wt(tpe(s.clk)) != wt(ClockType()) ) errors += new ReqClk(s.info)
               if (wt(tpe(s.en)) != wt(ut()) ) errors += new EnNotUInt(s.info)
            }
            case (s:Print)=> {
               for (x <- s.args ) {
                  if (wt(tpe(x)) != wt(ut()) && wt(tpe(x)) != wt(st()) ) errors += new PrintfArgNotGround(s.info)
               }
               if (wt(tpe(s.clk)) != wt(ClockType()) ) errors += new ReqClk(s.info)
               if (wt(tpe(s.en)) != wt(ut()) ) errors += new EnNotUInt(s.info)
            }
            case (s:Conditionally) => if (wt(tpe(s.pred)) != wt(ut()) ) errors += new PredNotUInt(s.info)
            case (s:DefNode) => if (!passive(tpe(s.value)) ) errors += new NodePassiveType(s.info)
            case (s) => false
         }
         sMap(check_types_s,s)
      }
      
      for (m <- c.modules ) {
         mname = m.name
         (m) match { 
            case (m:ExModule) => false
            case (m:InModule) => check_types_s(m.body)
         }
      }
      if (errors.nonEmpty) throw new PassExceptions(errors)
      c
   }
}

object CheckGenders extends Pass {
   def name = "Check Genders"
   var mname = ""
   class WrongGender (info:Info,expr:String,wrong:String,right:String) extends PassException(s"${info}: [module ${mname}]  Expression ${expr} is used as a ${wrong} but can only be used as a ${right}.")
   
   def dir_to_gender (d:Direction) : Gender = {
      d match {
         case INPUT => MALE
         case OUTPUT => FEMALE //BI-GENDER
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
      val errors = ArrayBuffer[PassException]()
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
            def flip_rec (t:Type,f:Flip) : Type = {
               (t) match { 
                  case (t:BundleType) => {
                     for (field <- t.fields) {
                        flip_rec(field.tpe,times(f, field.flip))
                     }
                  }
                  case (t:VectorType) => flip_rec(t.tpe,f)
                  case (t) => if (f == REVERSE) fQ = true
               }
               t
            }
            flip_rec(t,DEFAULT)
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
            case (MALE, FEMALE) => errors += new WrongGender(info,e.serialize,as_srcsnk(desired),as_srcsnk(gender))
            case (FEMALE, MALE) =>
               if ((kindx == PortKind() || kindx == InstanceKind()) && has_flipQ == false) {
                  //; OK!
                  false
               } else {
                  //; Not Ok!
                  errors += new WrongGender(info,e.serialize,as_srcsnk(desired),as_srcsnk(gender))
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
            case (e:UIntValue) => MALE
            case (e:SIntValue) => MALE
            case (e:Mux) => MALE
            case (e:ValidIf) => MALE
         }
      }
   
      def check_genders_e (info:Info,genders:HashMap[String,Gender])(e:Expression) : Expression = {
         eMap(check_genders_e(info,genders) _,e)
         (e) match { 
            case (e:WRef) => false
            case (e:WSubField) => false
            case (e:WSubIndex) => false
            case (e:WSubAccess) => false
            case (e:DoPrim) => for (e <- e.args ) { check_gender(info,genders,MALE)(e) }
            case (e:Mux) => eMap(check_gender(info,genders,MALE) _,e)
            case (e:ValidIf) => eMap(check_gender(info,genders,MALE) _,e)
            case (e:UIntValue) => false
            case (e:SIntValue) => false
         }
         e
      }
        
      def check_genders_s (genders:HashMap[String,Gender])(s:Stmt) : Stmt = {
         eMap(check_genders_e(get_info(s),genders) _,s)
         sMap(check_genders_s(genders) _,s)
         (s) match { 
            case (s:DefWire) => genders(s.name) = BIGENDER
            case (s:DefPoison) => genders(s.name) = MALE
            case (s:DefRegister) => genders(s.name) = BIGENDER
            case (s:DefNode) => {
               check_gender(s.info,genders,MALE)(s.value)
               genders(s.name) = MALE
            }
            case (s:DefMemory) => genders(s.name) = MALE
            case (s:WDefInstance) => genders(s.name) = MALE
            case (s:Connect) => {
               check_gender(s.info,genders,FEMALE)(s.loc)
               check_gender(s.info,genders,MALE)(s.exp)
            }
            case (s:Print) => {
               for (x <- s.args ) {
                  check_gender(s.info,genders,MALE)(x)
               }
               check_gender(s.info,genders,MALE)(s.en)
               check_gender(s.info,genders,MALE)(s.clk)
            }
            case (s:BulkConnect) => {
               check_gender(s.info,genders,FEMALE)(s.loc)
               check_gender(s.info,genders,MALE)(s.exp)
            }
            case (s:Conditionally) => {
               check_gender(s.info,genders,MALE)(s.pred)
            }
            case (s:Empty) => false
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
            case (m:ExModule) => false
            case (m:InModule) => check_genders_s(genders)(m.body)
         }
      }
      if (errors.nonEmpty) throw new PassExceptions(errors)
      c
   }
}

object CheckWidths extends Pass with StanzaPass {
   def name = "Width Check"
   var mname = ""
   class UninferredWidth (info:Info) extends PassException(s"${info} : [module ${mname}]  Uninferred width.")
   class WidthTooSmall (info:Info,v:String) extends PassException(s"${info} : [module ${mname}  Width too small for constant ${v}.")
   class NegWidthException(info:Info) extends PassException(s"${info}: [module ${mname}] Width cannot be negative or zero.")
   def run (c:Circuit): Circuit = {
      val errors = ArrayBuffer[PassException]()
      def check_width_m (m:Module) : Unit = {
         def check_width_w (info:Info)(w:Width) : Width = {
            (w) match { 
               case (w:IntWidth)=> if (w.width <= 0) errors += new NegWidthException(info)
               case (w) => errors += new UninferredWidth(info)
            }
            w
         }
         def check_width_e (info:Info)(e:Expression) : Expression = {
            (eMap(check_width_e(info) _,e)) match { 
               case (e:UIntValue) => {
                  (e.width) match { 
                     case (w:IntWidth) => 
                        if (scala.math.max(1,e.value.bitLength) > w.width) {
                           errors += new WidthTooSmall(info,e.value.serialize)
                        }
                     case (w) => errors += new UninferredWidth(info)
                  }
                  check_width_w(info)(e.width)
               }
               case (e:SIntValue) => {
                  (e.width) match { 
                     case (w:IntWidth) => 
                        if (e.value.bitLength + 1 > w.width) errors += new WidthTooSmall(info,e.value.serialize)
                     case (w) => errors += new UninferredWidth(info)
                  }
                  check_width_w(info)(e.width)
               }
               case (e:DoPrim) => false
               case (e) => false
            }
            e
         }
         def check_width_s (s:Stmt) : Stmt = {
            eMap(check_width_e(get_info(s)) _,sMap(check_width_s _,s))
            def tm (t:Type) : Type = mapr(check_width_w(info(s)) _,t)
            tMap(tm _,s)
         }
      
         for (p <- m.ports) {
            mapr(check_width_w(p.info) _,p.tpe)
         }
   
         (m) match { 
            case (m:ExModule) => {}
            case (m:InModule) => check_width_s(m.body)
         }
      }
      
      for (m <- c.modules) {
         mname = m.name
         check_width_m(m)
      }
      if (errors.nonEmpty) throw new PassExceptions(errors)
      c
   }
}

object CheckInitialization extends Pass with StanzaPass {
   def name = "Check Initialization"
   var mname = ""
   class RefNotInitialized (info:Info, name:String) extends PassException(s"${info} : [module ${mname}  Reference ${name} is not fully initialized.")
   def run (c:Circuit): Circuit = {
      val errors = ArrayBuffer[PassException]()
      def check_init_m (m:InModule) : Unit = {
         def get_name (e:Expression) : String = {
            (e) match { 
               case (e:WRef) => e.name
               case (e:WSubField) => get_name(e.exp) + "." + e.name
               case (e:WSubIndex) => get_name(e.exp) + "[" + e.value + "]"
               case (e) => error("Shouldn't be here"); ""
            }
         }
         def has_voidQ (e:Expression) : Boolean = {
            var void = false
            def has_void (e:Expression) : Expression = {
               (e) match { 
                  case (e:WVoid) => void = true; e
                  case (e) => eMap(has_void,e)
               }
            }
            has_void(e)
            void
         }
         def check_init_s (s:Stmt) : Stmt = {
            (s) match { 
               case (s:Connect) => {
                  if (has_voidQ(s.exp)) errors += new RefNotInitialized(s.info,get_name(s.loc))
                  s
               }
               case (s) => sMap(check_init_s,s)
            }
         }
         check_init_s(m.body)
      }
         
      for (m <- c.modules) {
         mname = m.name
         (m) match  { 
            case (m:InModule) => check_init_m(m)
            case (m) => false
         }
      }

    if (errors.nonEmpty) throw new PassExceptions(errors)
    c
   }
}
