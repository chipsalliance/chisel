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
// Utility functions for FIRRTL IR

/* TODO
 *  - Adopt style more similar to Chisel3 Emitter?
 */

/* TODO Richard
 *  - add new IR nodes to all Util functions
 */

package firrtl

import scala.collection.mutable.StringBuilder
import java.io.PrintWriter
import com.typesafe.scalalogging.LazyLogging
import WrappedExpression._
import firrtl.WrappedType._
import firrtl.Mappers._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedHashMap
//import scala.reflect.runtime.universe._

object Utils extends LazyLogging {
  private[firrtl] def time[R](name: String)(block: => R): R = {
    logger.info(s"Starting $name")
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    logger.info(s"Finished $name")
    val timeMillis = (t1 - t0) / 1000000.0
    logger.info(f"$name took $timeMillis%.1f ms\n")
    result
  }

   implicit class WithAs[T](x: T) {
     import scala.reflect._
     def as[O: ClassTag]: Option[O] = x match {
       case o: O => Some(o)
       case _ => None }
     def typeof[O: ClassTag]: Boolean = x match {
       case o: O => true
       case _ => false }
   }
   implicit def toWrappedExpression (x:Expression) = new WrappedExpression(x)
   def ceil_log2(x: BigInt): BigInt = (x-1).bitLength
   def ceil_log2(x: Int): Int = scala.math.ceil(scala.math.log(x) / scala.math.log(2)).toInt
   val gen_names = Map[String,Int]()
   val delin = "_"
   def BoolType () = { UIntType(IntWidth(1)) } 
   val one  = UIntValue(BigInt(1),IntWidth(1))
   val zero = UIntValue(BigInt(0),IntWidth(1))
   def uint (i:Int) : UIntValue = {
      val num_bits = req_num_bits(i)
      val w = IntWidth(scala.math.max(1,num_bits - 1))
      UIntValue(BigInt(i),w)
   }
   def req_num_bits (i: Int) : Int = {
      val ix = if (i < 0) ((-1 * i) - 1) else i
      ceil_log2(ix + 1) + 1
   }
   def AND (e1:WrappedExpression,e2:WrappedExpression) : Expression = {
      if (e1 == e2) e1.e1
      else if ((e1 == we(zero)) | (e2 == we(zero))) zero
      else if (e1 == we(one)) e2.e1
      else if (e2 == we(one)) e1.e1
      else DoPrim(AND_OP,Seq(e1.e1,e2.e1),Seq(),UIntType(IntWidth(1)))
   }
   
   def OR (e1:WrappedExpression,e2:WrappedExpression) : Expression = {
      if (e1 == e2) e1.e1
      else if ((e1 == we(one)) | (e2 == we(one))) one
      else if (e1 == we(zero)) e2.e1
      else if (e2 == we(zero)) e1.e1
      else DoPrim(OR_OP,Seq(e1.e1,e2.e1),Seq(),UIntType(IntWidth(1)))
   }
   def EQV (e1:Expression,e2:Expression) : Expression = { DoPrim(EQUAL_OP,Seq(e1,e2),Seq(),tpe(e1)) }
   def NOT (e1:WrappedExpression) : Expression = {
      if (e1 == we(one)) zero
      else if (e1 == we(zero)) one
      else DoPrim(EQUAL_OP,Seq(e1.e1,zero),Seq(),UIntType(IntWidth(1)))
   }

   
   //def MUX (p:Expression,e1:Expression,e2:Expression) : Expression = {
   //   Mux(p,e1,e2,mux_type(tpe(e1),tpe(e2)))
   //}

   def create_mask (dt:Type) : Type = {
      dt match {
         case t:VectorType => VectorType(create_mask(t.tpe),t.size)
         case t:BundleType => {
            val fieldss = t.fields.map { f => Field(f.name,f.flip,create_mask(f.tpe)) }
            BundleType(fieldss)
         }
         case t:UIntType => BoolType()
         case t:SIntType => BoolType()
      }
   }
   def create_exps (n:String, t:Type) : Seq[Expression] =
      create_exps(WRef(n,t,ExpKind(),UNKNOWNGENDER))
   def create_exps (e:Expression) : Seq[Expression] = {
      e match {
         case (e:Mux) => {
            val e1s = create_exps(e.tval)
            val e2s = create_exps(e.fval)
            (e1s, e2s).zipped.map { (e1,e2) => Mux(e.cond,e1,e2,mux_type_and_widths(e1,e2)) }
         }
         case (e:ValidIf) => create_exps(e.value).map { e1 => ValidIf(e.cond,e1,tpe(e1)) }
         case (e) => {
            tpe(e) match {
               case (t:UIntType) => Seq(e)
               case (t:SIntType) => Seq(e)
               case (t:ClockType) => Seq(e)
               case (t:BundleType) => {
                  t.fields.flatMap { f => create_exps(WSubField(e,f.name,f.tpe,times(gender(e), f.flip))) }
               }
               case (t:VectorType) => {
                  (0 until t.size).flatMap { i => create_exps(WSubIndex(e,i,t.tpe,gender(e))) }
               }
            }
         }
      }
   }
   def get_flip (t:Type, i:Int, f:Flip) : Flip = { 
      if (i >= get_size(t)) error("Shouldn't be here")
      val x = t match {
         case (t:UIntType) => f
         case (t:SIntType) => f
         case (t:ClockType) => f
         case (t:BundleType) => {
            var n = i
            var ret:Option[Flip] = None
            t.fields.foreach { x => {
               if (n < get_size(x.tpe)) {
                  ret match {
                     case None => ret = Some(get_flip(x.tpe,n,times(x.flip,f)))
                     case ret => {}
                  }
               } else { n = n - get_size(x.tpe) }
            }}
            ret.asInstanceOf[Some[Flip]].get
         }
         case (t:VectorType) => {
            var n = i
            var ret:Option[Flip] = None
            for (j <- 0 until t.size) {
               if (n < get_size(t.tpe)) {
                  ret = Some(get_flip(t.tpe,n,f))
               } else {
                  n = n - get_size(t.tpe)
               }
            }
            ret.asInstanceOf[Some[Flip]].get
         }
      }
      x
   }
   
   def get_point (e:Expression) : Int = { 
      e match {
         case (e:WRef) => 0
         case (e:WSubField) => {
            var i = 0
            tpe(e.exp).asInstanceOf[BundleType].fields.find { f => {
               val b = f.name == e.name
               if (!b) { i = i + get_size(f.tpe)}
               b
            }}
            i
         }
         case (e:WSubIndex) => e.value * get_size(e.tpe)
         case (e:WSubAccess) => get_point(e.exp)
      }
   }

//============== TYPES ================
   def mux_type (e1:Expression,e2:Expression) : Type = mux_type(tpe(e1),tpe(e2))
   def mux_type (t1:Type,t2:Type) : Type = {
      if (wt(t1) == wt(t2)) {
         (t1,t2) match { 
            case (t1:UIntType,t2:UIntType) => UIntType(UnknownWidth())
            case (t1:SIntType,t2:SIntType) => SIntType(UnknownWidth())
            case (t1:VectorType,t2:VectorType) => VectorType(mux_type(t1.tpe,t2.tpe),t1.size)
            case (t1:BundleType,t2:BundleType) => 
               BundleType((t1.fields,t2.fields).zipped.map((f1,f2) => {
                  Field(f1.name,f1.flip,mux_type(f1.tpe,f2.tpe))
               }))
         }
      } else UnknownType()
   }
   def mux_type_and_widths (e1:Expression,e2:Expression) : Type = mux_type_and_widths(tpe(e1),tpe(e2))
   def mux_type_and_widths (t1:Type,t2:Type) : Type = {
      def wmax (w1:Width,w2:Width) : Width = {
         (w1,w2) match {
            case (w1:IntWidth,w2:IntWidth) => IntWidth(w1.width.max(w2.width))
            case (w1,w2) => MaxWidth(Seq(w1,w2))
         }
      }
      val wt1 = new WrappedType(t1)
      val wt2 = new WrappedType(t2)
      if (wt1 == wt2) {
         (t1,t2) match {
            case (t1:UIntType,t2:UIntType) => UIntType(wmax(t1.width,t2.width))
            case (t1:SIntType,t2:SIntType) => SIntType(wmax(t1.width,t2.width))
            case (t1:VectorType,t2:VectorType) => VectorType(mux_type_and_widths(t1.tpe,t2.tpe),t1.size)
            case (t1:BundleType,t2:BundleType) => BundleType((t1.fields zip t2.fields).map{case (f1, f2) => Field(f1.name,f1.flip,mux_type_and_widths(f1.tpe,f2.tpe))})
         }
      } else UnknownType()
   }
   def module_type (m:Module) : Type = {
      BundleType(m.ports.map(p => p.toField))
   }
   def sub_type (v:Type) : Type = {
      v match {
         case v:VectorType => v.tpe
         case v => UnknownType()
      }
   }
   def field_type (v:Type,s:String) : Type = {
      v match {
         case v:BundleType => {
            val ft = v.fields.find(p => p.name == s)
            ft match {
               case ft:Some[Field] => ft.get.tpe
               case ft => UnknownType()
            }
         }
         case v => UnknownType()
      }
   }
   
////=====================================
   def widthBANG (t:Type) : Width = {
      t match {
         case t:UIntType => t.width
         case t:SIntType => t.width
         case t:ClockType => IntWidth(1)
         case t => error("No width!")
      }
   }
   def long_BANG (t:Type) : Long = {
      (t) match {
         case (t:UIntType) => t.width.as[IntWidth].get.width.toLong
         case (t:SIntType) => t.width.as[IntWidth].get.width.toLong
         case (t:BundleType) => {
            var w = 0
            for (f <- t.fields) { w = w + long_BANG(f.tpe).toInt }
            w
         }
         case (t:VectorType) => t.size * long_BANG(t.tpe)
         case (t:ClockType) => 1
      }
   }
// =================================
   def error(str:String) = throw new FIRRTLException(str)

   implicit class ASTUtils(ast: AST) {
     def getType(): Type = 
       ast match {
         case e: Expression => e.getType
         case s: Stmt => s.getType
         //case f: Field => f.getType
         case t: Type => t.getType
         case p: Port => p.getType
         case _ => UnknownType()
       }
   }

//// =============== EXPANSION FUNCTIONS ================
   def get_size (t:Type) : Int = {
      t match {
         case (t:BundleType) => {
            var sum = 0
            for (f <- t.fields) {
               sum = sum + get_size(f.tpe)
            }
            sum
         }
         case (t:VectorType) => t.size * get_size(t.tpe)
         case (t) => 1
      }
   }
   def get_valid_points (t1:Type,t2:Type,flip1:Flip,flip2:Flip) : Seq[(Int,Int)] = {
      //;println_all(["Inside with t1:" t1 ",t2:" t2 ",f1:" flip1 ",f2:" flip2])
      (t1,t2) match {
         case (t1:UIntType,t2:UIntType) => if (flip1 == flip2) Seq((0, 0)) else Seq()
         case (t1:SIntType,t2:SIntType) => if (flip1 == flip2) Seq((0, 0)) else Seq()
         case (t1:BundleType,t2:BundleType) => {
            val points = ArrayBuffer[(Int,Int)]()
            var ilen = 0
            var jlen = 0
            for (i <- 0 until t1.fields.size) {
               for (j <- 0 until t2.fields.size) {
                  val f1 = t1.fields(i)
                  val f2 = t2.fields(j)
                  if (f1.name == f2.name) {
                     val ls = get_valid_points(f1.tpe,f2.tpe,times(flip1, f1.flip),times(flip2, f2.flip))
                     for (x <- ls) {
                        points += ((x._1 + ilen, x._2 + jlen))
                     }
                  }
                  jlen = jlen + get_size(t2.fields(j).tpe)
               }
               ilen = ilen + get_size(t1.fields(i).tpe)
               jlen = 0
            }
            points
         }
         case (t1:VectorType,t2:VectorType) => {
            val points = ArrayBuffer[(Int,Int)]()
            var ilen = 0
            var jlen = 0
            for (i <- 0 until scala.math.min(t1.size,t2.size)) {
               val ls = get_valid_points(t1.tpe,t2.tpe,flip1,flip2)
               for (x <- ls) {
                  val y = ((x._1 + ilen), (x._2 + jlen))
                  points += y
               }
               ilen = ilen + get_size(t1.tpe)
               jlen = jlen + get_size(t2.tpe)
            }
            points
         }
      }
   }
// =========== GENDER/FLIP UTILS ============
   def swap (g:Gender) : Gender = {
      g match {
         case UNKNOWNGENDER => UNKNOWNGENDER
         case MALE => FEMALE
         case FEMALE => MALE
         case BIGENDER => BIGENDER
      }
   }
   def swap (d:Direction) : Direction = {
      d match {
         case OUTPUT => INPUT
         case INPUT => OUTPUT
      }
   }
   def swap (f:Flip) : Flip = {
      f match {
         case DEFAULT => REVERSE
         case REVERSE => DEFAULT
      }
   }
   def to_dir (g:Gender) : Direction = {
      g match {
         case MALE => INPUT
         case FEMALE => OUTPUT
      }
   }
   def to_gender (d:Direction) : Gender = {
      d match {
         case INPUT => MALE
         case OUTPUT => FEMALE
      }
   }
  def toGender(f: Flip): Gender = f match {
    case DEFAULT => FEMALE
    case REVERSE => MALE
  }
  def toFlip(g: Gender): Flip = g match {
    case MALE => REVERSE
    case FEMALE => DEFAULT
  }

   def field_flip (v:Type,s:String) : Flip = {
      v match {
         case v:BundleType => {
            val ft = v.fields.find {p => p.name == s}
            ft match {
               case ft:Some[Field] => ft.get.flip
               case ft => DEFAULT
            }
         }
         case v => DEFAULT
      }
   }
   def get_field (v:Type,s:String) : Field = {
      v match {
         case v:BundleType => {
            val ft = v.fields.find {p => p.name == s}
            ft match {
               case ft:Some[Field] => ft.get
               case ft => error("Shouldn't be here"); Field("blah",DEFAULT,UnknownType())
            }
         }
         case v => error("Shouldn't be here"); Field("blah",DEFAULT,UnknownType())
      }
   }
   def times (flip:Flip,d:Direction) : Direction = times(flip, d)
   def times (d:Direction,flip:Flip) : Direction = {
      flip match {
         case DEFAULT => d
         case REVERSE => swap(d)
      }
   }
   def times (g: Gender, d: Direction): Direction = times(d, g)
   def times (d: Direction, g: Gender): Direction = g match {
     case FEMALE => d
     case MALE => swap(d) // MALE == INPUT == REVERSE
   }

   def times (g:Gender,flip:Flip) : Gender = times(flip, g)
   def times (flip:Flip,g:Gender) : Gender = {
      flip match {
         case DEFAULT => g
         case REVERSE => swap(g)
      }
   }
   def times (f1:Flip,f2:Flip) : Flip = {
      f2 match {
         case DEFAULT => f1
         case REVERSE => swap(f1)
      }
   }


// =========== ACCESSORS =========
   def info (s:Stmt) : Info = {
      s match {
         case s:DefWire => s.info
         case s:DefPoison => s.info
         case s:DefRegister => s.info
         case s:DefInstance => s.info
         case s:WDefInstance => s.info
         case s:DefMemory => s.info
         case s:DefNode => s.info
         case s:Conditionally => s.info
         case s:BulkConnect => s.info
         case s:Connect => s.info
         case s:IsInvalid => s.info
         case s:Stop => s.info
         case s:Print => s.info
         case s:Begin => NoInfo
         case s:Empty => NoInfo
      }
   }
   def gender (e:Expression) : Gender = {
     e match {
        case e:WRef => e.gender
        case e:WSubField => e.gender
        case e:WSubIndex => e.gender
        case e:WSubAccess => e.gender
        case e:DoPrim => MALE
        case e:UIntValue => MALE
        case e:SIntValue => MALE
        case e:Mux => MALE
        case e:ValidIf => MALE
        case e:WInvalid => MALE
        case e => println(e); error("Shouldn't be here")
    }}
   def get_gender (s:Stmt) : Gender =
     s match {
       case s:DefWire => BIGENDER
       case s:DefRegister => BIGENDER
       case s:WDefInstance => MALE
       case s:DefNode => MALE
       case s:DefInstance => MALE
       case s:DefPoison => UNKNOWNGENDER
       case s:DefMemory => MALE
       case s:Begin => UNKNOWNGENDER
       case s:Connect => UNKNOWNGENDER
       case s:BulkConnect => UNKNOWNGENDER
       case s:Stop => UNKNOWNGENDER
       case s:Print => UNKNOWNGENDER
       case s:Empty => UNKNOWNGENDER
       case s:IsInvalid => UNKNOWNGENDER
     }
   def get_gender (p:Port) : Gender =
     if (p.direction == INPUT) MALE else FEMALE
   def kind (e:Expression) : Kind =
      e match {
         case e:WRef => e.kind
         case e:WSubField => kind(e.exp)
         case e:WSubIndex => kind(e.exp)
         case e => ExpKind()
      }
   def tpe (e:Expression) : Type =
      e match {
         case e:Ref => e.tpe
         case e:SubField => e.tpe
         case e:SubIndex => e.tpe
         case e:SubAccess => e.tpe
         case e:WRef => e.tpe
         case e:WSubField => e.tpe
         case e:WSubIndex => e.tpe
         case e:WSubAccess => e.tpe
         case e:DoPrim => e.tpe
         case e:Mux => e.tpe
         case e:ValidIf => e.tpe
         case e:UIntValue => UIntType(e.width)
         case e:SIntValue => SIntType(e.width)
         case e:WVoid => UnknownType()
         case e:WInvalid => UnknownType()
      }
   def get_type (s:Stmt) : Type = {
      s match {
       case s:DefWire => s.tpe
       case s:DefPoison => s.tpe
       case s:DefRegister => s.tpe
       case s:DefNode => tpe(s.value)
       case s:DefMemory => {
          val depth = s.depth
          val addr = Field("addr",DEFAULT,UIntType(IntWidth(scala.math.max(ceil_log2(depth), 1))))
          val en = Field("en",DEFAULT,BoolType())
          val clk = Field("clk",DEFAULT,ClockType())
          val def_data = Field("data",DEFAULT,s.data_type)
          val rev_data = Field("data",REVERSE,s.data_type)
          val mask = Field("mask",DEFAULT,create_mask(s.data_type))
          val wmode = Field("wmode",DEFAULT,UIntType(IntWidth(1)))
          val rdata = Field("rdata",REVERSE,s.data_type)
          val read_type = BundleType(Seq(rev_data,addr,en,clk))
          val write_type = BundleType(Seq(def_data,mask,addr,en,clk))
          val readwrite_type = BundleType(Seq(wmode,rdata,def_data,mask,addr,en,clk))

          val mem_fields = ArrayBuffer[Field]()
          s.readers.foreach {x => mem_fields += Field(x,REVERSE,read_type)}
          s.writers.foreach {x => mem_fields += Field(x,REVERSE,write_type)}
          s.readwriters.foreach {x => mem_fields += Field(x,REVERSE,readwrite_type)}
          BundleType(mem_fields)
       }
       case s:DefInstance => UnknownType()
       case s:WDefInstance => s.tpe
       case _ => UnknownType()
    }}
   def get_name (s:Stmt) : String = {
      s match {
       case s:DefWire => s.name
       case s:DefPoison => s.name
       case s:DefRegister => s.name
       case s:DefNode => s.name
       case s:DefMemory => s.name
       case s:DefInstance => s.name
       case s:WDefInstance => s.name
       case _ => error("Shouldn't be here"); "blah"
    }}
   def get_info (s:Stmt) : Info = {
      s match {
       case s:DefWire => s.info
       case s:DefPoison => s.info
       case s:DefRegister => s.info
       case s:DefInstance => s.info
       case s:WDefInstance => s.info
       case s:DefMemory => s.info
       case s:DefNode => s.info
       case s:Conditionally => s.info
       case s:BulkConnect => s.info
       case s:Connect => s.info
       case s:IsInvalid => s.info
       case s:Stop => s.info
       case s:Print => s.info
       case _ => NoInfo
    }}

  /** Splits an Expression into root Ref and tail
    *
    * @example
    *   Given:   SubField(SubIndex(SubField(Ref("a", UIntType(IntWidth(32))), "b"), 2), "c")
    *   Returns: (Ref("a"), SubField(SubIndex(Ref("b"), 2), "c"))
    *   a.b[2].c -> (a, b[2].c)
    * @example
    *   Given:   SubField(SubIndex(Ref("b"), 2), "c")
    *   Returns: (Ref("b"), SubField(SubIndex(EmptyExpression, 2), "c"))
    *   b[2].c -> (b, EMPTY[2].c)
    * @note This function only supports WRef, WSubField, and WSubIndex
    */
  def splitRef(e: Expression): (WRef, Expression) = e match {
    case e: WRef => (e, EmptyExpression)
    case e: WSubIndex =>
      val (root, tail) = splitRef(e.exp)
      (root, WSubIndex(tail, e.value, e.tpe, e.gender))
    case e: WSubField =>
      val (root, tail) = splitRef(e.exp)
      tail match {
        case EmptyExpression => (root, WRef(e.name, e.tpe, root.kind, e.gender))
        case exp => (root, WSubField(tail, e.name, e.tpe, e.gender))
      }
  }

  /** Adds a root reference to some SubField/SubIndex chain */
  def mergeRef(root: WRef, body: Expression): Expression = body match {
    case e: WRef =>
      WSubField(root, e.name, e.tpe, e.gender)
    case e: WSubIndex =>
      WSubIndex(mergeRef(root, e.exp), e.value, e.tpe, e.gender)
    case e: WSubField =>
      WSubField(mergeRef(root, e.exp), e.name, e.tpe, e.gender)
    case EmptyExpression => root
  }

// =============== RECURISVE MAPPERS ===================
   def mapr (f: Width => Width, t:Type) : Type = {
      def apply_t (t:Type) : Type = t map (apply_t) map (f)
      apply_t(t)
   }
   def mapr (f: Width => Width, s:Stmt) : Stmt = {
      def apply_t (t:Type) : Type = mapr(f,t)
      def apply_e (e:Expression) : Expression = e map (apply_e) map (apply_t) map (f)
      def apply_s (s:Stmt) : Stmt = s map (apply_s) map (apply_e) map (apply_t)
      apply_s(s)
   }
   val ONE = IntWidth(1)
   //def digits (s:String) : Boolean {
   //   val digits = "0123456789"
   //   var yes:Boolean = true
   //   for (c <- s) {
   //      if !digits.contains(c) : yes = false
   //   }
   //   yes
   //}
   //def generated (s:String) : Option[Int] = {
   //   (1 until s.length() - 1).find{
   //      i => {
   //         val sub = s.substring(i + 1)
   //         s.substring(i,i).equals("_") & digits(sub) & !s.substring(i - 1,i-1).equals("_")
   //      }
   //   }
   //}
   //def get-sym-hash (m:InModule) : LinkedHashMap[String,Int] = { get-sym-hash(m,Seq()) }
   //def get-sym-hash (m:InModule,keywords:Seq[String]) : LinkedHashMap[String,Int] = {
   //   val sym-hash = LinkedHashMap[String,Int]()
   //   for (k <- keywords) { sym-hash += (k -> 0) }
   //   def add-name (s:String) : String = {
   //      val sx = to-string(s)
   //      val ix = generated(sx)
   //      ix match {
   //         case (i:False) => {
   //            if (sym_hash.contains(s)) {
   //               val num = sym-hash(s)
   //               sym-hash += (s -> max(num,0))
   //            } else {
   //               sym-hash += (s -> 0)
   //            }
   //         }
   //         case (i:Int) => {
   //            val name = sx.substring(0,i)
   //            val digit = to-int(substring(sx,i + 1))
   //            if key?(sym-hash,name) :
   //               val num = sym-hash[name]
   //               sym-hash[name] = max(num,digit)
   //            else :
   //               sym-hash[name] = digit
   //         }
   //      s
   //         
   //   defn to-port (p:Port) : add-name(name(p))
   //   defn to-stmt (s:Stmt) -> Stmt :
   //     map{to-stmt,_} $ map(add-name,s)
   //
   //   to-stmt(body(m))
   //   map(to-port,ports(m))
   //   sym-hash
   implicit class StmtUtils(stmt: Stmt) {

     def getType(): Type =
       stmt match {
         case s: DefWire    => s.tpe
         case s: DefRegister => s.tpe
         case s: DefMemory  => s.data_type
         case _ => UnknownType()
       }

     def getInfo: Info =
       stmt match {
         case s: DefWire => s.info
         case s: DefPoison => s.info
         case s: DefRegister => s.info
         case s: DefInstance => s.info
         case s: DefMemory => s.info
         case s: DefNode => s.info
         case s: Conditionally => s.info
         case s: BulkConnect => s.info
         case s: Connect => s.info
         case s: IsInvalid => s.info
         case s: Stop => s.info
         case s: Print => s.info
         case _ => NoInfo
       }
   }

   implicit class FlipUtils(f: Flip) {
     def flip(): Flip = {
       f match {
         case REVERSE => DEFAULT
         case DEFAULT => REVERSE
       }
     }
         
     def toDirection(): Direction = {
       f match {
         case DEFAULT => OUTPUT
         case REVERSE => INPUT
       }
     }
   }

   implicit class FieldUtils(field: Field) {
     def flip(): Field = Field(field.name, field.flip.flip, field.tpe)

     def getType(): Type = field.tpe
     def toPort(info: Info = NoInfo): Port =
       Port(info, field.name, field.flip.toDirection, field.tpe)
   }

   implicit class TypeUtils(t: Type) {
     def isGround: Boolean = t match {
       case (_: UIntType | _: SIntType | _: ClockType) => true
       case (_: BundleType | _: VectorType) => false
     }
     def isAggregate: Boolean = !t.isGround

     def getType(): Type = 
       t match {
         case v: VectorType => v.tpe
         case tpe: Type => UnknownType()
       }

     def wipeWidth(): Type = 
       t match {
         case t: UIntType => UIntType(UnknownWidth())
         case t: SIntType => SIntType(UnknownWidth())
         case _ => t
       }
   }

   implicit class DirectionUtils(d: Direction) {
     def toFlip(): Flip = {
       d match {
         case INPUT => REVERSE
         case OUTPUT => DEFAULT
       }
     }
   }
  
  implicit class PortUtils(p: Port) {
    def getType(): Type = p.tpe
    def toField(): Field = Field(p.name, p.direction.toFlip, p.tpe)
  }


   val v_keywords = Map[String,Boolean]() +
      ("alias" -> true) +
      ("always" -> true) +
      ("always_comb" -> true) +
      ("always_ff" -> true) +
      ("always_latch" -> true) +
      ("and" -> true) +
      ("assert" -> true) +
      ("assign" -> true) +
      ("assume" -> true) +
      ("attribute" -> true) +
      ("automatic" -> true) +
      ("before" -> true) +
      ("begin" -> true) +
      ("bind" -> true) +
      ("bins" -> true) +
      ("binsof" -> true) +
      ("bit" -> true) +
      ("break" -> true) +
      ("buf" -> true) +
      ("bufif0" -> true) +
      ("bufif1" -> true) +
      ("byte" -> true) +
      ("case" -> true) +
      ("casex" -> true) +
      ("casez" -> true) +
      ("cell" -> true) +
      ("chandle" -> true) +
      ("class" -> true) +
      ("clocking" -> true) +
      ("cmos" -> true) +
      ("config" -> true) +
      ("const" -> true) +
      ("constraint" -> true) +
      ("context" -> true) +
      ("continue" -> true) +
      ("cover" -> true) +
      ("covergroup" -> true) +
      ("coverpoint" -> true) +
      ("cross" -> true) +
      ("deassign" -> true) +
      ("default" -> true) +
      ("defparam" -> true) +
      ("design" -> true) +
      ("disable" -> true) +
      ("dist" -> true) +
      ("do" -> true) +
      ("edge" -> true) +
      ("else" -> true) +
      ("end" -> true) +
      ("endattribute" -> true) +
      ("endcase" -> true) +
      ("endclass" -> true) +
      ("endclocking" -> true) +
      ("endconfig" -> true) +
      ("endfunction" -> true) +
      ("endgenerate" -> true) +
      ("endgroup" -> true) +
      ("endinterface" -> true) +
      ("endmodule" -> true) +
      ("endpackage" -> true) +
      ("endprimitive" -> true) +
      ("endprogram" -> true) +
      ("endproperty" -> true) +
      ("endspecify" -> true) +
      ("endsequence" -> true) +
      ("endtable" -> true) +
      ("endtask" -> true) +
      ("enum" -> true) +
      ("event" -> true) +
      ("expect" -> true) +
      ("export" -> true) +
      ("extends" -> true) +
      ("extern" -> true) +
      ("final" -> true) +
      ("first_match" -> true) +
      ("for" -> true) +
      ("force" -> true) +
      ("foreach" -> true) +
      ("forever" -> true) +
      ("fork" -> true) +
      ("forkjoin" -> true) +
      ("function" -> true) +
      ("generate" -> true) +
      ("genvar" -> true) +
      ("highz0" -> true) +
      ("highz1" -> true) +
      ("if" -> true) +
      ("iff" -> true) +
      ("ifnone" -> true) +
      ("ignore_bins" -> true) +
      ("illegal_bins" -> true) +
      ("import" -> true) +
      ("incdir" -> true) +
      ("include" -> true) +
      ("initial" -> true) +
      ("initvar" -> true) +
      ("inout" -> true) +
      ("input" -> true) +
      ("inside" -> true) +
      ("instance" -> true) +
      ("int" -> true) +
      ("integer" -> true) +
      ("interconnect" -> true) +
      ("interface" -> true) +
      ("intersect" -> true) +
      ("join" -> true) +
      ("join_any" -> true) +
      ("join_none" -> true) +
      ("large" -> true) +
      ("liblist" -> true) +
      ("library" -> true) +
      ("local" -> true) +
      ("localparam" -> true) +
      ("logic" -> true) +
      ("longint" -> true) +
      ("macromodule" -> true) +
      ("matches" -> true) +
      ("medium" -> true) +
      ("modport" -> true) +
      ("module" -> true) +
      ("nand" -> true) +
      ("negedge" -> true) +
      ("new" -> true) +
      ("nmos" -> true) +
      ("nor" -> true) +
      ("noshowcancelled" -> true) +
      ("not" -> true) +
      ("notif0" -> true) +
      ("notif1" -> true) +
      ("null" -> true) +
      ("or" -> true) +
      ("output" -> true) +
      ("package" -> true) +
      ("packed" -> true) +
      ("parameter" -> true) +
      ("pmos" -> true) +
      ("posedge" -> true) +
      ("primitive" -> true) +
      ("priority" -> true) +
      ("program" -> true) +
      ("property" -> true) +
      ("protected" -> true) +
      ("pull0" -> true) +
      ("pull1" -> true) +
      ("pulldown" -> true) +
      ("pullup" -> true) +
      ("pulsestyle_onevent" -> true) +
      ("pulsestyle_ondetect" -> true) +
      ("pure" -> true) +
      ("rand" -> true) +
      ("randc" -> true) +
      ("randcase" -> true) +
      ("randsequence" -> true) +
      ("rcmos" -> true) +
      ("real" -> true) +
      ("realtime" -> true) +
      ("ref" -> true) +
      ("reg" -> true) +
      ("release" -> true) +
      ("repeat" -> true) +
      ("return" -> true) +
      ("rnmos" -> true) +
      ("rpmos" -> true) +
      ("rtran" -> true) +
      ("rtranif0" -> true) +
      ("rtranif1" -> true) +
      ("scalared" -> true) +
      ("sequence" -> true) +
      ("shortint" -> true) +
      ("shortreal" -> true) +
      ("showcancelled" -> true) +
      ("signed" -> true) +
      ("small" -> true) +
      ("solve" -> true) +
      ("specify" -> true) +
      ("specparam" -> true) +
      ("static" -> true) +
      ("strength" -> true) +
      ("string" -> true) +
      ("strong0" -> true) +
      ("strong1" -> true) +
      ("struct" -> true) +
      ("super" -> true) +
      ("supply0" -> true) +
      ("supply1" -> true) +
      ("table" -> true) +
      ("tagged" -> true) +
      ("task" -> true) +
      ("this" -> true) +
      ("throughout" -> true) +
      ("time" -> true) +
      ("timeprecision" -> true) +
      ("timeunit" -> true) +
      ("tran" -> true) +
      ("tranif0" -> true) +
      ("tranif1" -> true) +
      ("tri" -> true) +
      ("tri0" -> true) +
      ("tri1" -> true) +
      ("triand" -> true) +
      ("trior" -> true) +
      ("trireg" -> true) +
      ("type" -> true) +
      ("typedef" -> true) +
      ("union" -> true) +
      ("unique" -> true) +
      ("unsigned" -> true) +
      ("use" -> true) +
      ("var" -> true) +
      ("vectored" -> true) +
      ("virtual" -> true) +
      ("void" -> true) +
      ("wait" -> true) +
      ("wait_order" -> true) +
      ("wand" -> true) +
      ("weak0" -> true) +
      ("weak1" -> true) +
      ("while" -> true) +
      ("wildcard" -> true) +
      ("wire" -> true) +
      ("with" -> true) +
      ("within" -> true) +
      ("wor" -> true) +
      ("xnor" -> true) +
      ("xor" -> true) +
      ("SYNTHESIS" -> true) +
      ("PRINTF_COND" -> true) +
      ("VCS" -> true)
}

object MemoizedHash {
  implicit def convertTo[T](e: T): MemoizedHash[T] = new MemoizedHash(e)
  implicit def convertFrom[T](f: MemoizedHash[T]): T = f.t
}

class MemoizedHash[T](val t: T) {
  override lazy val hashCode = t.hashCode
  override def equals(that: Any) = that match {
    case x: MemoizedHash[_] => t equals x.t
    case _ => false
  }
}
