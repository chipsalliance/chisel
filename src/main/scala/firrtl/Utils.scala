// Utility functions for FIRRTL IR

/* TODO
 *  - Adopt style more similar to Chisel3 Emitter?
 *  - Find way to have generic map function instead of mapE and mapS under Stmt implicits
 */

/* TODO Richard
 *  - add new IR nodes to all Util functions
 */

package firrtl

import scala.collection.mutable.StringBuilder
import java.io.PrintWriter
import PrimOps._
import WrappedExpression._
import firrtl.WrappedType._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedHashMap
//import scala.reflect.runtime.universe._

object Utils {
//
//   // Is there a more elegant way to do this?
   private type FlagMap = Map[String, Boolean]
   private val FlagMap = Map[String, Boolean]().withDefaultValue(false)
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
   val sym_hash = LinkedHashMap[String,LinkedHashMap[String,Int]]()
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
   def firrtl_gensym (s:String):String = { firrtl_gensym(s,LinkedHashMap[String,Int]()) }
   def firrtl_gensym (sym_hash:LinkedHashMap[String,Int]):String = { firrtl_gensym("GEN",sym_hash) }
   def firrtl_gensym_module (s:String):String = {
      val sh = sym_hash.getOrElse(s,LinkedHashMap[String,Int]())
      val name = firrtl_gensym("GEN",sh)
      sym_hash(s) = sh
      name
   }
   def firrtl_gensym (s:String,sym_hash:LinkedHashMap[String,Int]):String = {
      if (sym_hash contains s) {
         val num = sym_hash(s) + 1
         sym_hash += (s -> num)
         (s + delin + num)
      } else {
         sym_hash += (s -> 0)
         (s + delin + 0)
      }
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
   def lowered_name (e:Expression) : String = {
      (e) match {
         case (e:WRef) => e.name
         case (e:WSubField) => lowered_name(e.exp) + "_" + e.name
         case (e:WSubIndex) => lowered_name(e.exp) + "_" + e.value
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
   def debug(node: AST)(implicit flags: FlagMap): String = {
     if (!flags.isEmpty) {
       var str = ""
       if (flags("types")) {
         val tpe = node.getType
         tpe match {
            case t:UnknownType => str += s"@<t:${tpe.wipeWidth.serialize}>"
         }
       }
       str
     }
     else {
       ""
     }
   }

   implicit class BigIntUtils(bi: BigInt){
     def serialize(implicit flags: FlagMap = FlagMap): String = 
        if (bi < BigInt(0)) "\"h" + bi.toString(16).substring(1) + "\""
        else "\"h" + bi.toString(16) + "\""
   }

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

   implicit class PrimOpUtils(op: PrimOp) {
     def serialize(implicit flags: FlagMap = FlagMap): String = op.getString
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
          val addr = Field("addr",DEFAULT,UIntType(IntWidth(ceil_log2(depth))))
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
       case _ => error("Shouldn't be here"); NoInfo
    }}


// =============== MAPPERS ===================
   def sMap(f:Stmt => Stmt, stmt: Stmt): Stmt =
      stmt match {
        case w: Conditionally => Conditionally(w.info, w.pred, f(w.conseq), f(w.alt))
        case b: Begin => {
           val stmtsx = ArrayBuffer[Stmt]()
           for (i <- 0 until b.stmts.size) {
              stmtsx += f(b.stmts(i))
           }
           Begin(stmtsx)
        }
        case s: Stmt => s
      }
   def eMap(f:Expression => Expression, stmt:Stmt) : Stmt =
      stmt match { 
        case r: DefRegister => DefRegister(r.info, r.name, r.tpe, f(r.clock), f(r.reset), f(r.init))
        case n: DefNode => DefNode(n.info, n.name, f(n.value))
        case c: Connect => Connect(c.info, f(c.loc), f(c.exp))
        case b: BulkConnect => BulkConnect(b.info, f(b.loc), f(b.exp))
        case w: Conditionally => Conditionally(w.info, f(w.pred), w.conseq, w.alt)
        case i: IsInvalid => IsInvalid(i.info, f(i.exp))
        case s: Stop => Stop(s.info, s.ret, f(s.clk), f(s.en))
        case p: Print => Print(p.info, p.string, p.args.map(f), f(p.clk), f(p.en))
        case c: CDefMPort => CDefMPort(c.info,c.name,c.tpe,c.mem,c.exps.map(f),c.direction)
        case s: Stmt => s 
      }
   def eMap(f: Expression => Expression, exp:Expression): Expression = 
      exp match {
        case s: SubField => SubField(f(s.exp), s.name, s.tpe)
        case s: SubIndex => SubIndex(f(s.exp), s.value, s.tpe)
        case s: SubAccess => SubAccess(f(s.exp), f(s.index), s.tpe)
        case m: Mux => Mux(f(m.cond), f(m.tval), f(m.fval), m.tpe)
        case v: ValidIf => ValidIf(f(v.cond), f(v.value), v.tpe)
        case p: DoPrim => DoPrim(p.op, p.args.map(f), p.consts, p.tpe)
        case s: WSubField => WSubField(f(s.exp), s.name, s.tpe, s.gender)
        case s: WSubIndex => WSubIndex(f(s.exp), s.value, s.tpe, s.gender)
        case s: WSubAccess => WSubAccess(f(s.exp), f(s.index), s.tpe, s.gender)
        case e: Expression => e
      }
   def tMap (f: Type => Type, t:Type):Type = {
      t match {
         case t:BundleType => BundleType(t.fields.map(p => Field(p.name, p.flip, f(p.tpe))))
         case t:VectorType => VectorType(f(t.tpe), t.size)
         case t => t
      }
   }
   def tMap (f: Type => Type, c:Expression) : Expression = {
      c match {
         case c:DoPrim => DoPrim(c.op,c.args,c.consts,f(c.tpe))
         case c:Mux => Mux(c.cond,c.tval,c.fval,f(c.tpe))
         case c:ValidIf => ValidIf(c.cond,c.value,f(c.tpe))
         case c:WRef => WRef(c.name,f(c.tpe),c.kind,c.gender)
         case c:WSubField => WSubField(c.exp,c.name,f(c.tpe),c.gender)
         case c:WSubIndex => WSubIndex(c.exp,c.value,f(c.tpe),c.gender)
         case c:WSubAccess => WSubAccess(c.exp,c.index,f(c.tpe),c.gender)
         case c => c
      }
   }
   def tMap (f: Type => Type, c:Stmt) : Stmt = {
      c match {
         case c:DefPoison => DefPoison(c.info,c.name,f(c.tpe))
         case c:DefWire => DefWire(c.info,c.name,f(c.tpe))
         case c:DefRegister => DefRegister(c.info,c.name,f(c.tpe),c.clock,c.reset,c.init)
         case c:DefMemory => DefMemory(c.info,c.name, f(c.data_type), c.depth, c.write_latency, c.read_latency, c.readers, c.writers, c.readwriters)
         case c:CDefMemory => CDefMemory(c.info,c.name, f(c.tpe), c.size, c.seq)
         case c:CDefMPort => CDefMPort(c.info,c.name, f(c.tpe), c.mem, c.exps,c.direction)
         case c => c
      }
   }
   def wMap (f: Width => Width, c:Expression) : Expression = {
      c match {
         case c:UIntValue => UIntValue(c.value,f(c.width))
         case c:SIntValue => SIntValue(c.value,f(c.width))
         case c => c
      }
   }
   def wMap (f: Width => Width, c:Type) : Type = {
      c match {
         case c:UIntType => UIntType(f(c.width))
         case c:SIntType => SIntType(f(c.width))
         case c => c
      }
   }
   def wMap (f: Width => Width, w:Width) : Width = {
      w match {
         case w:MaxWidth => MaxWidth(w.args.map(f))
         case w:MinWidth => MinWidth(w.args.map(f))
         case w:PlusWidth => PlusWidth(f(w.arg1),f(w.arg2))
         case w:MinusWidth => MinusWidth(f(w.arg1),f(w.arg2))
         case w:ExpWidth => ExpWidth(f(w.arg1))
         case w => w
      }
   }
   def stMap (f: String => String, c:Stmt) : Stmt = {
      c match {
         case (c:DefWire) => DefWire(c.info,f(c.name),c.tpe)
         case (c:DefPoison) => DefPoison(c.info,f(c.name),c.tpe)
         case (c:DefRegister) => DefRegister(c.info,f(c.name), c.tpe, c.clock, c.reset, c.init)
         case (c:DefMemory) => DefMemory(c.info,f(c.name), c.data_type, c.depth, c.write_latency, c.read_latency, c.readers, c.writers, c.readwriters)
         case (c:DefNode) => DefNode(c.info,f(c.name),c.value)
         case (c:DefInstance) => DefInstance(c.info,f(c.name), c.module)
         case (c:WDefInstance) => WDefInstance(c.info,f(c.name), c.module,c.tpe)
         case (c:CDefMemory) => CDefMemory(c.info,f(c.name),c.tpe,c.size,c.seq)
         case (c:CDefMPort) => CDefMPort(c.info,f(c.name),c.tpe,c.mem,c.exps,c.direction)
         case (c) => c
      }
   }
   def mapr (f: Width => Width, t:Type) : Type = {
      def apply_t (t:Type) : Type = wMap(f,tMap(apply_t _,t))
      apply_t(t)
   }
   def mapr (f: Width => Width, s:Stmt) : Stmt = {
      def apply_t (t:Type) : Type = mapr(f,t)
      def apply_e (e:Expression) : Expression =
         wMap(f,tMap(apply_t _,eMap(apply_e _,e)))
      def apply_s (s:Stmt) : Stmt =
         tMap(apply_t _,eMap(apply_e _,sMap(apply_s _,s)))
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
   //private trait StmtMagnet {
   //  def map(stmt: Stmt): Stmt
   //}
   //private object StmtMagnet {
   //  implicit def forStmt(f: Stmt => Stmt) = new StmtMagnet {
   //    override def map(stmt: Stmt): Stmt =
   //      stmt match {
   //        case w: Conditionally => Conditionally(w.info, w.pred, f(w.conseq), f(w.alt))
   //        case b: Begin => Begin(b.stmts.map(f))
   //        case s: Stmt => s
   //      }
   //  }
   //  implicit def forExp(f: Expression => Expression) = new StmtMagnet {
   //    override def map(stmt: Stmt): Stmt =
   //      stmt match { 
   //        case r: DefRegister => DefRegister(r.info, r.name, r.tpe, f(r.clock), f(r.reset), f(r.init))
   //        case n: DefNode => DefNode(n.info, n.name, f(n.value))
   //        case c: Connect => Connect(c.info, f(c.loc), f(c.exp))
   //        case b: BulkConnect => BulkConnect(b.info, f(b.loc), f(b.exp))
   //        case w: Conditionally => Conditionally(w.info, f(w.pred), w.conseq, w.alt)
   //        case i: IsInvalid => IsInvalid(i.info, f(i.exp))
   //        case s: Stop => Stop(s.info, s.ret, f(s.clk), f(s.en))
   //        case p: Print => Print(p.info, p.string, p.args.map(f), f(p.clk), f(p.en))
   //        case s: Stmt => s 
   //      }
   //  }
   //}
   implicit class ExpUtils(exp: Expression) {
     def serialize(implicit flags: FlagMap = FlagMap): String = {
       val ret = exp match {
         case v: UIntValue => s"UInt${v.width.serialize}(${v.value.serialize})"
         case v: SIntValue => s"SInt${v.width.serialize}(${v.value.serialize})"
         case r: Ref => r.name
         case s: SubField => s"${s.exp.serialize}.${s.name}"
         case s: SubIndex => s"${s.exp.serialize}[${s.value}]"
         case s: SubAccess => s"${s.exp.serialize}[${s.index.serialize}]"
         case m: Mux => s"mux(${m.cond.serialize}, ${m.tval.serialize}, ${m.fval.serialize})"
         case v: ValidIf => s"validif(${v.cond.serialize}, ${v.value.serialize})"
         case p: DoPrim => 
           s"${p.op.serialize}(" + (p.args.map(_.serialize) ++ p.consts.map(_.toString)).mkString(", ") + ")"
         case r: WRef => r.name
         case s: WSubField => s"${s.exp.serialize}.${s.name}"
         case s: WSubIndex => s"${s.exp.serialize}[${s.value}]"
         case s: WSubAccess => s"${s.exp.serialize}[${s.index.serialize}]"
         case r: WVoid => "VOID"
       } 
       ret + debug(exp)
     }
   }

   //  def map(f: Expression => Expression): Expression = 
   //    exp match {
   //      case s: SubField => SubField(f(s.exp), s.name, s.tpe)
   //      case s: SubIndex => SubIndex(f(s.exp), s.value, s.tpe)
   //      case s: SubAccess => SubAccess(f(s.exp), f(s.index), s.tpe)
   //      case m: Mux => Mux(f(m.cond), f(m.tval), f(m.fval), m.tpe)
   //      case v: ValidIf => ValidIf(f(v.cond), f(v.value), v.tpe)
   //      case p: DoPrim => DoPrim(p.op, p.args.map(f), p.consts, p.tpe)
   //      case s: WSubField => SubField(f(s.exp), s.name, s.tpe, s.gender)
   //      case s: WSubIndex => SubIndex(f(s.exp), s.value, s.tpe, s.gender)
   //      case s: WSubAccess => SubAccess(f(s.exp), f(s.index), s.tpe, s.gender)
   //      case e: Expression => e
   //    }
   //}

   implicit class StmtUtils(stmt: Stmt) {
     def serialize(implicit flags: FlagMap = FlagMap): String =
     {
       var ret = stmt match {
         case w: DefWire => s"wire ${w.name} : ${w.tpe.serialize}"
         case r: DefRegister => 
           val str = new StringBuilder(s"reg ${r.name} : ${r.tpe.serialize}, ${r.clock.serialize} with : ")
           withIndent {
             str ++= newline + s"reset => (${r.reset.serialize}, ${r.init.serialize})"
           }
           str
         case i: DefInstance => s"inst ${i.name} of ${i.module}"
         case i: WDefInstance => s"inst ${i.name} of ${i.module}"
         case m: DefMemory => {
           val str = new StringBuilder(s"mem ${m.name} : ")
           withIndent {
             str ++= newline + 
               s"data-type => ${m.data_type.serialize}" + newline +
               s"depth => ${m.depth}" + newline +
               s"read-latency => ${m.read_latency}" + newline +
               s"write-latency => ${m.write_latency}" + newline +
               (if (m.readers.nonEmpty) m.readers.map(r => s"reader => ${r}").mkString(newline) + newline
                else "") +
               (if (m.writers.nonEmpty) m.writers.map(w => s"writer => ${w}").mkString(newline) + newline
                else "") + 
               (if (m.readwriters.nonEmpty) m.readwriters.map(rw => s"readwriter => ${rw}").mkString(newline) + newline
                else "") +
               s"read-under-write => undefined"
           }
           str.result
         }
         case n: DefNode => s"node ${n.name} = ${n.value.serialize}"
         case c: Connect => s"${c.loc.serialize} <= ${c.exp.serialize}"
         case b: BulkConnect => s"${b.loc.serialize} <- ${b.exp.serialize}"
         case w: Conditionally => {
           var str = new StringBuilder(s"when ${w.pred.serialize} : ")
           withIndent { str ++= newline + w.conseq.serialize }
           w.alt match {
              case s:Empty => str.result
              case s => {
                str ++= newline + "else :"
                withIndent { str ++= newline + w.alt.serialize }
                str.result
                }
           }
         }
         case b: Begin => {
            val s = new StringBuilder
            for (i <- 0 until b.stmts.size) {
               if (i != 0) s ++= newline ++ b.stmts(i).serialize
               else s ++= b.stmts(i).serialize
            }
            s.result + debug(b)
         } 
         case i: IsInvalid => s"${i.exp.serialize} is invalid"
         case s: Stop => s"stop(${s.clk.serialize}, ${s.en.serialize}, ${s.ret})"
         case p: Print => {
            val q = '"'.toString
            s"printf(${p.clk.serialize}, ${p.en.serialize}, ${q}${p.string}${q}" + 
                          (if (p.args.nonEmpty) p.args.map(_.serialize).mkString(", ", ", ", "") else "") + ")"
         }
         case s:Empty => "skip"
         case s:CDefMemory => {
            if (s.seq) s"smem ${s.name} : ${s.tpe} [${s.size}]"
            else s"cmem ${s.name} : ${s.tpe} [${s.size}]"
         }
         case s:CDefMPort => {
            val dir = s.direction match {
               case MInfer => "infer"
               case MRead => "read"
               case MWrite => "write"
               case MReadWrite => "rdwr"
            }
            s"${dir} mport ${s.name} = ${s.mem}[${s.exps(0)}], s.exps(1)"
         }
       } 
       ret + debug(stmt)
     }

     // Using implicit types to allow overloading of function type to map, see StmtMagnet above
     //def map[T](f: T => T)(implicit magnet: (T => T) => StmtMagnet): Stmt = magnet(f).map(stmt)
     
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

   implicit class WidthUtils(w: Width) {
     def serialize(implicit flags: FlagMap = FlagMap): String = {
       val s = w match {
         case w:UnknownWidth => "" //"?"
         case w: IntWidth => s"<${w.width.toString}>"
         case w: VarWidth => s"<${w.name}>"
       } 
       s + debug(w)
     }
   }

   implicit class FlipUtils(f: Flip) {
     def serialize(implicit flags: FlagMap = FlagMap): String = {
       val s = f match {
         case REVERSE => "flip "
         case DEFAULT => ""
       } 
       s + debug(f)
     }
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
     def serialize(implicit flags: FlagMap = FlagMap): String = 
       s"${field.flip.serialize}${field.name} : ${field.tpe.serialize}" + debug(field)
     def flip(): Field = Field(field.name, field.flip.flip, field.tpe)

     def getType(): Type = field.tpe
     def toPort(): Port = Port(NoInfo, field.name, field.flip.toDirection, field.tpe)
   }

   implicit class TypeUtils(t: Type) {
     def serialize(implicit flags: FlagMap = FlagMap): String = {
       val commas = ", " // for mkString in BundleType
         val s = t match {
           case c:ClockType => "Clock"
           //case UnknownType => "UnknownType"
           case u:UnknownType => "?"
           case t: UIntType => s"UInt${t.width.serialize}"
           case t: SIntType => s"SInt${t.width.serialize}"
           case t: BundleType => s"{ ${t.fields.map(_.serialize).mkString(commas)}}"
           case t: VectorType => s"${t.tpe.serialize}[${t.size}]"
         } 
         s + debug(t)
     }

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
     def serialize(implicit flags: FlagMap = FlagMap): String = {
       val s = d match {
         case INPUT => "input"
         case OUTPUT => "output"
       } 
       s + debug(d)
     }
     def toFlip(): Flip = {
       d match {
         case INPUT => REVERSE
         case OUTPUT => DEFAULT
       }
     }
   }

   implicit class PortUtils(p: Port) {
     def serialize(implicit flags: FlagMap = FlagMap): String = 
       s"${p.direction.serialize} ${p.name} : ${p.tpe.serialize}" + debug(p)
     def getType(): Type = p.tpe
     def toField(): Field = Field(p.name, p.direction.toFlip, p.tpe)
   }

   implicit class ModuleUtils(m: Module) {
     def serialize(implicit flags: FlagMap = FlagMap): String = {
       m match {
          case m:InModule => {
             var s = new StringBuilder(s"module ${m.name} : ")
             withIndent {
               s ++= m.ports.map(newline ++ _.serialize).mkString
               s ++= newline ++ m.body.serialize
             }
             s ++= debug(m)
             s.toString
          }
       }
     }
   }

   implicit class CircuitUtils(c: Circuit) {
     def serialize(implicit flags: FlagMap = FlagMap): String = {
       var s = new StringBuilder(s"circuit ${c.main} : ")
       withIndent { s ++= newline ++ c.modules.map(_.serialize).mkString(newline + newline) }
       s ++= newline ++ newline
       s ++= debug(c)
       s.toString
     }
   }

   private var indentLevel = 0
   private def newline = "\n" + ("  " * indentLevel)
   private def indent(): Unit = indentLevel += 1
   private def unindent() { require(indentLevel > 0); indentLevel -= 1 }
   private def withIndent(f: => Unit) { indent(); f; unindent() }

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
