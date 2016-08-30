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

import firrtl.ir._
import firrtl.PrimOps._
import firrtl.Mappers._
import firrtl.WrappedExpression._
import firrtl.WrappedType._
import scala.collection.mutable.{StringBuilder, ArrayBuffer, LinkedHashMap, HashMap, HashSet}
import java.io.PrintWriter
import com.typesafe.scalalogging.LazyLogging
//import scala.reflect.runtime.universe._

class FIRRTLException(str: String) extends Exception(str)

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

  /** Removes all [[firrtl.ir.Empty]] statements and condenses
   * [[firrtl.ir.Block]] statements.
    */
  def squashEmpty(s: Statement): Statement = s map squashEmpty match {
    case Block(stmts) =>
      val newStmts = stmts filter (_ != EmptyStmt)
      newStmts.size match {
        case 0 => EmptyStmt
        case 1 => newStmts.head
        case _ => Block(newStmts)
      }
    case s => s
  }

  /** Indent the results of [[ir.FirrtlNode.serialize]] */
  def indent(str: String) = str replaceAllLiterally ("\n", "\n  ")
  def serialize(bi: BigInt): String =
    if (bi < BigInt(0)) "\"h" + bi.toString(16).substring(1) + "\""
    else "\"h" + bi.toString(16) + "\""

  implicit def toWrappedExpression (x:Expression) = new WrappedExpression(x)
  def ceil_log2(x: BigInt): BigInt = (x-1).bitLength
  def ceil_log2(x: Int): Int = scala.math.ceil(scala.math.log(x) / scala.math.log(2)).toInt
  def max(a: BigInt, b: BigInt): BigInt = if (a >= b) a else b
  def min(a: BigInt, b: BigInt): BigInt = if (a >= b) b else a
  def pow_minus_one(a: BigInt, b: BigInt): BigInt = a.pow(b.toInt) - 1
  val BoolType = UIntType(IntWidth(1))
  val one  = UIntLiteral(BigInt(1),IntWidth(1))
  val zero = UIntLiteral(BigInt(0),IntWidth(1))
  def uint (i:Int) : UIntLiteral = {
     val num_bits = req_num_bits(i)
     val w = IntWidth(scala.math.max(1,num_bits - 1))
     UIntLiteral(BigInt(i),w)
  }
  def req_num_bits (i: Int) : Int = {
     val ix = if (i < 0) ((-1 * i) - 1) else i
     ceil_log2(ix + 1) + 1
  }
  def EQV (e1:Expression,e2:Expression) : Expression =
     DoPrim(Eq, Seq(e1, e2), Nil, e1.tpe)
  // TODO: these should be fixed
  def AND (e1:WrappedExpression,e2:WrappedExpression) : Expression = {
     if (e1 == e2) e1.e1
     else if ((e1 == we(zero)) | (e2 == we(zero))) zero
     else if (e1 == we(one)) e2.e1
     else if (e2 == we(one)) e1.e1
     else DoPrim(And,Seq(e1.e1,e2.e1),Seq(),UIntType(IntWidth(1)))
  }
  def OR (e1:WrappedExpression,e2:WrappedExpression) : Expression = {
     if (e1 == e2) e1.e1
     else if ((e1 == we(one)) | (e2 == we(one))) one
     else if (e1 == we(zero)) e2.e1
     else if (e2 == we(zero)) e1.e1
     else DoPrim(Or,Seq(e1.e1,e2.e1),Seq(),UIntType(IntWidth(1)))
  }
  def NOT (e1:WrappedExpression) : Expression = {
     if (e1 == we(one)) zero
     else if (e1 == we(zero)) one
     else DoPrim(Eq,Seq(e1.e1,zero),Seq(),UIntType(IntWidth(1)))
  }

  def create_mask(dt: Type): Type = dt match {
    case t: VectorType => VectorType(create_mask(t.tpe),t.size)
    case t: BundleType => BundleType(t.fields.map (f => f.copy(tpe=create_mask(f.tpe))))
    case t: UIntType => BoolType
    case t: SIntType => BoolType
  }

  def create_exps(n: String, t: Type): Seq[Expression] =
    create_exps(WRef(n, t, ExpKind(), UNKNOWNGENDER))
  def create_exps(e: Expression): Seq[Expression] = e match {
    case (e: Mux) =>
      val e1s = create_exps(e.tval)
      val e2s = create_exps(e.fval)
      e1s zip e2s map {case (e1, e2) =>
        Mux(e.cond, e1, e2, mux_type_and_widths(e1,e2))
      }
    case (e: ValidIf) => create_exps(e.value) map (e1 => ValidIf(e.cond, e1, e1.tpe))
    case (e) => e.tpe match {
      case (_: GroundType) => Seq(e)
      case (t: BundleType) => (t.fields foldLeft Seq[Expression]())((exps, f) =>
        exps ++ create_exps(WSubField(e, f.name, f.tpe,times(gender(e), f.flip))))
      case (t: VectorType) => ((0 until t.size) foldLeft Seq[Expression]())((exps, i) =>
        exps ++ create_exps(WSubIndex(e, i, t.tpe,gender(e))))
    }
  }
   def get_flip(t: Type, i: Int, f: Orientation): Orientation = {
     if (i >= get_size(t)) error("Shouldn't be here")
     t match {
       case (_: GroundType) => f
       case (t: BundleType) => 
         val (_, flip) = ((t.fields foldLeft (i, None: Option[Orientation])){
            case ((n, ret), x) if n < get_size(x.tpe) => ret match {
              case None => (n, Some(get_flip(x.tpe,n,times(x.flip,f))))
              case Some(_) => (n, ret)
            }
            case ((n, ret), x) => (n - get_size(x.tpe), ret)
         })
         flip.get
       case (t: VectorType) => 
         val (_, flip) = (((0 until t.size) foldLeft (i, None: Option[Orientation])){
           case ((n, ret), x) if n < get_size(t.tpe) => ret match {
             case None => (n, Some(get_flip(t.tpe,n,f)))
             case Some(_) => (n, ret)
           }
           case ((n, ret), x) => (n - get_size(t.tpe), ret)
         })
         flip.get
     }
   }
  
   def get_point (e:Expression) : Int = e match {
     case (e: WRef) => 0
     case (e: WSubField) => e.exp.tpe match {case b: BundleType =>
       (b.fields takeWhile (_.name != e.name) foldLeft 0)(
         (point, f) => point + get_size(f.tpe))
    }
    case (e: WSubIndex) => e.value * get_size(e.tpe)
    case (e: WSubAccess) => get_point(e.exp)
  }

  /** Returns true if t, or any subtype, contains a flipped field
    * @param t [[firrtl.ir.Type]]
    * @return if t contains [[firrtl.ir.Flip]]
    */
  def hasFlip(t: Type): Boolean = t match {
    case t: BundleType =>
      (t.fields exists (_.flip == Flip)) ||
      (t.fields exists (f => hasFlip(f.tpe)))
    case t: VectorType => hasFlip(t.tpe)
    case _ => false
  }

//============== TYPES ================
  def mux_type (e1:Expression, e2:Expression) : Type = mux_type(e1.tpe, e2.tpe)
  def mux_type (t1:Type, t2:Type) : Type = (t1,t2) match { 
    case (t1:UIntType, t2:UIntType) => UIntType(UnknownWidth)
    case (t1:SIntType, t2:SIntType) => SIntType(UnknownWidth)
    case (t1:VectorType, t2:VectorType) => VectorType(mux_type(t1.tpe, t2.tpe), t1.size)
    case (t1:BundleType, t2:BundleType) => BundleType((t1.fields zip t2.fields) map {
      case (f1, f2) => Field(f1.name, f1.flip, mux_type(f1.tpe, f2.tpe))
    })
    case _ => UnknownType
  }
  def mux_type_and_widths (e1:Expression,e2:Expression) : Type =
    mux_type_and_widths(e1.tpe, e2.tpe)
  def mux_type_and_widths (t1:Type, t2:Type) : Type = {
    def wmax (w1:Width, w2:Width) : Width = (w1,w2) match {
      case (w1:IntWidth, w2:IntWidth) => IntWidth(w1.width max w2.width)
      case (w1, w2) => MaxWidth(Seq(w1, w2))
    }
    (t1, t2) match {
      case (t1:UIntType, t2:UIntType) => UIntType(wmax(t1.width, t2.width))
      case (t1:SIntType, t2:SIntType) => SIntType(wmax(t1.width, t2.width))
      case (t1:VectorType, t2:VectorType) => VectorType(
        mux_type_and_widths(t1.tpe, t2.tpe), t1.size)
      case (t1:BundleType, t2:BundleType) => BundleType((t1.fields zip t2.fields) map {
        case (f1, f2) => Field(f1.name,f1.flip,mux_type_and_widths(f1.tpe, f2.tpe))
      })
      case _ => UnknownType
    }
  }

  def module_type(m: DefModule): Type = BundleType(m.ports map {
    case Port(_, name, dir, tpe) => Field(name, to_flip(dir), tpe)
  })
  def sub_type(v: Type): Type = v match {
    case v: VectorType => v.tpe
    case v => UnknownType
  }
  def field_type(v:Type, s: String) : Type = v match {
    case v: BundleType => v.fields find (_.name == s) match {
      case Some(f) => f.tpe
      case None => UnknownType
    }
    case v => UnknownType
  }
   
////=====================================
  def widthBANG (t:Type) : Width = t match {
    case g: GroundType => g.width
    case t => error("No width!")
  }
  def long_BANG(t: Type): Long = t match {
    case (g: GroundType) => g.width match {
      case IntWidth(x) => x.toLong
      case _ => error(s"Expecting IntWidth, got: ${g.width}")
    }
    case (t: BundleType) => (t.fields foldLeft 0)((w, f) =>
      w + long_BANG(f.tpe).toInt)
    case (t: VectorType) => t.size * long_BANG(t.tpe)
  }

// =================================
  def error(str: String) = throw new FIRRTLException(str)

//// =============== EXPANSION FUNCTIONS ================
  def get_size(t: Type): Int = t match {
    case (t: BundleType) => (t.fields foldLeft 0)(
      (sum, f) => sum + get_size(f.tpe))
    case (t: VectorType) => t.size * get_size(t.tpe)
    case (t) => 1
  }

  def get_valid_points(t1: Type, t2: Type, flip1: Orientation, flip2: Orientation): Seq[(Int,Int)] = {
    //;println_all(["Inside with t1:" t1 ",t2:" t2 ",f1:" flip1 ",f2:" flip2])
    (t1, t2) match {
      case (t1: UIntType, t2: UIntType) => if (flip1 == flip2) Seq((0, 0)) else Nil
      case (t1: SIntType, t2: SIntType) => if (flip1 == flip2) Seq((0, 0)) else Nil
      case (t1: BundleType, t2: BundleType) =>
        def emptyMap = Map[String, (Type, Orientation, Int)]()
        val t1_fields = ((t1.fields foldLeft (emptyMap, 0)){case ((map, ilen), f1) =>
          (map + (f1.name -> (f1.tpe, f1.flip, ilen)), ilen + get_size(f1.tpe))})._1
        ((t2.fields foldLeft (Seq[(Int, Int)](), 0)){case ((points, jlen), f2) =>
          t1_fields get f2.name match {
            case None => (points, jlen + get_size(f2.tpe))
            case Some((f1_tpe, f1_flip, ilen))=>
              val f1_times = times(flip1, f1_flip)
              val f2_times = times(flip2, f2.flip)
              val ls = get_valid_points(f1_tpe, f2.tpe, f1_times, f2_times)
              (points ++ (ls map {case (x, y) => (x + ilen, y + jlen)}), jlen + get_size(f2.tpe))
          }
        })._1
      case (t1: VectorType, t2: VectorType) =>
        val size = math.min(t1.size, t2.size)
        (((0 until size) foldLeft (Seq[(Int, Int)](), 0, 0)){case ((points, ilen, jlen), _) =>
          val ls = get_valid_points(t1.tpe, t2.tpe, flip1, flip2)
          (points ++ (ls map {case (x, y) => ((x + ilen), (y + jlen))}),
            ilen + get_size(t1.tpe), jlen + get_size(t2.tpe))
        })._1
      case (ClockType, ClockType) => if (flip1 == flip2) Seq((0, 0)) else Nil
      case _ => error("shouldn't be here")
    }
  }

// =========== GENDER/FLIP UTILS ============
  def swap(g: Gender) : Gender = g match {
    case UNKNOWNGENDER => UNKNOWNGENDER
    case MALE => FEMALE
    case FEMALE => MALE
    case BIGENDER => BIGENDER
  }
  def swap(d: Direction) : Direction = d match {
    case Output => Input
    case Input => Output
  }
  def swap(f: Orientation) : Orientation = f match {
    case Default => Flip
    case Flip => Default
  }
  def to_dir(g: Gender): Direction = g match {
    case MALE => Input
    case FEMALE => Output
  }
  def to_gender(d: Direction): Gender = d match {
    case Input => MALE
    case Output => FEMALE
  }
  def to_flip(d: Direction): Orientation = d match {
    case Input => Flip
    case Output => Default
  }
  def to_flip(g: Gender): Orientation = g match {
    case MALE => Flip
    case FEMALE => Default
  }

  def field_flip(v:Type, s:String) : Orientation = v match {
    case (v:BundleType) => v.fields find (_.name == s) match {
      case Some(ft) => ft.flip
      case None => Default
    }
    case v => Default
  }
  def get_field(v:Type, s:String) : Field = v match {
    case (v:BundleType) => v.fields find (_.name == s) match {
      case Some(ft) => ft
      case None => error("Shouldn't be here")
    }
    case v => error("Shouldn't be here")
  }

  def times(flip: Orientation, d: Direction): Direction = times(flip, d)
  def times(d: Direction,flip: Orientation): Direction = flip match {
    case Default => d
    case Flip => swap(d)
  }
  def times(g: Gender, d: Direction): Direction = times(d, g)
  def times(d: Direction, g: Gender): Direction = g match {
    case FEMALE => d
    case MALE => swap(d) // MALE == INPUT == REVERSE
  }

  def times(g: Gender,flip: Orientation): Gender = times(flip, g)
  def times(flip: Orientation, g: Gender): Gender = flip match {
    case Default => g
    case Flip => swap(g)
  }
  def times(f1: Orientation, f2: Orientation): Orientation = f2 match {
    case Default => f1
    case Flip => swap(f1)
  }

// =========== ACCESSORS =========
  def kind(e: Expression): Kind = e match {
    case e: WRef => e.kind
    case e: WSubField => kind(e.exp)
    case e: WSubIndex => kind(e.exp)
    case e: WSubAccess => kind(e.exp)
    case e => ExpKind()
  }
  def gender (e: Expression): Gender = e match {
    case e: WRef => e.gender
    case e: WSubField => e.gender
    case e: WSubIndex => e.gender
    case e: WSubAccess => e.gender
    case e: DoPrim => MALE
    case e: UIntLiteral => MALE
    case e: SIntLiteral => MALE
    case e: Mux => MALE
    case e: ValidIf => MALE
    case e: WInvalid => MALE
    case e => println(e); error("Shouldn't be here")
  }
  def get_gender(s:Statement): Gender = s match {
    case s: DefWire => BIGENDER
    case s: DefRegister => BIGENDER
    case s: WDefInstance => MALE
    case s: DefNode => MALE
    case s: DefInstance => MALE
    case s: DefMemory => MALE
    case s: Block => UNKNOWNGENDER
    case s: Connect => UNKNOWNGENDER
    case s: PartialConnect => UNKNOWNGENDER
    case s: Stop => UNKNOWNGENDER
    case s: Print => UNKNOWNGENDER
    case s: IsInvalid => UNKNOWNGENDER
    case EmptyStmt => UNKNOWNGENDER
  }
  def get_gender(p: Port): Gender = if (p.direction == Input) MALE else FEMALE
  def get_type(s: Statement): Type = s match {
    case s: DefWire => s.tpe
    case s: DefRegister => s.tpe
    case s: DefNode => s.value.tpe
    case s: DefMemory =>
      val depth = s.depth
      val addr = Field("addr", Default, UIntType(IntWidth(scala.math.max(ceil_log2(depth), 1))))
      val en = Field("en", Default, BoolType)
      val clk = Field("clk", Default, ClockType)
      val def_data = Field("data", Default, s.dataType)
      val rev_data = Field("data", Flip, s.dataType)
      val mask = Field("mask", Default, create_mask(s.dataType))
      val wmode = Field("wmode", Default, UIntType(IntWidth(1)))
      val rdata = Field("rdata", Flip, s.dataType)
      val wdata = Field("wdata", Default, s.dataType)
      val wmask = Field("wmask", Default, create_mask(s.dataType))
      val read_type = BundleType(Seq(rev_data, addr, en, clk))
      val write_type = BundleType(Seq(def_data, mask, addr, en, clk))
      val readwrite_type = BundleType(Seq(wmode, rdata, wdata, wmask, addr, en, clk))
      BundleType(
        (s.readers map (Field(_, Flip, read_type))) ++
        (s.writers map (Field(_, Flip, write_type))) ++
        (s.readwriters map (Field(_, Flip, readwrite_type)))
      )
    case s: WDefInstance => s.tpe
    case _ => UnknownType
  }
  def get_name(s: Statement): String = s match {
    case s: HasName => s.name
    case _ => error("Shouldn't be here")
  }
  def get_info(s: Statement): Info = s match {
    case s: HasInfo => s.info
    case _ => NoInfo
  }

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

  case class DeclarationNotFoundException(msg: String) extends FIRRTLException(msg)

  /** Gets the root declaration of an expression
    *
    * @param m    the [[firrtl.ir.Module]] to search
    * @param expr the [[firrtl.ir.Expression]] that refers to some declaration
    * @return the [[firrtl.ir.IsDeclaration]] of `expr`
    * @throws DeclarationNotFoundException if no declaration of `expr` is found
    */
  def getDeclaration(m: Module, expr: Expression): IsDeclaration = {
    def getRootDecl(name: String)(s: Statement): Option[IsDeclaration] = s match {
      case decl: IsDeclaration => if (decl.name == name) Some(decl) else None
      case c: Conditionally =>
        val m = (getRootDecl(name)(c.conseq), getRootDecl(name)(c.alt))
        (m: @unchecked) match {
          case (Some(decl), None) => Some(decl)
          case (None, Some(decl)) => Some(decl)
          case (None, None) => None
        }
      case begin: Block =>
        val stmts = begin.stmts flatMap getRootDecl(name) // can we short circuit?
        if (stmts.nonEmpty) Some(stmts.head) else None
      case _ => None
    }
    expr match {
      case (_: WRef | _: WSubIndex | _: WSubField) =>
        val (root, tail) = splitRef(expr)
        val rootDecl = m.ports find (_.name == root.name) match {
          case Some(decl) => decl
          case None =>
            getRootDecl(root.name)(m.body) match {
              case Some(decl) => decl
              case None => throw new DeclarationNotFoundException(
                s"[module ${m.name}]  Reference ${expr.serialize} not declared!")
            }
        }
        rootDecl
      case e => error(s"getDeclaration does not support Expressions of type ${e.getClass}")
    }
  }

// =============== RECURISVE MAPPERS ===================
   def mapr (f: Width => Width, t:Type) : Type = {
      def apply_t (t:Type) : Type = t map (apply_t) map (f)
      apply_t(t)
   }
   def mapr (f: Width => Width, s:Statement) : Statement = {
      def apply_t (t:Type) : Type = mapr(f,t)
      def apply_e (e:Expression) : Expression = e map (apply_e) map (apply_t) map (f)
      def apply_s (s:Statement) : Statement = s map (apply_s) map (apply_e) map (apply_t)
      apply_s(s)
   }
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

  val v_keywords = Set(
    "alias", "always", "always_comb", "always_ff", "always_latch",
    "and", "assert", "assign", "assume", "attribute", "automatic",

    "before", "begin", "bind", "bins", "binsof", "bit", "break",
    "buf", "bufif0", "bufif1", "byte",

    "case", "casex", "casez", "cell", "chandle", "class", "clocking",
    "cmos", "config", "const", "constraint", "context", "continue",
    "cover", "covergroup", "coverpoint", "cross",

    "deassign", "default", "defparam", "design", "disable", "dist", "do",

    "edge", "else", "end", "endattribute", "endcase", "endclass",
    "endclocking", "endconfig", "endfunction", "endgenerate",
    "endgroup", "endinterface", "endmodule", "endpackage",
    "endprimitive", "endprogram", "endproperty", "endspecify",
    "endsequence", "endtable", "endtask",
    "enum", "event", "expect", "export", "extends", "extern",

    "final", "first_match", "for", "force", "foreach", "forever",
    "fork", "forkjoin", "function",

    "generate", "genvar",

    "highz0", "highz1",

    "if", "iff", "ifnone", "ignore_bins", "illegal_bins", "import",
    "incdir", "include", "initial", "initvar", "inout", "input",
    "inside", "instance", "int", "integer", "interconnect",
    "interface", "intersect",

    "join", "join_any", "join_none", "large", "liblist", "library",
    "local", "localparam", "logic", "longint",

    "macromodule", "matches", "medium", "modport", "module",

    "nand", "negedge", "new", "nmos", "nor", "noshowcancelled",
    "not", "notif0", "notif1", "null",

    "or", "output",

    "package", "packed", "parameter", "pmos", "posedge",
    "primitive", "priority", "program", "property", "protected",
    "pull0", "pull1", "pulldown", "pullup",
    "pulsestyle_onevent", "pulsestyle_ondetect", "pure",

    "rand", "randc", "randcase", "randsequence", "rcmos",
    "real", "realtime", "ref", "reg", "release", "repeat",
    "return", "rnmos", "rpmos", "rtran", "rtranif0", "rtranif1",

    "scalared", "sequence", "shortint", "shortreal", "showcancelled",
    "signed", "small", "solve", "specify", "specparam", "static",
    "strength", "string", "strong0", "strong1", "struct", "super",
    "supply0", "supply1",

    "table", "tagged", "task", "this", "throughout", "time", "timeprecision",
    "timeunit", "tran", "tranif0", "tranif1", "tri", "tri0", "tri1", "triand",
    "trior", "trireg", "type","typedef",

    "union", "unique", "unsigned", "use",

    "var", "vectored", "virtual", "void",

    "wait", "wait_order", "wand", "weak0", "weak1", "while",
    "wildcard", "wire", "with", "within", "wor",

    "xnor", "xor",

    "SYNTHESIS",
    "PRINTF_COND",
    "VCS")
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
