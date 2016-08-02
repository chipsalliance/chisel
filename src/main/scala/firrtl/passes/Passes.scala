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
import java.nio.file.{Paths, Files}

// Datastructures
import scala.collection.mutable.LinkedHashMap
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.PrimOps._
import firrtl.WrappedExpression._

trait Pass extends LazyLogging {
  def name: String
  def run(c: Circuit): Circuit
}

// Error handling
class PassException(message: String) extends Exception(message)
class PassExceptions(exceptions: Seq[PassException]) extends Exception("\n" + exceptions.mkString("\n"))
class Errors {
  val errors = ArrayBuffer[PassException]()
  def append(pe: PassException) = errors.append(pe)
  def trigger = errors.size match {
    case 0 =>
    case 1 => throw errors.head
    case _ =>
      append(new PassException(s"${errors.length} errors detected!"))
      throw new PassExceptions(errors)
  }
}

// These should be distributed into separate files
object ToWorkingIR extends Pass {
   private var mname = ""
   def name = "Working IR"
   def run (c:Circuit): Circuit = {
      def toExp (e:Expression) : Expression = {
         e map (toExp) match {
            case e:Reference => WRef(e.name, e.tpe, NodeKind(), UNKNOWNGENDER)
            case e:SubField => WSubField(e.expr, e.name, e.tpe, UNKNOWNGENDER)
            case e:SubIndex => WSubIndex(e.expr, e.value, e.tpe, UNKNOWNGENDER)
            case e:SubAccess => WSubAccess(e.expr, e.index, e.tpe, UNKNOWNGENDER)
            case e => e
         }
      }
      def toStmt (s:Statement) : Statement = {
         s map (toExp) match {
            case s:DefInstance => WDefInstance(s.info,s.name,s.module,UnknownType)
            case s => s map (toStmt)
         }
      }
      val modulesx = c.modules.map { m => 
         mname = m.name
         m match {
            case m:Module => Module(m.info,m.name, m.ports, toStmt(m.body))
            case m:ExtModule => m
         }
      }
      Circuit(c.info,modulesx,c.main)
   }
}

object ResolveKinds extends Pass {
   private var mname = ""
   def name = "Resolve Kinds"
   def run (c:Circuit): Circuit = {
      def resolve_kinds (m:DefModule, c:Circuit):DefModule = {
         val kinds = LinkedHashMap[String,Kind]()
         def resolve (body:Statement) = {
            def resolve_expr (e:Expression):Expression = {
               e match {
                  case e:WRef => WRef(e.name,tpe(e),kinds(e.name),e.gender)
                  case e => e map (resolve_expr)
               }
            }
            def resolve_stmt (s:Statement):Statement = s map (resolve_stmt) map (resolve_expr)
            resolve_stmt(body)
         }
   
         def find (m:DefModule) = {
            def find_stmt (s:Statement):Statement = {
               s match {
                  case s:DefWire => kinds(s.name) = WireKind()
                  case s:DefNode => kinds(s.name) = NodeKind()
                  case s:DefRegister => kinds(s.name) = RegKind()
                  case s:WDefInstance => kinds(s.name) = InstanceKind()
                  case s:DefMemory => kinds(s.name) = MemKind(s.readers ++ s.writers ++ s.readwriters)
                  case s => false
               }
               s map (find_stmt)
            }
            m.ports.foreach { p => kinds(p.name) = PortKind() }
            m match {
               case m:Module => find_stmt(m.body)
               case m:ExtModule => false
            }
         }
       
         mname = m.name
         find(m)   
         m match {
            case m:Module => {
               val bodyx = resolve(m.body)
               Module(m.info,m.name,m.ports,bodyx)
            }
            case m:ExtModule => ExtModule(m.info,m.name,m.ports)
         }
      }
      val modulesx = c.modules.map(m => resolve_kinds(m,c))
      Circuit(c.info,modulesx,c.main)
   }
}

object InferTypes extends Pass {
   private var mname = ""
   def name = "Infer Types"
   def set_type (s:Statement, t:Type) : Statement = {
      s match {
         case s:DefWire => DefWire(s.info,s.name,t)
         case s:DefRegister => DefRegister(s.info,s.name,t,s.clock,s.reset,s.init)
         case s:DefMemory => DefMemory(s.info,s.name,t,s.depth,s.writeLatency,s.readLatency,s.readers,s.writers,s.readwriters)
         case s:DefNode => s
      }
   }
   def remove_unknowns_w (w:Width)(implicit namespace: Namespace):Width = {
      w match {
         case UnknownWidth => VarWidth(namespace.newName("w"))
         case w => w
      }
   }
   def remove_unknowns (t:Type)(implicit n: Namespace): Type = mapr(remove_unknowns_w _,t)
   def run (c:Circuit): Circuit = {
      val module_types = LinkedHashMap[String,Type]()
      implicit val wnamespace = Namespace()
      def infer_types (m:DefModule) : DefModule = {
         val types = LinkedHashMap[String,Type]()
         def infer_types_e (e:Expression) : Expression = {
            e map (infer_types_e) match {
               case e:ValidIf => ValidIf(e.cond,e.value,tpe(e.value))
               case e:WRef => WRef(e.name, types(e.name),e.kind,e.gender)
               case e:WSubField => WSubField(e.exp,e.name,field_type(tpe(e.exp),e.name),e.gender)
               case e:WSubIndex => WSubIndex(e.exp,e.value,sub_type(tpe(e.exp)),e.gender)
               case e:WSubAccess => WSubAccess(e.exp,e.index,sub_type(tpe(e.exp)),e.gender)
               case e:DoPrim => set_primop_type(e)
               case e:Mux => Mux(e.cond,e.tval,e.fval,mux_type_and_widths(e.tval,e.fval))
               case e:UIntLiteral => e
               case e:SIntLiteral => e
            }
         }
         def infer_types_s (s:Statement) : Statement = {
            s match {
               case s:DefRegister => {
                  val t = remove_unknowns(get_type(s))
                  types(s.name) = t
                  set_type(s,t) map (infer_types_e)
               }
               case s:DefWire => {
                  val sx = s map(infer_types_e)
                  val t = remove_unknowns(get_type(sx))
                  types(s.name) = t
                  set_type(sx,t)
               }
               case s:DefNode => {
                  val sx = s map (infer_types_e)
                  val t = remove_unknowns(get_type(sx))
                  types(s.name) = t
                  set_type(sx,t)
               }
               case s:DefMemory => {
                  val t = remove_unknowns(get_type(s))
                  types(s.name) = t
                  val dt = remove_unknowns(s.dataType)
                  set_type(s,dt)
               }
               case s:WDefInstance => {
                  types(s.name) = module_types(s.module)
                  WDefInstance(s.info,s.name,s.module,module_types(s.module))
               }
               case s => s map (infer_types_s) map (infer_types_e)
            }
         }
 
         mname = m.name
         m.ports.foreach(p => types(p.name) = p.tpe)
         m match {
            case m:Module => Module(m.info,m.name,m.ports,infer_types_s(m.body))
            case m:ExtModule => m
         }
       }
 
      val modulesx = c.modules.map { 
         m => {
            mname = m.name
            val portsx = m.ports.map(p => Port(p.info,p.name,p.direction,remove_unknowns(p.tpe)))
            m match {
               case m:Module => Module(m.info,m.name,portsx,m.body)
               case m:ExtModule => ExtModule(m.info,m.name,portsx)
            }
         }
      }
      modulesx.foreach(m => module_types(m.name) = module_type(m))
      Circuit(c.info,modulesx.map({m => mname = m.name; infer_types(m)}) , c.main )
   }
}

object ResolveGenders extends Pass {
   private var mname = ""
   def name = "Resolve Genders"
   def run (c:Circuit): Circuit = {
      def resolve_e (g:Gender)(e:Expression) : Expression = {
         e match {
            case e:WRef => WRef(e.name,e.tpe,e.kind,g)
            case e:WSubField => {
               val expx = 
                  field_flip(tpe(e.exp),e.name) match {
                     case Default => resolve_e(g)(e.exp)
                     case Flip => resolve_e(swap(g))(e.exp)
                  }
               WSubField(expx,e.name,e.tpe,g)
            }
            case e:WSubIndex => {
               val expx = resolve_e(g)(e.exp)
               WSubIndex(expx,e.value,e.tpe,g)
            }
            case e:WSubAccess => {
               val expx = resolve_e(g)(e.exp)
               val indexx = resolve_e(MALE)(e.index)
               WSubAccess(expx,indexx,e.tpe,g)
            }
            case e => e map (resolve_e(g))
         }
      }
            
      def resolve_s (s:Statement) : Statement = {
         s match {
            case s:IsInvalid => {
               val expx = resolve_e(FEMALE)(s.expr)
               IsInvalid(s.info,expx)
            }
            case s:Connect => {
               val locx = resolve_e(FEMALE)(s.loc)
               val expx = resolve_e(MALE)(s.expr)
               Connect(s.info,locx,expx)
            }
            case s:PartialConnect => {
               val locx = resolve_e(FEMALE)(s.loc)
               val expx = resolve_e(MALE)(s.expr)
               PartialConnect(s.info,locx,expx)
            }
            case s => s map (resolve_e(MALE)) map (resolve_s)
         }
      }
      val modulesx = c.modules.map { 
         m => {
            mname = m.name
            m match {
               case m:Module => {
                  val bodyx = resolve_s(m.body)
                  Module(m.info,m.name,m.ports,bodyx)
               }
               case m:ExtModule => m
            }
         }
      }
      Circuit(c.info,modulesx,c.main)
   }
}

object InferWidths extends Pass {
   def name = "Infer Widths"
   var mname = ""
   def solve_constraints (l:Seq[WGeq]) : LinkedHashMap[String,Width] = {
      def unique (ls:Seq[Width]) : Seq[Width] = ls.map(w => new WrappedWidth(w)).distinct.map(_.w)
      def make_unique (ls:Seq[WGeq]) : LinkedHashMap[String,Width] = {
         val h = LinkedHashMap[String,Width]()
         for (g <- ls) {
            (g.loc) match {
               case (w:VarWidth) => {
                  val n = w.name
                  if (h.contains(n)) h(n) = MaxWidth(Seq(g.exp,h(n))) else h(n) = g.exp
               }
               case (w) => w 
            }
         }
         h 
      }
      def simplify (w:Width) : Width = {
         (w map (simplify)) match {
            case (w:MinWidth) => {
               val v = ArrayBuffer[Width]()
               for (wx <- w.args) {
                  (wx) match {
                     case (wx:MinWidth) => for (x <- wx.args) { v += x }
                     case (wx) => v += wx } }
               MinWidth(unique(v)) }
            case (w:MaxWidth) => {
               val v = ArrayBuffer[Width]()
               for (wx <- w.args) {
                  (wx) match {
                     case (wx:MaxWidth) => for (x <- wx.args) { v += x }
                     case (wx) => v += wx } }
               MaxWidth(unique(v)) }
            case (w:PlusWidth) => {
               (w.arg1,w.arg2) match {
                  case (w1:IntWidth,w2:IntWidth) => IntWidth(w1.width + w2.width)
                  case (w1,w2) => w }}
            case (w:MinusWidth) => {
               (w.arg1,w.arg2) match {
                  case (w1:IntWidth,w2:IntWidth) => IntWidth(w1.width - w2.width)
                  case (w1,w2) => w }}
            case (w:ExpWidth) => {
               (w.arg1) match {
                  case (w1:IntWidth) => IntWidth(BigInt((scala.math.pow(2,w1.width.toDouble) - 1).toLong))
                  case (w1) => w }}
            case (w) => w } }
      def substitute (h:LinkedHashMap[String,Width])(w:Width) : Width = {
         //;println-all-debug(["Substituting for [" w "]"])
         val wx = simplify(w)
         //;println-all-debug(["After Simplify: [" wx "]"])
         (simplify(w) map (substitute(h))) match {
            case (w:VarWidth) => {
               //;("matched  println-debugvarwidth!")
               if (h.contains(w.name)) {
                  //;println-debug("Contained!")
                  //;println-all-debug(["Width: " w])
                  //;println-all-debug(["Accessed: " h[name(w)]])
                  val t = simplify(substitute(h)(h(w.name)))
                  //;val t = h[name(w)]
                  //;println-all-debug(["Width after sub: " t])
                  h(w.name) = t
                  t
               } else w
            }
            case (w) => w
               //;println-all-debug(["not varwidth!" w])
         }
      }
      def b_sub (h:LinkedHashMap[String,Width])(w:Width) : Width = {
         (w map (b_sub(h))) match {
            case (w:VarWidth) => if (h.contains(w.name)) h(w.name) else w
            case (w) => w
         }
      }
      def remove_cycle (n:String)(w:Width) : Width = {
         //;println-all-debug(["Removing cycle for " n " inside " w])
         val wx = (w map (remove_cycle(n))) match {
            case (w:MaxWidth) => MaxWidth(w.args.filter{ w => {
               w match {
                  case (w:VarWidth) => !(n equals w.name)
                  case (w) => true
               }}})
            case (w:MinusWidth) => {
               w.arg1 match {
                  case (v:VarWidth) => if (n == v.name) v else w
                  case (v) => w }}
            case (w) => w
         }
         //;println-all-debug(["After removing cycle for " n ", returning " wx])
         wx
      }
      def self_rec (n:String,w:Width) : Boolean = {
         var has = false
         def look (w:Width) : Width = {
            (w map (look)) match {
               case (w:VarWidth) => if (w.name == n) has = true
               case (w) => w }
            w }
         look(w)
         has }
          
      //; Forward solve
      //; Returns a solved list where each constraint undergoes:
      //;  1) Continuous Solving (using triangular solving)
      //;  2) Remove Cycles
      //;  3) Move to solved if not self-recursive
      val u = make_unique(l)
      
      //println("======== UNIQUE CONSTRAINTS ========")
      //for (x <- u) { println(x) }
      //println("====================================")
      
   
      val f = LinkedHashMap[String,Width]()
      val o = ArrayBuffer[String]()
      for (x <- u) {
         //println("==== SOLUTIONS TABLE ====")
         //for (x <- f) println(x)
         //println("=========================")
   
         val (n, e) = (x._1, x._2)
         val e_sub = substitute(f)(e)

         //println("Solving " + n + " => " + e)
         //println("After Substitute: " + n + " => " + e_sub)
         //println("==== SOLUTIONS TABLE (Post Substitute) ====")
         //for (x <- f) println(x)
         //println("=========================")

         val ex = remove_cycle(n)(e_sub)

         //println("After Remove Cycle: " + n + " => " + ex)
         if (!self_rec(n,ex)) {
            //println("Not rec!: " + n + " => " + ex)
            //println("Adding [" + n + "=>" + ex + "] to Solutions Table")
            o += n
            f(n) = ex
         }
      }
   
      //println("Forward Solved Constraints")
      //for (x <- f) println(x)
   
      //; Backwards Solve
      val b = LinkedHashMap[String,Width]()
      for (i <- 0 until o.size) {
         val n = o(o.size - 1 - i)
         /*
         println("SOLVE BACK: [" + n + " => " + f(n) + "]")
         println("==== SOLUTIONS TABLE ====")
         for (x <- b) println(x)
         println("=========================")
         */
         val ex = simplify(b_sub(b)(f(n)))
         /*
         println("BACK RETURN: [" + n + " => " + ex + "]")
         */
         b(n) = ex
         /*
         println("==== SOLUTIONS TABLE (Post backsolve) ====")
         for (x <- b) println(x)
         println("=========================")
         */
      }
      b
   }
      
   def width_BANG (t:Type) : Width = {
      (t) match {
         case (t:UIntType) => t.width
         case (t:SIntType) => t.width
         case ClockType => IntWidth(1)
         case (t) => error("No width!"); IntWidth(-1) } }
   def width_BANG (e:Expression) : Width = width_BANG(tpe(e))

   def reduce_var_widths(c: Circuit, h: LinkedHashMap[String,Width]): Circuit = {
      def evaluate(w: Width): Width = {
         def map2(a: Option[BigInt], b: Option[BigInt], f: (BigInt,BigInt) => BigInt): Option[BigInt] =
            for (a_num <- a; b_num <- b) yield f(a_num, b_num)
         def reduceOptions(l: Seq[Option[BigInt]], f: (BigInt,BigInt) => BigInt): Option[BigInt] =
            l.reduce(map2(_, _, f))

         // This function shouldn't be necessary
         // Added as protection in case a constraint accidentally uses MinWidth/MaxWidth
         // without any actual Widths. This should be elevated to an earlier error
         def forceNonEmpty(in: Seq[Option[BigInt]], default: Option[BigInt]): Seq[Option[BigInt]] =
            if(in.isEmpty) Seq(default)
            else in


         def solve(w: Width): Option[BigInt] = w match {
            case (w: VarWidth) =>
               for{
                  v <- h.get(w.name) if !v.isInstanceOf[VarWidth]
                  result <- solve(v)
               } yield result
            case (w: MaxWidth) => reduceOptions(forceNonEmpty(w.args.map(solve _), Some(BigInt(0))), max)
            case (w: MinWidth) => reduceOptions(forceNonEmpty(w.args.map(solve _), None), min)
            case (w: PlusWidth) => map2(solve(w.arg1), solve(w.arg2), {_ + _})
            case (w: MinusWidth) => map2(solve(w.arg1), solve(w.arg2), {_ - _})
            case (w: ExpWidth) => map2(Some(BigInt(2)), solve(w.arg1), pow_minus_one)
            case (w: IntWidth) => Some(w.width)
            case (w) => println(w); error("Shouldn't be here"); None;
         }

         val s = solve(w)
         (s) match {
            case Some(s) => IntWidth(s)
            case (s) => w
         }
      }

      def reduce_var_widths_w (w:Width) : Width = {
         //println-all-debug(["REPLACE: " w])
         val wx = evaluate(w)
         //println-all-debug(["WITH: " wx])
         wx
      }
      def reduce_var_widths_s (s: Statement): Statement = {
        def onType(t: Type): Type = t map onType map reduce_var_widths_w
        s map onType
      }
   
      val modulesx = c.modules.map{ m => {
         val portsx = m.ports.map{ p => {
            Port(p.info,p.name,p.direction,mapr(reduce_var_widths_w _,p.tpe)) }}
         (m) match {
            case (m:ExtModule) => ExtModule(m.info,m.name,portsx)
            case (m:Module) =>
              mname = m.name
              Module(m.info,m.name,portsx,m.body map reduce_var_widths_s _) }}}
      InferTypes.run(Circuit(c.info,modulesx,c.main))
   }
   
   def run (c:Circuit): Circuit = {
      val v = ArrayBuffer[WGeq]()
      def constrain (w1:Width,w2:Width) : Unit = v += WGeq(w1,w2)
      def get_constraints_t (t1:Type,t2:Type,f:Orientation) : Unit = {
         (t1,t2) match {
            case (t1:UIntType,t2:UIntType) => constrain(t1.width,t2.width)
            case (t1:SIntType,t2:SIntType) => constrain(t1.width,t2.width)
            case (t1:BundleType,t2:BundleType) => {
               (t1.fields,t2.fields).zipped.foreach{ (f1,f2) => {
                  get_constraints_t(f1.tpe,f2.tpe,times(f1.flip,f)) }}}
            case (t1:VectorType,t2:VectorType) => get_constraints_t(t1.tpe,t2.tpe,f) }}
      def get_constraints_e (e:Expression) : Expression = {
         (e map (get_constraints_e)) match {
            case (e:Mux) => {
               constrain(width_BANG(e.cond),ONE)
               constrain(ONE,width_BANG(e.cond))
               e }
            case (e) => e }}
      def get_constraints (s:Statement) : Statement = {
         (s map (get_constraints_e)) match {
            case (s:Connect) => {
               val n = get_size(tpe(s.loc))
               val ce_loc = create_exps(s.loc)
               val ce_exp = create_exps(s.expr)
               for (i <- 0 until n) {
                  val locx = ce_loc(i)
                  val expx = ce_exp(i)
                  get_flip(tpe(s.loc),i,Default) match {
                     case Default => constrain(width_BANG(locx),width_BANG(expx))
                     case Flip => constrain(width_BANG(expx),width_BANG(locx)) }}
               s }
            case (s:PartialConnect) => {
               val ls = get_valid_points(tpe(s.loc),tpe(s.expr),Default,Default)
               for (x <- ls) {
                  val locx = create_exps(s.loc)(x._1)
                  val expx = create_exps(s.expr)(x._2)
                  get_flip(tpe(s.loc),x._1,Default) match {
                     case Default => constrain(width_BANG(locx),width_BANG(expx))
                     case Flip => constrain(width_BANG(expx),width_BANG(locx)) }}
               s }
            case (s:DefRegister) => {
               constrain(width_BANG(s.reset),ONE)
               constrain(ONE,width_BANG(s.reset))
               get_constraints_t(s.tpe,tpe(s.init),Default)
               s }
            case (s:Conditionally) => {
               v += WGeq(width_BANG(s.pred),ONE)
               v += WGeq(ONE,width_BANG(s.pred))
               s map (get_constraints) }
            case (s) => s map (get_constraints) }}

      for (m <- c.modules) {
         (m) match {
            case (m:Module) => mname = m.name; get_constraints(m.body)
            case (m) => false }}
      //println-debug("======== ALL CONSTRAINTS ========")
      //for x in v do : println-debug(x)
      //println-debug("=================================")
      val h = solve_constraints(v)
      //println-debug("======== SOLVED CONSTRAINTS ========")
      //for x in h do : println-debug(x)
      //println-debug("====================================")
      reduce_var_widths(Circuit(c.info,c.modules,c.main),h)
   }
}

object PullMuxes extends Pass {
   private var mname = ""
   def name = "Pull Muxes"
   def run (c:Circuit): Circuit = {
      def pull_muxes_e (e:Expression) : Expression = {
         val ex = e map (pull_muxes_e) match {
            case (e:WRef) => e
            case (e:WSubField) => {
               e.exp match {
                  case (ex:Mux) => Mux(ex.cond,WSubField(ex.tval,e.name,e.tpe,e.gender),WSubField(ex.fval,e.name,e.tpe,e.gender),e.tpe)
                  case (ex:ValidIf) => ValidIf(ex.cond,WSubField(ex.value,e.name,e.tpe,e.gender),e.tpe)
                  case (ex) => e
               }
            }
            case (e:WSubIndex) => {
               e.exp match {
                  case (ex:Mux) => Mux(ex.cond,WSubIndex(ex.tval,e.value,e.tpe,e.gender),WSubIndex(ex.fval,e.value,e.tpe,e.gender),e.tpe)
                  case (ex:ValidIf) => ValidIf(ex.cond,WSubIndex(ex.value,e.value,e.tpe,e.gender),e.tpe)
                  case (ex) => e
               }
            }
            case (e:WSubAccess) => {
               e.exp match {
                  case (ex:Mux) => Mux(ex.cond,WSubAccess(ex.tval,e.index,e.tpe,e.gender),WSubAccess(ex.fval,e.index,e.tpe,e.gender),e.tpe)
                  case (ex:ValidIf) => ValidIf(ex.cond,WSubAccess(ex.value,e.index,e.tpe,e.gender),e.tpe)
                  case (ex) => e
               }
            }
            case (e:Mux) => e
            case (e:ValidIf) => e
            case (e) => e
         }
         ex map (pull_muxes_e)
      }
      def pull_muxes (s:Statement) : Statement = s map (pull_muxes) map (pull_muxes_e)
      val modulesx = c.modules.map {
         m => {
            mname = m.name
            m match {
               case (m:Module) => Module(m.info,m.name,m.ports,pull_muxes(m.body))
               case (m:ExtModule) => m
            }
         }
      }
      Circuit(c.info,modulesx,c.main)
   }
}

object ExpandConnects extends Pass {
   private var mname = ""
   def name = "Expand Connects"
   def run (c:Circuit): Circuit = {
      def expand_connects (m:Module) : Module = {
         mname = m.name
         val genders = LinkedHashMap[String,Gender]()
         def expand_s (s:Statement) : Statement = {
            def set_gender (e:Expression) : Expression = {
               e map (set_gender) match {
                  case (e:WRef) => WRef(e.name,e.tpe,e.kind,genders(e.name))
                  case (e:WSubField) => {
                     val f = get_field(tpe(e.exp),e.name)
                     val genderx = times(gender(e.exp),f.flip)
                     WSubField(e.exp,e.name,e.tpe,genderx)
                  }
                  case (e:WSubIndex) => WSubIndex(e.exp,e.value,e.tpe,gender(e.exp))
                  case (e:WSubAccess) => WSubAccess(e.exp,e.index,e.tpe,gender(e.exp))
                  case (e) => e
               }
            }
            s match {
               case (s:DefWire) => { genders(s.name) = BIGENDER; s }
               case (s:DefRegister) => { genders(s.name) = BIGENDER; s }
               case (s:WDefInstance) => { genders(s.name) = MALE; s }
               case (s:DefMemory) => { genders(s.name) = MALE; s }
               case (s:DefNode) => { genders(s.name) = MALE; s }
               case (s:IsInvalid) => {
                  val n = get_size(tpe(s.expr))
                  val invalids = ArrayBuffer[Statement]()
                  val exps = create_exps(s.expr)
                  for (i <- 0 until n) {
                     val expx = exps(i)
                     val gexpx = set_gender(expx)
                     gender(gexpx) match {
                        case BIGENDER => invalids += IsInvalid(s.info,expx)
                        case FEMALE => invalids += IsInvalid(s.info,expx)
                        case _ => {}
                     }
                  }
                  if (invalids.length == 0) {
                     EmptyStmt
                  } else if (invalids.length == 1) {
                     invalids(0)
                  } else Block(invalids)
               }
               case (s:Connect) => {
                  val n = get_size(tpe(s.loc))
                  val connects = ArrayBuffer[Statement]()
                  val locs = create_exps(s.loc)
                  val exps = create_exps(s.expr)
                  for (i <- 0 until n) {
                     val locx = locs(i)
                     val expx = exps(i)
                     val sx = get_flip(tpe(s.loc),i,Default) match {
                        case Default => Connect(s.info,locx,expx)
                        case Flip => Connect(s.info,expx,locx)
                     }
                     connects += sx
                  }
                  Block(connects)
               }
               case (s:PartialConnect) => {
                  val ls = get_valid_points(tpe(s.loc),tpe(s.expr),Default,Default)
                  val connects = ArrayBuffer[Statement]()
                  val locs = create_exps(s.loc)
                  val exps = create_exps(s.expr)
                  ls.foreach { x => {
                     val locx = locs(x._1)
                     val expx = exps(x._2)
                     val sx = get_flip(tpe(s.loc),x._1,Default) match {
                        case Default => Connect(s.info,locx,expx)
                        case Flip => Connect(s.info,expx,locx)
                     }
                     connects += sx
                  }}
                  Block(connects)
               }
               case (s) => s map (expand_s)
            }
         }
   
         m.ports.foreach { p => genders(p.name) = to_gender(p.direction) }
         Module(m.info,m.name,m.ports,expand_s(m.body))
      }
   
      val modulesx = c.modules.map { 
         m => {
            m match {
               case (m:ExtModule) => m
               case (m:Module) => expand_connects(m)
            }
         }
      }
      Circuit(c.info,modulesx,c.main)
   }
}


// Replace shr by amount >= arg width with 0 for UInts and MSB for SInts
// TODO replace UInt with zero-width wire instead
object Legalize extends Pass {
  def name = "Legalize"
  def legalizeShiftRight (e: DoPrim): Expression = e.op match {
    case Shr => {
      val amount = e.consts(0).toInt
      val width = long_BANG(tpe(e.args(0)))
      lazy val msb = width - 1
      if (amount >= width) {
        e.tpe match {
          case t: UIntType => UIntLiteral(0, IntWidth(1))
          case t: SIntType =>
            DoPrim(Bits, e.args, Seq(msb, msb), SIntType(IntWidth(1)))
          case t => error(s"Unsupported type ${t} for Primop Shift Right")
        }
      } else {
        e
      }
    }
    case _ => e
  }
  def legalizeConnect(c: Connect): Statement = {
    val t = tpe(c.loc)
    val w = long_BANG(t)
    if (w >= long_BANG(tpe(c.expr))) c
    else {
      val newType = t match {
        case _: UIntType => UIntType(IntWidth(w))
        case _: SIntType => SIntType(IntWidth(w))
      }
      Connect(c.info, c.loc, DoPrim(Bits, Seq(c.expr), Seq(w-1, 0), newType))
    }
  }
  def run (c: Circuit): Circuit = {
    def legalizeE (e: Expression): Expression = {
      e map (legalizeE) match {
        case e: DoPrim => legalizeShiftRight(e)
        case e => e
      }
    }
    def legalizeS (s: Statement): Statement = {
      val legalizedStmt = s match {
        case c: Connect => legalizeConnect(c)
        case _ => s
      }
      legalizedStmt map legalizeS map legalizeE
    }
    def legalizeM (m: DefModule): DefModule = m map (legalizeS)
    Circuit(c.info, c.modules.map(legalizeM), c.main)
  }
}

object VerilogWrap extends Pass {
   def name = "Verilog Wrap"
   var mname = ""
   def v_wrap_e (e:Expression) : Expression = {
      e map (v_wrap_e) match {
         case (e:DoPrim) => {
            def a0 () = e.args(0)
            if (e.op == Tail) {
               (a0()) match {
                  case (e0:DoPrim) => {
                     if (e0.op == Add) DoPrim(Addw,e0.args,Seq(),tpe(e))
                     else if (e0.op == Sub) DoPrim(Subw,e0.args,Seq(),tpe(e))
                     else e
                  }
                  case (e0) => e
               }
            }
            else e
         }
         case (e) => e
      }
   }
   def v_wrap_s (s:Statement) : Statement = {
      s map (v_wrap_s) map (v_wrap_e) match {
        case s: Print =>
           Print(s.info, VerilogStringLitHandler.format(s.string), s.args, s.clk, s.en)
        case s => s
      }
   }
   def run (c:Circuit): Circuit = {
      val modulesx = c.modules.map{ m => {
         (m) match {
            case (m:Module) => {
               mname = m.name
               Module(m.info,m.name,m.ports,v_wrap_s(m.body))
            }
            case (m:ExtModule) => m
         }
      }}
      Circuit(c.info,modulesx,c.main)
   }
}

object VerilogRename extends Pass {
   def name = "Verilog Rename"
   def run (c:Circuit): Circuit = {
      def verilog_rename_n (n:String) : String = {
         if (v_keywords.contains(n)) (n + "$") else n
      }
      def verilog_rename_e (e:Expression) : Expression = {
         (e) match {
           case (e:WRef) => WRef(verilog_rename_n(e.name),e.tpe,kind(e),gender(e))
           case (e) => e map (verilog_rename_e)
         }
      }
      def verilog_rename_s (s:Statement) : Statement = {
        s map (verilog_rename_s) map (verilog_rename_e) map (verilog_rename_n)
      }
      val modulesx = c.modules.map{ m => {
         val portsx = m.ports.map{ p => {
            Port(p.info,verilog_rename_n(p.name),p.direction,p.tpe)
         }}
         m match {
            case (m:Module) => Module(m.info,m.name,portsx,verilog_rename_s(m.body))
            case (m:ExtModule) => m
         }
      }}
      Circuit(c.info,modulesx,c.main)
   }
}

object CInferTypes extends Pass {
   def name = "CInfer Types"
   var mname = ""
   def set_type (s:Statement, t:Type) : Statement = {
      (s) match { 
         case (s:DefWire) => DefWire(s.info,s.name,t)
         case (s:DefRegister) => DefRegister(s.info,s.name,t,s.clock,s.reset,s.init)
         case (s:CDefMemory) => CDefMemory(s.info,s.name,t,s.size,s.seq)
         case (s:CDefMPort) => CDefMPort(s.info,s.name,t,s.mem,s.exps,s.direction)
         case (s:DefNode) => s
      }
   }
   
   def to_field (p:Port) : Field = {
      if (p.direction == Output) Field(p.name,Default,p.tpe)
      else if (p.direction == Input) Field(p.name,Flip,p.tpe)
      else error("Shouldn't be here"); Field(p.name,Flip,p.tpe)
   }
   def module_type (m:DefModule) : Type = BundleType(m.ports.map(p => to_field(p)))
   def field_type (v:Type,s:String) : Type = {
      (v) match { 
         case (v:BundleType) => {
            val ft = v.fields.find(p => p.name == s)
            if (ft != None) ft.get.tpe
            else  UnknownType
         }
         case (v) => UnknownType
      }
   }
   def sub_type (v:Type) : Type =
      (v) match { 
         case (v:VectorType) => v.tpe
         case (v) => UnknownType
      }
   def run (c:Circuit) : Circuit = {
      val module_types = LinkedHashMap[String,Type]()
      def infer_types (m:DefModule) : DefModule = {
         val types = LinkedHashMap[String,Type]()
         def infer_types_e (e:Expression) : Expression = {
            e map infer_types_e match {
               case (e:Reference) => Reference(e.name, types.getOrElse(e.name,UnknownType))
               case (e:SubField) => SubField(e.expr,e.name,field_type(tpe(e.expr),e.name))
               case (e:SubIndex) => SubIndex(e.expr,e.value,sub_type(tpe(e.expr)))
               case (e:SubAccess) => SubAccess(e.expr,e.index,sub_type(tpe(e.expr)))
               case (e:DoPrim) => set_primop_type(e)
               case (e:Mux) => Mux(e.cond,e.tval,e.fval,mux_type(e.tval,e.tval))
               case (e:ValidIf) => ValidIf(e.cond,e.value,tpe(e.value))
               case (_:UIntLiteral | _:SIntLiteral) => e
            }
         }
         def infer_types_s (s:Statement) : Statement = {
            s match {
               case (s:DefRegister) => {
                  types(s.name) = s.tpe
                  s map infer_types_e
                  s
               }
               case (s:DefWire) => {
                  types(s.name) = s.tpe
                  s
               }
               case (s:DefNode) => {
                  val sx = s map infer_types_e
                  val t = get_type(sx)
                  types(s.name) = t
                  sx
               }
               case (s:DefMemory) => {
                  types(s.name) = get_type(s)
                  s
               }
               case (s:CDefMPort) => {
                  val t = types.getOrElse(s.mem,UnknownType)
                  types(s.name) = t
                  CDefMPort(s.info,s.name,t,s.mem,s.exps,s.direction)
               }
               case (s:CDefMemory) => {
                  types(s.name) = s.tpe
                  s
               }
               case (s:DefInstance) => {
                  types(s.name) = module_types.getOrElse(s.module,UnknownType)
                  s
               }
               case (s) => s map infer_types_s map infer_types_e
            }
         }
         for (p <- m.ports) {
            types(p.name) = p.tpe
         }
         m match {
            case (m:Module) => Module(m.info,m.name,m.ports,infer_types_s(m.body))
            case (m:ExtModule) => m
         }
      }
   
      //; MAIN
      for (m <- c.modules) {
         module_types(m.name) = module_type(m)
      }
      val modulesx = c.modules.map(m => infer_types(m))
      Circuit(c.info, modulesx, c.main)
   }
}

object CInferMDir extends Pass {
   def name = "CInfer MDir"
   var mname = ""
   def run (c:Circuit) : Circuit = {
      def infer_mdir (m:DefModule) : DefModule = {
         val mports = LinkedHashMap[String,MPortDir]()
         def infer_mdir_e (dir:MPortDir)(e:Expression) : Expression = {
            (e map (infer_mdir_e(dir))) match { 
               case (e:Reference) => {
                  if (mports.contains(e.name)) {
                     val new_mport_dir = {
                        (mports(e.name),dir) match {
                           case (MInfer,MInfer) => error("Shouldn't be here")
                           case (MInfer,MWrite) => MWrite
                           case (MInfer,MRead) => MRead
                           case (MInfer,MReadWrite) => MReadWrite
                           case (MWrite,MInfer) => error("Shouldn't be here")
                           case (MWrite,MWrite) => MWrite
                           case (MWrite,MRead) => MReadWrite
                           case (MWrite,MReadWrite) => MReadWrite
                           case (MRead,MInfer) => error("Shouldn't be here")
                           case (MRead,MWrite) => MReadWrite
                           case (MRead,MRead) => MRead
                           case (MRead,MReadWrite) => MReadWrite
                           case (MReadWrite,MInfer) => error("Shouldn't be here")
                           case (MReadWrite,MWrite) => MReadWrite
                           case (MReadWrite,MRead) => MReadWrite
                           case (MReadWrite,MReadWrite) => MReadWrite
                        }
                     }
                     mports(e.name) = new_mport_dir
                  }
                  e
               }
               case (e) => e
            }
         }
         def infer_mdir_s (s:Statement) : Statement = {
            (s) match { 
               case (s:CDefMPort) => {
                  mports(s.name) = s.direction
                  s map (infer_mdir_e(MRead))
               }
               case (s:Connect) => {
                  infer_mdir_e(MRead)(s.expr)
                  infer_mdir_e(MWrite)(s.loc)
                  s
               }
               case (s:PartialConnect) => {
                  infer_mdir_e(MRead)(s.expr)
                  infer_mdir_e(MWrite)(s.loc)
                  s
               }
               case (s) => s map (infer_mdir_s) map (infer_mdir_e(MRead))
            }
         }
         def set_mdir_s (s:Statement) : Statement = {
            (s) match { 
               case (s:CDefMPort) => 
                  CDefMPort(s.info,s.name,s.tpe,s.mem,s.exps,mports(s.name))
               case (s) => s map (set_mdir_s)
            }
         }
         (m) match { 
            case (m:Module) => {
               infer_mdir_s(m.body)
               Module(m.info,m.name,m.ports,set_mdir_s(m.body))
            }
            case (m:ExtModule) => m
         }
      }
   
      //; MAIN
      Circuit(c.info, c.modules.map(m => infer_mdir(m)), c.main)
   }
}

case class MPort( val name : String, val clk : Expression)
case class MPorts( val readers : ArrayBuffer[MPort], val writers : ArrayBuffer[MPort], val readwriters : ArrayBuffer[MPort])
case class DataRef( val exp : Expression, val male : String, val female : String, val mask : String, val rdwrite : Boolean)

object RemoveCHIRRTL extends Pass {
   def name = "Remove CHIRRTL"
   var mname = ""
   def create_exps (e:Expression) : Seq[Expression] = {
      (e) match { 
         case (e:Mux)=> 
            (create_exps(e.tval),create_exps(e.fval)).zipped.map((e1,e2) => {
               Mux(e.cond,e1,e2,mux_type(e1,e2))
            })
         case (e:ValidIf) => 
            create_exps(e.value).map(e1 => {
               ValidIf(e.cond,e1,tpe(e1))
            })
         case (e) => (tpe(e)) match  { 
            case (_:UIntType|_:SIntType|ClockType) => Seq(e)
            case (t:BundleType) => 
               t.fields.flatMap(f => create_exps(SubField(e,f.name,f.tpe)))
            case (t:VectorType)=> 
               (0 until t.size).flatMap(i => create_exps(SubIndex(e,i,t.tpe)))
            case UnknownType => Seq(e)
         }
      }
   }
   def run (c:Circuit) : Circuit = {
      def remove_chirrtl_m (m:Module) : Module = {
         val hash = LinkedHashMap[String,MPorts]()
         val repl = LinkedHashMap[String,DataRef]()
         val raddrs = HashMap[String, Expression]()
         val ut = UnknownType
         val mport_types = LinkedHashMap[String,Type]()
         def EMPs () : MPorts = MPorts(ArrayBuffer[MPort](),ArrayBuffer[MPort](),ArrayBuffer[MPort]())
         def collect_mports (s:Statement) : Statement = {
            (s) match { 
               case (s:CDefMPort) => {
                  val mports = hash.getOrElse(s.mem,EMPs())
                  s.direction match {
                     case MRead => mports.readers += MPort(s.name,s.exps(1))
                     case MWrite => mports.writers += MPort(s.name,s.exps(1))
                     case MReadWrite => mports.readwriters += MPort(s.name,s.exps(1))
                  }
                  hash(s.mem) = mports
                  s
               }
               case (s) => s map (collect_mports)
            }
         }
         def collect_refs (s:Statement) : Statement = {
            (s) match { 
               case (s:CDefMemory) => {
                  mport_types(s.name) = s.tpe
                  val stmts = ArrayBuffer[Statement]()
                  val taddr = UIntType(IntWidth(scala.math.max(1,ceil_log2(s.size))))
                  val tdata = s.tpe
                  def set_poison (vec:Seq[MPort],addr:String) : Unit = {
                     for (r <- vec ) {
                        stmts += IsInvalid(s.info,SubField(SubField(Reference(s.name,ut),r.name,ut),addr,taddr))
                        stmts += IsInvalid(s.info,SubField(SubField(Reference(s.name,ut),r.name,ut),"clk",taddr))
                     }
                  }
                  def set_enable (vec:Seq[MPort],en:String) : Unit = {
                     for (r <- vec ) {
                        stmts += Connect(s.info,SubField(SubField(Reference(s.name,ut),r.name,ut),en,taddr),zero)
                     }}
                  def set_wmode (vec:Seq[MPort],wmode:String) : Unit = {
                     for (r <- vec) {
                        stmts += Connect(s.info,SubField(SubField(Reference(s.name,ut),r.name,ut),wmode,taddr),zero)
                     }}
                  def set_write (vec:Seq[MPort],data:String,mask:String) : Unit = {
                     val tmask = create_mask(s.tpe)
                     for (r <- vec ) {
                        stmts += IsInvalid(s.info,SubField(SubField(Reference(s.name,ut),r.name,ut),data,tdata))
                        for (x <- create_exps(SubField(SubField(Reference(s.name,ut),r.name,ut),mask,tmask)) ) {
                           stmts += Connect(s.info,x,zero)
                        }}}
                  val rds = (hash.getOrElse(s.name,EMPs())).readers
                  set_poison(rds,"addr")
                  set_enable(rds,"en")
                  val wrs = (hash.getOrElse(s.name,EMPs())).writers
                  set_poison(wrs,"addr")
                  set_enable(wrs,"en")
                  set_write(wrs,"data","mask")
                  val rws = (hash.getOrElse(s.name,EMPs())).readwriters
                  set_poison(rws,"addr")
                  set_wmode(rws,"wmode")
                  set_enable(rws,"en")
                  set_write(rws,"data","mask")
                  val read_l = if (s.seq) 1 else 0
                  val mem = DefMemory(s.info,s.name,s.tpe,s.size,1,read_l,rds.map(_.name),wrs.map(_.name),rws.map(_.name))
                  Block(Seq(mem,Block(stmts)))
               }
               case (s:CDefMPort) => {
                  mport_types(s.name) = mport_types(s.mem)
                  val addrs = ArrayBuffer[String]()
                  val clks = ArrayBuffer[String]()
                  val ens = ArrayBuffer[String]()
                  val masks = ArrayBuffer[String]()
                  s.direction match {
                     case MReadWrite => {
                        repl(s.name) = DataRef(SubField(Reference(s.mem,ut),s.name,ut),"rdata","data","mask",true)
                        addrs += "addr"
                        clks += "clk"
                        ens += "en"
                        masks += "mask"
                     }
                     case MWrite => {
                        repl(s.name) = DataRef(SubField(Reference(s.mem,ut),s.name,ut),"data","data","mask",false)
                        addrs += "addr"
                        clks += "clk"
                        ens += "en"
                        masks += "mask"
                     }
                     case MRead => {
                        repl(s.name) = DataRef(SubField(Reference(s.mem,ut),s.name,ut),"data","data","blah",false)
                        addrs += "addr"
                        clks += "clk"
                        s.exps(0) match {
                           case e: Reference =>
                              raddrs(e.name) = SubField(SubField(Reference(s.mem,ut),s.name,ut),"en",ut)
                           case _=>
                        }
                     }
                  }
                  val stmts = ArrayBuffer[Statement]()
                  for (x <- addrs ) {
                     stmts += Connect(s.info,SubField(SubField(Reference(s.mem,ut),s.name,ut),x,ut),s.exps(0))
                  }
                  for (x <- clks ) {
                     stmts += Connect(s.info,SubField(SubField(Reference(s.mem,ut),s.name,ut),x,ut),s.exps(1))
                  }
                  for (x <- ens ) {
                     stmts += Connect(s.info,SubField(SubField(Reference(s.mem,ut),s.name,ut),x,ut),one)
                  }
                  Block(stmts)
               }
               case (s) => s map (collect_refs)
            }
         }
         def remove_chirrtl_s (s:Statement) : Statement = {
            var has_write_mport = false
            var has_read_mport: Option[Expression] = None
            var has_readwrite_mport:Option[Expression] = None
            def remove_chirrtl_e (g:Gender)(e:Expression) : Expression = {
               (e) match {
                  case (e:Reference) if repl contains e.name =>
                     val vt = repl(e.name)
                     g match {
                        case MALE => SubField(vt.exp,vt.male,e.tpe)
                        case FEMALE => {
                           has_write_mport = true
                           if (vt.rdwrite) 
                              has_readwrite_mport = Some(SubField(vt.exp,"wmode",UIntType(IntWidth(1))))
                           SubField(vt.exp,vt.female,e.tpe)
                        }
                     }
                  case (e:Reference) if g == FEMALE && (raddrs contains e.name) =>
                     has_read_mport = Some(raddrs(e.name))
                     e
                  case (e:Reference) => e
                  case (e:SubAccess) => SubAccess(remove_chirrtl_e(g)(e.expr),remove_chirrtl_e(MALE)(e.index),e.tpe)
                  case (e) => e map (remove_chirrtl_e(g))
               }
            }
            def get_mask (e:Expression) : Expression = {
               (e map (get_mask)) match { 
                  case (e:Reference) => {
                     if (repl.contains(e.name)) {
                        val vt = repl(e.name)
                        val t = create_mask(e.tpe)
                        SubField(vt.exp,vt.mask,t)
                     } else e
                  }
                  case (e) => e
               }
            }
            (s) match {
               case (s:DefNode) => {
                  val stmts = ArrayBuffer[Statement]()
                  val valuex = remove_chirrtl_e(MALE)(s.value)
                  stmts += DefNode(s.info,s.name,valuex)
                  has_read_mport match {
                    case None =>
                    case Some(en) => stmts += Connect(s.info,en,one)
                  }
                  if (stmts.size > 1) Block(stmts)
                  else stmts(0)
               }
               case (s:Connect) => {
                  val stmts = ArrayBuffer[Statement]()
                  val rocx = remove_chirrtl_e(MALE)(s.expr)
                  val locx = remove_chirrtl_e(FEMALE)(s.loc)
                  stmts += Connect(s.info,locx,rocx)
                  has_read_mport match {
                    case None =>
                    case Some(en) => stmts += Connect(s.info,en,one)
                  }
                  if (has_write_mport) {
                     val e = get_mask(s.loc)
                     for (x <- create_exps(e) ) {
                        stmts += Connect(s.info,x,one)
                     }
                     has_readwrite_mport match {
                        case None =>
                        case Some(wmode) => stmts += Connect(s.info,wmode,one)
                     }
                  }
                  if (stmts.size > 1) Block(stmts)
                  else stmts(0)
               }
               case (s:PartialConnect) => {
                  val stmts = ArrayBuffer[Statement]()
                  val locx = remove_chirrtl_e(FEMALE)(s.loc)
                  val rocx = remove_chirrtl_e(MALE)(s.expr)
                  stmts += PartialConnect(s.info,locx,rocx)
                  has_read_mport match {
                    case None =>
                    case Some(en) => stmts += Connect(s.info,en,one)
                  }
                  if (has_write_mport) {
                     val ls = get_valid_points(tpe(s.loc),tpe(s.expr),Default,Default)
                     val locs = create_exps(get_mask(s.loc))
                     for (x <- ls ) {
                        val locx = locs(x._1)
                        stmts += Connect(s.info,locx,one)
                     }
                     has_readwrite_mport match {
                        case None =>
                        case Some(wmode) => stmts += Connect(s.info,wmode,one)
                     }
                  }
                  if (stmts.size > 1) Block(stmts)
                  else stmts(0)
               }
               case (s) => s map (remove_chirrtl_s) map (remove_chirrtl_e(MALE))
            }
         }
         collect_mports(m.body)
         val sx = collect_refs(m.body)
         Module(m.info,m.name, m.ports, remove_chirrtl_s(sx))
      }
      val modulesx = c.modules.map{ m => {
            (m) match { 
               case (m:Module) => remove_chirrtl_m(m)
               case (m:ExtModule) => m
            }}}
      Circuit(c.info,modulesx, c.main)
   }
}
