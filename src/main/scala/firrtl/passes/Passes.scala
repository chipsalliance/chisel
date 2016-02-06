
package firrtl.passes

import com.typesafe.scalalogging.LazyLogging
import java.nio.file.{Paths, Files}

// For calling Stanza 
import scala.sys.process._
import scala.io.Source

// Datastructures
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer

import firrtl._
import firrtl.Utils._
import firrtl.PrimOps._

trait Pass extends LazyLogging {
  def name: String
  def run(c: Circuit): Circuit
}

// Trait for migration, trap to Stanza implementation for passes not yet implemented in Scala
trait StanzaPass extends LazyLogging {
  def stanzaPass(c: Circuit, n: String): Circuit = {
    // For migration from Stanza, handle unimplemented Passes
    logger.debug(s"Pass ${n} is not yet implemented in Scala")
    val stanzaPasses = Seq("resolve", n) 
    val toStanza = Files.createTempFile(Paths.get(""), n, ".fir")
    val fromStanza = Files.createTempFile(Paths.get(""), n, ".fir")
    Files.write(toStanza, c.serialize.getBytes)

    val cmd = Seq("firrtl-stanza", "-i", toStanza.toString, "-o", fromStanza.toString, "-b", "firrtl", "-p", "c") ++ 
              stanzaPasses.flatMap(x=>Seq("-x", x))
    logger.debug(cmd.mkString(" "))
    val ret = cmd.!
    //println(ret)
    val newC = Parser.parse(fromStanza.toString, Source.fromFile(fromStanza.toString).getLines)
    Files.delete(toStanza)
    Files.delete(fromStanza)
    newC
  }
}

object PassUtils extends LazyLogging {
  val listOfPasses: Seq[Pass] = Seq(ToWorkingIR,ResolveKinds,InferTypes,ResolveGenders,InferWidths,PullMuxes,ExpandConnects,RemoveAccesses,ExpandWhens,LowerTypes)
  lazy val mapNameToPass: Map[String, Pass] = listOfPasses.map(p => p.name -> p).toMap

  def executePasses(c: Circuit, passes: Seq[Pass]): Circuit = { 
    if (passes.isEmpty) c
    else {
       val p = passes.head
       val name = p.name
       logger.debug(s"Starting ${name}")
       val x = p.run(c)
       logger.debug(x.serialize())
       logger.debug(s"Finished ${name}")
       executePasses(x, passes.tail)
    }
  }
}

// These should be distributed into separate files
object CheckHighForm extends Pass with StanzaPass {
  def name = "High Form Check"
  def run (c:Circuit): Circuit = stanzaPass(c, "high-form-check")
}

object ToWorkingIR extends Pass {
   private var mname = ""
   def name = "Working IR"
   def run (c:Circuit): Circuit = {
      def toExp (e:Expression) : Expression = {
         eMap(toExp _,e) match {
            case e:Ref => WRef(e.name, e.tpe, NodeKind(), UNKNOWNGENDER)
            case e:SubField => WSubField(e.exp, e.name, e.tpe, UNKNOWNGENDER)
            case e:SubIndex => WSubIndex(e.exp, e.value, e.tpe, UNKNOWNGENDER)
            case e:SubAccess => WSubAccess(e.exp, e.index, e.tpe, UNKNOWNGENDER)
            case e => e
         }
      }
      def toStmt (s:Stmt) : Stmt = {
         eMap(toExp _,s) match {
            case s:DefInstance => WDefInstance(s.info,s.name,s.module,UnknownType())
            case s => sMap(toStmt _,s)
         }
      }
      val modulesx = c.modules.map { m => 
         mname = m.name
         m match {
            case m:InModule => InModule(m.info,m.name, m.ports, toStmt(m.body))
            case m:ExModule => m
         }
      }
      Circuit(c.info,modulesx,c.main)
   }
}

object Resolve extends Pass with StanzaPass {
  def name = "Resolve"
  def run (c:Circuit): Circuit = stanzaPass(c, "resolve")
}

object ResolveKinds extends Pass {
   private var mname = ""
   def name = "Resolve Kinds"
   def run (c:Circuit): Circuit = {
      def resolve_kinds (m:Module, c:Circuit):Module = {
         val kinds = HashMap[String,Kind]()
         def resolve (body:Stmt) = {
            def resolve_expr (e:Expression):Expression = {
               e match {
                  case e:WRef => WRef(e.name,tpe(e),kinds(e.name),e.gender)
                  case e => eMap(resolve_expr,e)
               }
            }
            def resolve_stmt (s:Stmt):Stmt = eMap(resolve_expr,sMap(resolve_stmt,s))
            resolve_stmt(body)
         }
   
         def find (m:Module) = {
            def find_stmt (s:Stmt):Stmt = {
               s match {
                  case s:DefWire => kinds += (s.name -> WireKind())
                  case s:DefPoison => kinds += (s.name -> PoisonKind())
                  case s:DefNode => kinds += (s.name -> NodeKind())
                  case s:DefRegister => kinds += (s.name -> RegKind())
                  case s:WDefInstance => kinds += (s.name -> InstanceKind())
                  case s:DefMemory => kinds += (s.name -> MemKind(s.readers ++ s.writers ++ s.readwriters))
                  case s => false
               }
               sMap(find_stmt,s)
            }
            m.ports.foreach { p => kinds += (p.name -> PortKind()) }
            m match {
               case m:InModule => find_stmt(m.body)
               case m:ExModule => false
            }
         }
       
         mname = m.name
         find(m)   
         m match {
            case m:InModule => {
               val bodyx = resolve(m.body)
               InModule(m.info,m.name,m.ports,bodyx)
            }
            case m:ExModule => ExModule(m.info,m.name,m.ports)
         }
      }
      val modulesx = c.modules.map(m => resolve_kinds(m,c))
      Circuit(c.info,modulesx,c.main)
   }
}

object InferTypes extends Pass {
   private var mname = ""
   def name = "Infer Types"
   val width_name_hash = HashMap[String,Int]()
   def set_type (s:Stmt,t:Type) : Stmt = {
      s match {
         case s:DefWire => DefWire(s.info,s.name,t)
         case s:DefRegister => DefRegister(s.info,s.name,t,s.clock,s.reset,s.init)
         case s:DefMemory => DefMemory(s.info,s.name,t,s.depth,s.write_latency,s.read_latency,s.readers,s.writers,s.readwriters)
         case s:DefNode => s
         case s:DefPoison => DefPoison(s.info,s.name,t)
      }
   }
   def remove_unknowns_w (w:Width):Width = {
      w match {
         case w:UnknownWidth => VarWidth(firrtl_gensym("w",width_name_hash))
         case w => w
      }
   }
   def remove_unknowns (t:Type): Type = mapr(remove_unknowns_w _,t)
   def run (c:Circuit): Circuit = {
      val module_types = HashMap[String,Type]()
      def infer_types (m:Module) : Module = {
         val types = HashMap[String,Type]()
         def infer_types_e (e:Expression) : Expression = {
            eMap(infer_types_e _,e) match {
               case e:ValidIf => ValidIf(e.cond,e.value,tpe(e.value))
               case e:WRef => WRef(e.name, types(e.name),e.kind,e.gender)
               case e:WSubField => WSubField(e.exp,e.name,field_type(tpe(e.exp),e.name),e.gender)
               case e:WSubIndex => WSubIndex(e.exp,e.value,sub_type(tpe(e.exp)),e.gender)
               case e:WSubAccess => WSubAccess(e.exp,e.index,sub_type(tpe(e.exp)),e.gender)
               case e:DoPrim => set_primop_type(e)
               case e:Mux => Mux(e.cond,e.tval,e.fval,mux_type_and_widths(e.tval,e.fval))
               case e:UIntValue => e
               case e:SIntValue => e
            }
         }
         def infer_types_s (s:Stmt) : Stmt = {
            s match {
               case s:DefRegister => {
                  val t = remove_unknowns(get_type(s))
                  types += (s.name -> t)
                  eMap(infer_types_e _,set_type(s,t))
               }
               case s:DefWire => {
                  val sx = eMap(infer_types_e _,s)
                  val t = remove_unknowns(get_type(sx))
                  types += (s.name -> t)
                  set_type(sx,t)
               }
               case s:DefPoison => {
                  val sx = eMap(infer_types_e _,s)
                  val t = remove_unknowns(get_type(sx))
                  types += (s.name -> t)
                  set_type(sx,t)
               }
               case s:DefNode => {
                  val sx = eMap(infer_types_e _,s)
                  val t = remove_unknowns(get_type(sx))
                  types += (s.name -> t)
                  set_type(sx,t)
               }
               case s:DefMemory => {
                  val t = remove_unknowns(get_type(s))
                  types += (s.name -> t)
                  val dt = remove_unknowns(s.data_type)
                  set_type(s,dt)
               }
               case s:WDefInstance => {
                  types += (s.name -> module_types(s.module))
                  WDefInstance(s.info,s.name,s.module,module_types(s.module))
               }
               case s => eMap(infer_types_e _,sMap(infer_types_s,s))
            }
         }
 
         mname = m.name
         m.ports.foreach(p => types += (p.name -> p.tpe))
         m match {
            case m:InModule => InModule(m.info,m.name,m.ports,infer_types_s(m.body))
            case m:ExModule => m
         }
       }
 
      val modulesx = c.modules.map { 
         m => {
            mname = m.name
            val portsx = m.ports.map(p => Port(p.info,p.name,p.direction,remove_unknowns(p.tpe)))
            m match {
               case m:InModule => InModule(m.info,m.name,portsx,m.body)
               case m:ExModule => ExModule(m.info,m.name,portsx)
            }
         }
      }
      modulesx.foreach(m => module_types += (m.name -> module_type(m)))
      Circuit(c.info,modulesx.map({m => mname = m.name; infer_types(m)}) , c.main )
   }
}
object CheckTypes extends Pass with StanzaPass {
  def name = "Check Types"
  def run (c:Circuit): Circuit = stanzaPass(c, "check-types")
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
                     case DEFAULT => resolve_e(g)(e.exp)
                     case REVERSE => resolve_e(swap(g))(e.exp)
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
            case e => eMap(resolve_e(g) _,e)
         }
      }
            
      def resolve_s (s:Stmt) : Stmt = {
         s match {
            case s:IsInvalid => {
               val expx = resolve_e(FEMALE)(s.exp)
               IsInvalid(s.info,expx)
            }
            case s:Connect => {
               val locx = resolve_e(FEMALE)(s.loc)
               val expx = resolve_e(MALE)(s.exp)
               Connect(s.info,locx,expx)
            }
            case s:BulkConnect => {
               val locx = resolve_e(FEMALE)(s.loc)
               val expx = resolve_e(MALE)(s.exp)
               BulkConnect(s.info,locx,expx)
            }
            case s => sMap(resolve_s,eMap(resolve_e(MALE) _,s))
         }
      }
      val modulesx = c.modules.map { 
         m => {
            mname = m.name
            m match {
               case m:InModule => {
                  val bodyx = resolve_s(m.body)
                  InModule(m.info,m.name,m.ports,bodyx)
               }
               case m:ExModule => m
            }
         }
      }
      Circuit(c.info,modulesx,c.main)
   }
}

object CheckGenders extends Pass with StanzaPass {
   def name = "Check Genders"
   def run (c:Circuit): Circuit = stanzaPass(c, "check-genders")
}

object InferWidths extends Pass {
   def name = "Infer Widths"
   var mname = ""
   def solve_constraints (l:Seq[WGeq]) : HashMap[String,Width] = {
      def unique (ls:Seq[Width]) : Seq[Width] = ls.map(w => new WrappedWidth(w)).distinct.map(_.w)
      def make_unique (ls:Seq[WGeq]) : HashMap[String,Width] = {
         val h = HashMap[String,Width]()
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
         (wMap(simplify _,w)) match {
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
                  case (w1:IntWidth) => IntWidth((2 ^ w1.width) - 1)
                  case (w1) => w }}
            case (w) => w } }
      def substitute (h:HashMap[String,Width])(w:Width) : Width = {
         //;println-all-debug(["Substituting for [" w "]"])
         val wx = simplify(w)
         //;println-all-debug(["After Simplify: [" wx "]"])
         (wMap(substitute(h) _,simplify(w))) match {
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
      def b_sub (h:HashMap[String,Width])(w:Width) : Width = {
         (wMap(b_sub(h) _,w)) match {
            case (w:VarWidth) => if (h.contains(w.name)) h(w.name) else w
            case (w) => w
         }
      }
      def remove_cycle (n:String)(w:Width) : Width = {
         //;println-all-debug(["Removing cycle for " n " inside " w])
         val wx = (wMap(remove_cycle(n) _,w)) match {
            case (w:MaxWidth) => MaxWidth(w.args.filter{ w => {
               w match {
                  case (w:VarWidth) => n equals w.name
                  case (w) => false
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
            (wMap(look _,w)) match {
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
      //println-debug("======== UNIQUE CONSTRAINTS ========")
      //for (x <- u) { println-debug(x) }
      //println-debug("====================================")
   
      val f = HashMap[String,Width]()
      val o = ArrayBuffer[String]()
      for (x <- u) {
         //println-debug("==== SOLUTIONS TABLE ====")
         //for x in f do : println-debug(x)
         //println-debug("=========================")
   
         val (n, e) = (x._1, x._2)
   
         val e_sub = substitute(f)(e)
         //println-debug(["Solving " n " => " e])
         //println-debug(["After Substitute: " n " => " e-sub])
         //println-debug("==== SOLUTIONS TABLE (Post Substitute) ====")
         //for x in f do : println-debug(x)
         //println-debug("=========================")
         val ex = remove_cycle(n)(e_sub)
         //;println-debug(["After Remove Cycle: " n " => " ex])
         if (!self_rec(n,ex)) {
            //;println-all-debug(["Not rec!: " n " => " ex])
            //;println-all-debug(["Adding [" n "=>" ex "] to Solutions Table"])
            o += n
            f(n) = ex
         }
      }
   
      //println-debug("Forward Solved Constraints")
      //for x in f do : println-debug(x)
   
      //; Backwards Solve
      val b = HashMap[String,Width]()
      for (i <- 0 until o.size) {
         val n = o(o.size - 1 - i)
         //println-all-debug(["SOLVE BACK: [" n " => " f[n] "]"])
         //println-debug("==== SOLUTIONS TABLE ====")
         //for x in b do : println-debug(x)
         //println-debug("=========================")
         val ex = simplify(b_sub(b)(f(n)))
         //println-all-debug(["BACK RETURN: [" n " => " ex "]"])
         b(n) = ex
         //println-debug("==== SOLUTIONS TABLE (Post backsolve) ====")
         //for x in b do : println-debug(x)
         //println-debug("=========================")
      }
      b
   }
      
   def width_BANG (t:Type) : Width = {
      (t) match {
         case (t:UIntType) => t.width
         case (t:SIntType) => t.width
         case (t:ClockType) => IntWidth(1)
         case (t) => error("No width!"); IntWidth(-1) } }
   def width_BANG (e:Expression) : Width = width_BANG(tpe(e))
   def reduce_var_widths (c:Circuit,h:HashMap[String,Width]) : Circuit = {
      def evaluate (w:Width) : Width = {
         def apply_2 (a:Option[BigInt],b:Option[BigInt], f: (BigInt,BigInt) => BigInt) : Option[BigInt] = {
            (a,b) match {
               case (a:Some[BigInt],b:Some[BigInt]) => Some(f(a.get,b.get))
               case (a,b) => None } }
         def apply_1 (a:Option[BigInt], f: (BigInt) => BigInt) : Option[BigInt] = {
            (a) match {
               case (a:Some[BigInt]) => Some(f(a.get))
               case (a) => None } }
         def apply_l (l:Seq[Option[BigInt]],f:(BigInt,BigInt) => BigInt) : Option[BigInt] = {
            if (l.size == 0) Some(BigInt(0)) else apply_2(l.head,apply_l(l.tail,f),f) 
         }
         def max (a:BigInt,b:BigInt) : BigInt = if (a >= b) a else b
         def min (a:BigInt,b:BigInt) : BigInt = if (a >= b) b else a
         def solve (w:Width) : Option[BigInt] = {
            (w) match {
               case (w:VarWidth) => {
                  val wx = h.get(w.name)
                  (wx) match {
                     case (wx:Some[Width]) => {
                        wx.get match {
                           case (v:VarWidth) => None
                           case (v) => solve(v) }}
                     case (None) => None }}
               case (w:MaxWidth) => apply_l(w.args.map(solve _),max)
               case (w:MinWidth) => apply_l(w.args.map(solve _),min)
               case (w:PlusWidth) => apply_2(solve(w.arg1),solve(w.arg2),{_ + _})
               case (w:MinusWidth) => apply_2(solve(w.arg1),solve(w.arg2),{_ - _})
               case (w:ExpWidth) => apply_2(Some(BigInt(2)),solve(w.arg1),{(x,y) => (x ^ y) - BigInt(1)})
               case (w:IntWidth) => Some(w.width)
               case (w) => println(w); error("Shouldn't be here"); None;
            }
         }
         val s = solve(w)
         (s) match {
            case (s:Some[BigInt]) => IntWidth(s.get)
            case (s) => w }
      }
   
      def reduce_var_widths_w (w:Width) : Width = {
         //println-all-debug(["REPLACE: " w])
         val wx = evaluate(w)
         //println-all-debug(["WITH: " wx])
         wx
      }
   
      val modulesx = c.modules.map{ m => {
         val portsx = m.ports.map{ p => {
            Port(p.info,p.name,p.direction,mapr(reduce_var_widths_w _,p.tpe)) }}
         (m) match {
            case (m:ExModule) => ExModule(m.info,m.name,portsx)
            case (m:InModule) => mname = m.name; InModule(m.info,m.name,portsx,mapr(reduce_var_widths_w _,m.body)) }}}
      Circuit(c.info,modulesx,c.main)
   }
   
   def run (c:Circuit): Circuit = {
      val v = ArrayBuffer[WGeq]()
      def constrain (w1:Width,w2:Width) : Unit = v += WGeq(w1,w2)
      def get_constraints_t (t1:Type,t2:Type,f:Flip) : Unit = {
         (t1,t2) match {
            case (t1:UIntType,t2:UIntType) => constrain(t1.width,t2.width)
            case (t1:SIntType,t2:SIntType) => constrain(t1.width,t2.width)
            case (t1:BundleType,t2:BundleType) => {
               (t1.fields,t2.fields).zipped.foreach{ (f1,f2) => {
                  get_constraints_t(f1.tpe,f2.tpe,times(f1.flip,f)) }}}
            case (t1:VectorType,t2:VectorType) => get_constraints_t(t1.tpe,t2.tpe,f) }}
      def get_constraints_e (e:Expression) : Expression = {
         (eMap(get_constraints_e _,e)) match {
            case (e:Mux) => {
               constrain(width_BANG(e.cond),ONE)
               constrain(ONE,width_BANG(e.cond))
               e }
            case (e) => e }}
      def get_constraints (s:Stmt) : Stmt = {
         (eMap(get_constraints_e _,s)) match {
            case (s:Connect) => {
               val n = get_size(tpe(s.loc))
               val ce_loc = create_exps(s.loc)
               val ce_exp = create_exps(s.exp)
               for (i <- 0 until n) {
                  val locx = ce_loc(i)
                  val expx = ce_exp(i)
                  get_flip(tpe(s.loc),i,DEFAULT) match {
                     case DEFAULT => constrain(width_BANG(locx),width_BANG(expx))
                     case REVERSE => constrain(width_BANG(expx),width_BANG(locx)) }}
               s }
            case (s:BulkConnect) => {
               val ls = get_valid_points(tpe(s.loc),tpe(s.exp),DEFAULT,DEFAULT)
               for (x <- ls) {
                  val locx = create_exps(s.loc)(x._1)
                  val expx = create_exps(s.exp)(x._2)
                  get_flip(tpe(s.loc),x._1,DEFAULT) match {
                     case DEFAULT => constrain(width_BANG(locx),width_BANG(expx))
                     case REVERSE => constrain(width_BANG(expx),width_BANG(locx)) }}
               s }
            case (s:DefRegister) => {
               constrain(width_BANG(s.reset),ONE)
               constrain(ONE,width_BANG(s.reset))
               get_constraints_t(s.tpe,tpe(s.init),DEFAULT)
               s }
            case (s:Conditionally) => {
               v += WGeq(width_BANG(s.pred),ONE)
               v += WGeq(ONE,width_BANG(s.pred))
               sMap(get_constraints _,s) }
            case (s) => sMap(get_constraints _,s) }}

      for (m <- c.modules) {
         (m) match {
            case (m:InModule) => mname = m.name; get_constraints(m.body)
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

object CheckWidths extends Pass with StanzaPass {
   def name = "Width Check"
   def run (c:Circuit): Circuit = stanzaPass(c, "width-check")
}

object PullMuxes extends Pass {
   private var mname = ""
   def name = "Pull Muxes"
   def run (c:Circuit): Circuit = {
      def pull_muxes_e (e:Expression) : Expression = {
         val ex = eMap(pull_muxes_e _,e) match {
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
         eMap(pull_muxes_e _,ex)
      }
      def pull_muxes (s:Stmt) : Stmt = eMap(pull_muxes_e _,sMap(pull_muxes _,s))
      val modulesx = c.modules.map {
         m => {
            mname = m.name
            m match {
               case (m:InModule) => InModule(m.info,m.name,m.ports,pull_muxes(m.body))
               case (m:ExModule) => m
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
      def expand_connects (m:InModule) : InModule = { 
         mname = m.name
         val genders = HashMap[String,Gender]()
         def expand_s (s:Stmt) : Stmt = {
            def set_gender (e:Expression) : Expression = {
               eMap(set_gender _,e) match {
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
               case (s:DefWire) => { genders += (s.name -> BIGENDER); s }
               case (s:DefRegister) => { genders += (s.name -> BIGENDER); s }
               case (s:WDefInstance) => { genders += (s.name -> MALE); s }
               case (s:DefMemory) => { genders += (s.name -> MALE); s }
               case (s:DefPoison) => { genders += (s.name -> MALE); s }
               case (s:DefNode) => { genders += (s.name -> MALE); s }
               case (s:IsInvalid) => {
                  val n = get_size(tpe(s.exp))
                  val invalids = ArrayBuffer[Stmt]()
                  val exps = create_exps(s.exp)
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
                     Empty()
                  } else if (invalids.length == 1) {
                     invalids(0)
                  } else Begin(invalids)
               }
               case (s:Connect) => {
                  val n = get_size(tpe(s.loc))
                  val connects = ArrayBuffer[Stmt]()
                  val locs = create_exps(s.loc)
                  val exps = create_exps(s.exp)
                  for (i <- 0 until n) {
                     val locx = locs(i)
                     val expx = exps(i)
                     val sx = get_flip(tpe(s.loc),i,DEFAULT) match {
                        case DEFAULT => Connect(s.info,locx,expx)
                        case REVERSE => Connect(s.info,expx,locx)
                     }
                     connects += sx
                  }
                  Begin(connects)
               }
               case (s:BulkConnect) => {
                  val ls = get_valid_points(tpe(s.loc),tpe(s.exp),DEFAULT,DEFAULT)
                  val connects = ArrayBuffer[Stmt]()
                  val locs = create_exps(s.loc)
                  val exps = create_exps(s.exp)
                  ls.foreach { x => {
                     val locx = locs(x._1)
                     val expx = exps(x._2)
                     val sx = get_flip(tpe(s.loc),x._1,DEFAULT) match {
                        case DEFAULT => Connect(s.info,locx,expx)
                        case REVERSE => Connect(s.info,expx,locx)
                     }
                     connects += sx
                  }}
                  Begin(connects)
               }
               case (s) => sMap(expand_s _,s)
            }
         }
   
         m.ports.foreach { p => genders += (p.name -> to_gender(p.direction)) }
         InModule(m.info,m.name,m.ports,expand_s(m.body))
      }
   
      val modulesx = c.modules.map { 
         m => {
            m match {
               case (m:ExModule) => m
               case (m:InModule) => expand_connects(m)
            }
         }
      }
      Circuit(c.info,modulesx,c.main)
   }
}

case class Location(base:Expression,guard:Expression)
object RemoveAccesses extends Pass {
   private var mname = ""
   def name = "Remove Accesses"
   def get_locations (e:Expression) : Seq[Location] = {
       e match {
         case (e:WRef) => create_exps(e).map(Location(_,one))
         case (e:WSubIndex) => {
            val ls = get_locations(e.exp)
            val start = get_point(e)
            val end = start + get_size(tpe(e))
            val stride = get_size(tpe(e.exp))
            val lsx = ArrayBuffer[Location]()
            var c = 0
            for (i <- 0 until ls.size) {
               if (((i % stride) >= start) & ((i % stride) < end)) {
                  lsx += ls(i)
               }
            }
            lsx
         }
         case (e:WSubField) => {
            val ls = get_locations(e.exp)
            val start = get_point(e)
            val end = start + get_size(tpe(e))
            val stride = get_size(tpe(e.exp))
            val lsx = ArrayBuffer[Location]()
            var c = 0
            for (i <- 0 until ls.size) {
               if (((i % stride) >= start) & ((i % stride) < end)) { lsx += ls(i) }
            }
            lsx
         }
         case (e:WSubAccess) => {
            val ls = get_locations(e.exp)
            val stride = get_size(tpe(e))
            val wrap = tpe(e.exp).asInstanceOf[VectorType].size
            val lsx = ArrayBuffer[Location]()
            var c = 0
            for (i <- 0 until ls.size) {
               if ((c % wrap) == 0) { c = 0 }
               val basex = ls(i).base
               val guardx = AND(ls(i).guard,EQV(uint(c),e.index))
               lsx += Location(basex,guardx)
               if ((i + 1) % stride == 0) {
                  c = c + 1
               }
            }
            lsx
         }
      }
   }
   def has_access (e:Expression) : Boolean = {
      var ret:Boolean = false
      def rec_has_access (e:Expression) : Expression = {
         e match {
            case (e:WSubAccess) => { ret = true; e }
            case (e) => eMap(rec_has_access _,e)
         }
      }
      rec_has_access(e)
      ret
   }
   def run (c:Circuit): Circuit = {
      def remove_m (m:InModule) : InModule = {
         val sh = sym_hash
         mname = m.name
         def remove_s (s:Stmt) : Stmt = {
            val stmts = ArrayBuffer[Stmt]()
            def create_temp (e:Expression) : Expression = {
               val n = firrtl_gensym("GEN",sh)
               stmts += DefWire(info(s),n,tpe(e))
               WRef(n,tpe(e),kind(e),gender(e))
            }
            def remove_e (e:Expression) : Expression = { //NOT RECURSIVE (except primops) INTENTIONALLY!
               e match {
                  case (e:DoPrim) => eMap(remove_e,e)
                  case (e:Mux) => eMap(remove_e,e)
                  case (e:ValidIf) => eMap(remove_e,e)
                  case (e:SIntValue) => e
                  case (e:UIntValue) => e
                  case e => {
                     if (has_access(e)) {
                        val rs = get_locations(e)
                        val foo = rs.find(x => {x.guard != one})
                        foo match {
                           case None => error("Shouldn't be here")
                           case foo:Some[Location] => {
                              val temp = create_temp(e)
                              val temps = create_exps(temp)
                              def get_temp (i:Int) = temps(i % temps.size)
                              (rs,0 until rs.size).zipped.foreach {
                                 (x,i) => {
                                    if (i < temps.size) {
                                       stmts += Connect(info(s),get_temp(i),x.base)
                                    } else {
                                       stmts += Conditionally(info(s),x.guard,Connect(info(s),get_temp(i),x.base),Empty())
                                    }
                                 }
                              }
                              temp
                           }
                        }
                     } else { e}
                  }
               }
            }

            val sx = s match {
               case (s:Connect) => {
                  if (has_access(s.loc)) {
                     val ls = get_locations(s.loc)
                     val locx = 
                        if (ls.size == 1 & ls(0).guard == one) s.loc
                        else {
                           val temp = create_temp(s.loc)
                           for (x <- ls) { stmts += Conditionally(s.info,x.guard,Connect(s.info,x.base,temp),Empty()) }
                           temp
                        }
                     Connect(s.info,locx,remove_e(s.exp))
                  } else { Connect(s.info,s.loc,remove_e(s.exp)) }
               }
               case (s) => sMap(remove_s,eMap(remove_e,s))
            }
            stmts += sx
            if (stmts.size != 1) Begin(stmts) else stmts(0)
         }
         InModule(m.info,m.name,m.ports,remove_s(m.body))
      }
   
      val modulesx = c.modules.map{
         m => {
            m match {
               case (m:ExModule) => m
               case (m:InModule) => remove_m(m)
            }
         }
      }
      Circuit(c.info,modulesx,c.main)
   }
}

object ExpandWhens extends Pass {
   def name = "Expand Whens"
   var mname = ""
// ; ========== Expand When Utilz ==========
   def add (hash:HashMap[WrappedExpression,Expression],key:WrappedExpression,value:Expression) = {
      hash += (key -> value)
   }

   def get_entries (hash:HashMap[WrappedExpression,Expression],exps:Seq[Expression]) : HashMap[WrappedExpression,Expression] = {
      val hashx = HashMap[WrappedExpression,Expression]()
      exps.foreach { e => {
         val value = hash.get(e)
         value match {
            case (value:Some[Expression]) => add(hashx,e,value.get)
            case (None) => {}
         }
      }}
      hashx
   }
   def get_female_refs (n:String,t:Type,g:Gender) : Seq[Expression] = {
      val exps = create_exps(WRef(n,t,ExpKind(),g))
      val expsx = ArrayBuffer[Expression]()
      def get_gender (t:Type, i:Int, g:Gender) : Gender = {
         val f = get_flip(t,i,DEFAULT)
         times(g, f)
      }
      for (i <- 0 until exps.size) {
         get_gender(t,i,g) match {
            case BIGENDER => expsx += exps(i)
            case FEMALE => expsx += exps(i)
            case _ => false
         }
      }
      expsx
   }
   
   // ------------ Pass -------------------
   def run (c:Circuit): Circuit = {
      def void_all (m:InModule) : InModule = {
         mname = m.name
         def void_all_s (s:Stmt) : Stmt = {
            (s) match {
               case (_:DefWire|_:DefRegister|_:WDefInstance|_:DefMemory) => {
                  val voids = ArrayBuffer[Stmt]()
                  for (e <- get_female_refs(get_name(s),get_type(s),get_gender(s))) {
                     voids += Connect(get_info(s),e,WVoid())
                  }
                  Begin(Seq(s,Begin(voids)))
               }
               case (s) => sMap(void_all_s _,s)
            }
         }
         val voids = ArrayBuffer[Stmt]()
         for (p <- m.ports) {
            for (e <- get_female_refs(p.name,p.tpe,get_gender(p))) {
               voids += Connect(p.info,e,WVoid())
            }
         }
         val bodyx = void_all_s(m.body)
         voids += bodyx
         InModule(m.info,m.name,m.ports,Begin(voids))
      }
      def expand_whens (m:InModule) : Tuple2[HashMap[WrappedExpression,Expression],ArrayBuffer[Stmt]] = {
         val simlist = ArrayBuffer[Stmt]()
         mname = m.name
         def expand_whens (netlist:HashMap[WrappedExpression,Expression],p:Expression)(s:Stmt) : Stmt = {
            (s) match {
               case (s:Connect) => netlist(s.loc) = s.exp
               case (s:IsInvalid) => netlist(s.exp) = WInvalid()
               case (s:Conditionally) => {
                  val exps = ArrayBuffer[Expression]()
                  def prefetch (s:Stmt) : Stmt = {
                     (s) match {
                        case (s:Connect) => exps += s.loc; s
                        case (s) => sMap(prefetch _,s)
                     }
                  }
                  prefetch(s.conseq)
                  val c_netlist = get_entries(netlist,exps)
                  expand_whens(c_netlist,AND(p,s.pred))(s.conseq)
                  expand_whens(netlist,AND(p,NOT(s.pred)))(s.alt)
                  for (lvalue <- c_netlist.keys) {
                     val value = netlist.get(lvalue)
                     (value) match {
                        case (value:Some[Expression]) => {
                           val tv = c_netlist(lvalue)
                           val fv = value.get
                           val res = (tv,fv) match {
                              case (tv:WInvalid,fv:WInvalid) => WInvalid()
                              case (tv:WInvalid,fv) => ValidIf(NOT(s.pred),fv,tpe(fv))
                              case (tv,fv:WInvalid) => ValidIf(s.pred,tv,tpe(tv))
                              case (tv,fv) => Mux(s.pred,tv,fv,mux_type_and_widths(tv,fv))
                           }
                           netlist(lvalue) = res
                        }
                        case (None) => add(netlist,lvalue,c_netlist(lvalue))
                     }
                  }
               }
               case (s:Print) => {
                  if (p == one) {
                     simlist += s
                  } else {
                     simlist += Print(s.info,s.string,s.args,s.clk,AND(p,s.en))
                  }
               }
               case (s:Stop) => {
                  if (p == one) {
                     simlist += s
                  } else {
                     simlist += Stop(s.info,s.ret,s.clk,AND(p,s.en))
                  }
               }
               case (s) => sMap(expand_whens(netlist,p) _, s)
            }
            s
         }
         val netlist = HashMap[WrappedExpression,Expression]()
         expand_whens(netlist,one)(m.body)
   
         //println("Netlist:")
         //println(netlist)
         //println("Simlist:")
         //println(simlist)
         ( netlist, simlist )
      }
   
      def create_module (netlist:HashMap[WrappedExpression,Expression],simlist:ArrayBuffer[Stmt],m:InModule) : InModule = {
         mname = m.name
         val stmts = ArrayBuffer[Stmt]()
         val connections = ArrayBuffer[Stmt]()
         def replace_void (e:Expression)(rvalue:Expression) : Expression = {
            (rvalue) match {
               case (rv:WVoid) => e
               case (rv) => eMap(replace_void(e) _,rv)
            }
         }
         def create (s:Stmt) : Stmt = {
            (s) match {
               case (_:DefWire|_:WDefInstance|_:DefMemory) => {
                  stmts += s
                  for (e <- get_female_refs(get_name(s),get_type(s),get_gender(s))) {
                     val rvalue = netlist(e)
                     val con = (rvalue) match {
                        case (rvalue:WInvalid) => IsInvalid(get_info(s),e)
                        case (rvalue) => Connect(get_info(s),e,rvalue)
                     }
                     connections += con
                  }
               }
               case (s:DefRegister) => {
                  stmts += s
                  for (e <- get_female_refs(get_name(s),get_type(s),get_gender(s))) {
                     val rvalue = replace_void(e)(netlist(e))
                     val con = (rvalue) match {
                        case (rvalue:WInvalid) => IsInvalid(get_info(s),e)
                        case (rvalue) => Connect(get_info(s),e,rvalue)
                     }
                     connections += con
                  }
               }
               case (_:DefPoison|_:DefNode) => stmts += s
               case (s) => sMap(create _,s)
            }
            s
         }
         create(m.body)
         for (p <- m.ports) {
            for (e <- get_female_refs(p.name,p.tpe,get_gender(p))) {
               val rvalue = netlist(e)
               val con = (rvalue) match {
                  case (rvalue:WInvalid) => IsInvalid(p.info,e)
                  case (rvalue) => Connect(p.info,e,rvalue)
               }
               connections += con
            }
         }
         for (x <- simlist) { stmts += x }
         InModule(m.info,m.name,m.ports,Begin(Seq(Begin(stmts),Begin(connections))))
      }
   
      val voided_modules = c.modules.map{ m => {
            (m) match {
               case (m:ExModule) => m
               case (m:InModule) => void_all(m)
            } } }
      val modulesx = voided_modules.map{ m => {
            (m) match {
               case (m:ExModule) => m
               case (m:InModule) => {
                  val (netlist, simlist) = expand_whens(m)
                  create_module(netlist,simlist,m)
               }
            }}}
      Circuit(c.info,modulesx,c.main)
   }
}

object CheckInitialization extends Pass with StanzaPass {
   def name = "Check Initialization"
   def run (c:Circuit): Circuit = stanzaPass(c, "check-init")
}

object ConstProp extends Pass with StanzaPass {
   def name = "Constant Propogation"
   var mname = ""
   def run (c:Circuit): Circuit = stanzaPass(c, "const-prop")
   def const_prop_e (e:Expression) : Expression = {
      eMap(const_prop_e _,e) match {
         case (e:DoPrim) => {
            e.op match {
               case SHIFT_RIGHT_OP => {
                  (e.args(0)) match {
                     case (x:UIntValue) => {
                        val b = x.value >> e.consts(0).toInt
                        UIntValue(b,tpe(e).as[UIntType].get.width)
                     }
                     case (x:SIntValue) => {
                        val b = x.value >> e.consts(0).toInt
                        SIntValue(b,tpe(e).as[SIntType].get.width)
                     }
                     case (x) => e
                  }
               }
               case BITS_SELECT_OP => {
                  e.args(0) match {
                     case (x:UIntValue) => {
                        val hi = e.consts(0).toInt
                        val lo = e.consts(1).toInt
                        require(hi >= lo)
                        val b = (x.value >> lo) & ((BigInt(1) << (hi - lo + 1)) - 1)
                        UIntValue(b,tpe(e).as[UIntType].get.width)
                     }
                     case (x) => {
                        if (long_BANG(tpe(e)) == long_BANG(tpe(x))) {
                           if (tpe(x).typeof[UIntType] != None) x
                           else DoPrim(AS_UINT_OP,Seq(x),Seq(),tpe(e))
                        }
                        else e
                     }
                  }
               }
               case (_) => e
            }
         }
         case (e) => e
      }
   }
   def const_prop_s (s:Stmt) : Stmt = eMap(const_prop_e _, sMap(const_prop_s _,s))
   def const_prop (c:Circuit) : Circuit = {
      val modulesx = c.modules.map{ m => {
         m match {
            case (m:ExModule) => m
            case (m:InModule) => {
               mname = m.name
               InModule(m.info,m.name,m.ports,const_prop_s(m.body))
            }
         }
      }}
      Circuit(c.info,modulesx,c.main)
   }
}

object LoToVerilog extends Pass with StanzaPass {
   def name = "Lo To Verilog"
   def run (c:Circuit): Circuit = stanzaPass(c, "lo-to-verilog")
}

object FromCHIRRTL extends Pass with StanzaPass {
   def name = "From CHIRRTL"
   def run (c:Circuit): Circuit = stanzaPass(c, "from-chirrtl")
}

object VerilogWrap extends Pass {
   def name = "Verilog Wrap"
   var mname = ""
   def v_wrap_e (e:Expression) : Expression = {
      eMap(v_wrap_e _,e) match {
         case (e:DoPrim) => {
            def a0 () = e.args(0)
            if (e.op == TAIL_OP) {
               (a0()) match {
                  case (e0:DoPrim) => {
                     if (e0.op == ADD_OP) DoPrim(ADDW_OP,e0.args,Seq(),tpe(e))
                     else if (e0.op == SUB_OP) DoPrim(SUBW_OP,e0.args,Seq(),tpe(e))
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
   def v_wrap_s (s:Stmt) : Stmt = eMap(v_wrap_e _,sMap(v_wrap_s _,s))
   def run (c:Circuit): Circuit = {
      val modulesx = c.modules.map{ m => {
         (m) match {
            case (m:InModule) => {
               mname = m.name
               InModule(m.info,m.name,m.ports,v_wrap_s(m.body))
            }
            case (m:ExModule) => m
         }
      }}
      Circuit(c.info,modulesx,c.main)
   }
}

object SplitExp extends Pass {
   def name = "Split Expressions"
   var mname = ""
   def split_exp (m:InModule) : InModule = {
      mname = m.name
      val v = ArrayBuffer[Stmt]()
      val sh = sym_hash
      def split_exp_s (s:Stmt) : Stmt = {
         def split (e:Expression) : Expression = {
            val n = firrtl_gensym("GEN",sh)
            v += DefNode(info(s),n,e)
            WRef(n,tpe(e),kind(e),gender(e))
         }
         def split_exp_e (i:Int)(e:Expression) : Expression = {
            eMap(split_exp_e(i + 1) _,e) match {
               case (e:DoPrim) => if (i > 0) split(e) else e
               case (e) => e
            }
         }
         eMap(split_exp_e(0) _,s) match {
            case (s:Begin) => sMap(split_exp_s _,s)
            case (s) => v += s; s
         }
      }
      split_exp_s(m.body)
      InModule(m.info,m.name,m.ports,Begin(v))
   }
   
   def run (c:Circuit): Circuit = {
      val modulesx = c.modules.map{ m => {
         (m) match {
            case (m:InModule) => split_exp(m)
            case (m:ExModule) => m
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
           case (e) => eMap(verilog_rename_e,e)
         }
      }
      def verilog_rename_s (s:Stmt) : Stmt = {
         stMap(verilog_rename_n _,eMap(verilog_rename_e _,sMap(verilog_rename_s _,s)))
      }
      val modulesx = c.modules.map{ m => {
         val portsx = m.ports.map{ p => {
            Port(p.info,verilog_rename_n(p.name),p.direction,p.tpe)
         }}
         m match {
            case (m:InModule) => InModule(m.info,m.name,portsx,verilog_rename_s(m.body))
            case (m:ExModule) => m
         }
      }}
      Circuit(c.info,modulesx,c.main)
   }
}

object LowerTypes extends Pass {
   def name = "Lower Types"
   var mname = ""
   def is_ground (t:Type) : Boolean = {
      (t) match {
         case (_:UIntType|_:SIntType) => true
         case (t) => false
      }
   }
   def data (ex:Expression) : Boolean = {
      (kind(ex)) match {
         case (k:MemKind) => (ex) match {
            case (_:WRef|_:WSubIndex) => false
            case (ex:WSubField) => {
               var yes = ex.name match {
                  case "rdata" => true
                  case "data" => true
                  case "mask" => true
                  case _ => false
               }
               yes && ((ex.exp) match {
                  case (e:WSubField) => kind(e).as[MemKind].get.ports.contains(e.name) && (e.exp.typeof[WRef])
                  case (e) => false
               })
            }
            case (ex) => false
         }
         case (k) => false
      }
   }
   def expand_name (e:Expression) : Seq[String] = {
      val names = ArrayBuffer[String]()
      def expand_name_e (e:Expression) : Expression = {
         (eMap(expand_name_e _,e)) match {
            case (e:WRef) => names += e.name
            case (e:WSubField) => names += e.name
            case (e:WSubIndex) => names += e.value.toString
         }
         e
      }
      expand_name_e(e)
      names
   }
   def lower_other_mem (e:Expression, dt:Type) : Seq[Expression] = {
      val names = expand_name(e)
      if (names.size < 3) error("Shouldn't be here")
      create_exps(names(0),dt).map{ x => {
         var base = lowered_name(x)
         for (i <- 0 until names.size) {
            if (i >= 3) base = base + "_" + names(i)
         }
         val m = WRef(base, UnknownType(), kind(e), UNKNOWNGENDER)
         val p = WSubField(m,names(1),UnknownType(),UNKNOWNGENDER)
         WSubField(p,names(2),UnknownType(),UNKNOWNGENDER)
      }}
   }
   def lower_data_mem (e:Expression) : Expression = {
      val names = expand_name(e)
      if (names.size < 3) error("Shouldn't be here")
      else {
         var base = names(0)
         for (i <- 0 until names.size) {
            if (i >= 3) base = base + "_" + names(i)
         }
         val m = WRef(base, UnknownType(), kind(e), UNKNOWNGENDER)
         val p = WSubField(m,names(1),UnknownType(),UNKNOWNGENDER)
         WSubField(p,names(2),UnknownType(),UNKNOWNGENDER)
      }
   }
   def merge (a:String,b:String,x:String) : String = a + x + b
   def root_ref (e:Expression) : WRef = {
      (e) match {
         case (e:WRef) => e
         case (e:WSubField) => root_ref(e.exp)
         case (e:WSubIndex) => root_ref(e.exp)
         case (e:WSubAccess) => root_ref(e.exp)
      }
   }
   
   //;------------- Pass ------------------
   
   def lower_types (m:Module) : Module = {
      val mdt = HashMap[String,Type]()
      mname = m.name
      def lower_types (s:Stmt) : Stmt = {
         def lower_mem (e:Expression) : Seq[Expression] = {
            val names = expand_name(e)
            if (Seq("data","mask","rdata").contains(names(2))) Seq(lower_data_mem(e))
            else lower_other_mem(e,mdt(root_ref(e).name))
         }
         def lower_types_e (e:Expression) : Expression = {
            e match {
               case (_:WRef|_:UIntValue|_:SIntValue) => e
               case (_:WSubField|_:WSubIndex) => {
                  (kind(e)) match {
                     case (k:InstanceKind) => {
                        val names = expand_name(e)
                        var n = names(1)
                        for (i <- 0 until names.size) {
                           if (i > 1) n = n + "_" + names(i)
                        }
                        WSubField(root_ref(e),n,tpe(e),gender(e))
                     }
                     case (k:MemKind) => {
                        if (gender(e) != FEMALE) lower_mem(e)(0)
                        else e
                     }
                     case (k) => WRef(lowered_name(e),tpe(e),kind(e),gender(e))
                  }
               }
               case (e:DoPrim) => eMap(lower_types_e _,e)
               case (e:Mux) => eMap(lower_types_e _,e)
               case (e:ValidIf) => eMap(lower_types_e _,e)
            }
         }
         (s) match {
            case (s:DefWire) => {
               if (is_ground(s.tpe)) s else {
                  val es = create_exps(s.name,s.tpe)
                  val stmts = (es, 0 until es.size).zipped.map{ (e,i) => {
                     DefWire(s.info,lowered_name(e),tpe(e))
                  }}
                  Begin(stmts)
               }
            }
            case (s:DefPoison) => {
               if (is_ground(s.tpe)) s else {
                  val es = create_exps(s.name,s.tpe)
                  val stmts = (es, 0 until es.size).zipped.map{ (e,i) => {
                     DefPoison(s.info,lowered_name(e),tpe(e))
                  }}
                  Begin(stmts)
               }
            }
            case (s:DefRegister) => {
               if (is_ground(s.tpe)) s else {
                  val es = create_exps(s.name,s.tpe)
                  val inits = create_exps(s.init) 
                  val stmts = (es, 0 until es.size).zipped.map{ (e,i) => {
                     val init = lower_types_e(inits(i))
                     DefRegister(s.info,lowered_name(e),tpe(e),s.clock,s.reset,init)
                  }}
                  Begin(stmts)
               }
            }
            case (s:WDefInstance) => {
               val fieldsx = s.tpe.as[BundleType].get.fields.flatMap{ f => {
                  val es = create_exps(WRef(f.name,f.tpe,ExpKind(),times(f.flip,MALE)))
                  es.map{ e => {
                     gender(e) match {
                        case MALE => Field(lowered_name(e),DEFAULT,f.tpe)
                        case FEMALE => Field(lowered_name(e),REVERSE,f.tpe)
                     }
                  }}
               }}
               WDefInstance(s.info,s.name,s.module,BundleType(fieldsx))
            }
            case (s:DefMemory) => {
               mdt(s.name) = s.data_type
               if (is_ground(s.data_type)) s else {
                  val es = create_exps(s.name,s.data_type)
                  val stmts = es.map{ e => {
                     DefMemory(s.info,lowered_name(e),tpe(e),s.depth,s.write_latency,s.read_latency,s.readers,s.writers,s.readwriters)
                  }}
                  Begin(stmts)
               }
            }
            case (s:IsInvalid) => {
               val sx = eMap(lower_types_e _,s).as[IsInvalid].get
               kind(sx.exp) match {
                  case (k:MemKind) => {
                     val es = lower_mem(sx.exp)
                     Begin(es.map(e => {IsInvalid(sx.info,e)}))
                  }
                  case (_) => sx
               }
            }
            case (s:Connect) => {
               val sx = eMap(lower_types_e _,s).as[Connect].get
               kind(sx.loc) match {
                  case (k:MemKind) => {
                     val es = lower_mem(sx.loc)
                     Begin(es.map(e => {Connect(sx.info,e,sx.exp)}))
                  }
                  case (_) => sx
               }
            }
            case (s:DefNode) => {
               val locs = create_exps(s.name,tpe(s.value))
               val n = locs.size
               val nodes = ArrayBuffer[Stmt]()
               val exps = create_exps(s.value)
               for (i <- 0 until n) {
                  val locx = locs(i)
                  val expx = exps(i)
                  nodes += DefNode(s.info,lowered_name(locx),lower_types_e(expx))
               }
               if (n == 1) nodes(0) else Begin(nodes)
            }
            case (s) => eMap(lower_types_e _,sMap(lower_types _,s))
         }
      }
   
      val portsx = m.ports.flatMap{ p => {
         val es = create_exps(WRef(p.name,p.tpe,PortKind(),to_gender(p.direction)))
         es.map(e => { Port(p.info,lowered_name(e),to_dir(gender(e)),tpe(e)) })
      }}
      (m) match {
         case (m:ExModule) => ExModule(m.info,m.name,portsx)
         case (m:InModule) => InModule(m.info,m.name,portsx,lower_types(m.body))
      }
   }
   
   def run (c:Circuit) : Circuit = {
      val modulesx = c.modules.map(m => lower_types(m))
      Circuit(c.info,modulesx,c.main)
   }
}

