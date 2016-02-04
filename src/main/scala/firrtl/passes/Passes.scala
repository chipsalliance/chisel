
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
  val listOfPasses: Seq[Pass] = Seq(ToWorkingIR,ResolveKinds,ResolveGenders,PullMuxes,ExpandConnects,RemoveAccesses,ExpandWhens)
  lazy val mapNameToPass: Map[String, Pass] = listOfPasses.map(p => p.name -> p).toMap

  def executePasses(c: Circuit, passes: Seq[Pass]): Circuit = { 
    if (passes.isEmpty) c
    else {
       val p = passes.head
       val name = p.name
       logger.debug(c.serialize())
       logger.debug(s"Starting ${name}")
       executePasses(p.run(c), passes.tail)
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
   def mapr (f: Width => Width, t:Type) : Type = {
      def apply_t (t:Type) : Type = {
         wMap(f,tMap(apply_t _,t))
      }
      apply_t(t)
   }
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

object InferWidths extends Pass with StanzaPass {
   def name = "Infer Widths"
   def run (c:Circuit): Circuit = stanzaPass(c, "infer-widths")
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

object ExpandWhens extends Pass with StanzaPass {
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
  def run (c:Circuit): Circuit = stanzaPass(c, "const-prop")
}

object LoToVerilog extends Pass with StanzaPass {
  def name = "Lo To Verilog"
  def run (c:Circuit): Circuit = stanzaPass(c, "lo-to-verilog")
}

object VerilogWrap extends Pass with StanzaPass {
  def name = "Verilog Wrap"
  def run (c:Circuit): Circuit = stanzaPass(c, "verilog-wrap")
}

object SplitExp extends Pass with StanzaPass {
  def name = "Split Expressions"
  def run (c:Circuit): Circuit = stanzaPass(c, "split-expressions")
}

object VerilogRename extends Pass with StanzaPass {
  def name = "Verilog Rename"
  def run (c:Circuit): Circuit = stanzaPass(c, "verilog-rename")
}

object LowerTypes extends Pass with StanzaPass {
  def name = "Lower Types"
  def run (c:Circuit): Circuit = stanzaPass(c, "lower-types")
}

