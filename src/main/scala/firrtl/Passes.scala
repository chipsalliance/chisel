
package firrtl

import com.typesafe.scalalogging.LazyLogging
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer

import Utils._
import DebugUtils._
import PrimOps._

object Passes extends LazyLogging {

   // TODO Perhaps we should get rid of Logger since this map would be nice
   ////private val defaultLogger = Logger()
   //private def mapNameToPass = Map[String, Circuit => Circuit] (
   //  "infer-types" -> inferTypes
   //)
   var mname = ""
   def nameToPass(name: String): Circuit => Circuit = {
     //mapNameToPass.getOrElse(name, throw new Exception("No Standard FIRRTL Pass of name " + name))
     name match {
       case "to-working-ir" => toWorkingIr
       //case "infer-types" => inferTypes
         // errrrrrrrrrr...
       //case "renameall" => renameall(Map())
     }
   }
 
   private def toField(p: Port): Field = {
     logger.debug(s"toField called on port ${p.serialize}")
     p.direction match {
       case INPUT  => Field(p.name, REVERSE, p.tpe)
       case OUTPUT => Field(p.name, DEFAULT, p.tpe)
     }
   }
   // ============== RESOLVE ALL ===================
   def resolve (c:Circuit) = {
      val passes = Seq(
         toWorkingIr _,
         resolveKinds _,
         inferTypes _,
         resolveGenders _,
         pullMuxes _,
         expandConnects _)
      val names = Seq(
         "To Working IR",
         "Resolve Kinds",
         "Infer Types",
         "Resolve Genders",
         "Pull Muxes",
         "Expand Connects")
      var c_BANG = c
      (names, passes).zipped.foreach { 
         (n,p) => {
            println("Starting " + n)
            c_BANG = p(c_BANG)
            println(c_BANG.serialize())
            println("Finished " + n)
         }
      }
      c_BANG
   }
 
  // ============== TO WORKING IR ==================
  def toWorkingIr (c:Circuit) = {
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
  // ===============================================

  // ============== RESOLVE KINDS ==================
   def resolveKinds (c:Circuit) = {
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
  // ===============================================

  // ============== INFER TYPES ==================

  // ------------------ Utils -------------------------

   val width_name_hash = Map[String,Int]()
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
   

   
   // ------------------ Pass -------------------------
   
   def inferTypes (c:Circuit) : Circuit = {
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

// =================== RESOLVE GENDERS =======================
   def resolveGenders (c:Circuit) = {
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
  // ===============================================

  // =============== PULL MUXES ====================
   def pullMuxes (c:Circuit) : Circuit = {
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
  // ===============================================



   // ============ EXPAND CONNECTS ==================
   // ---------------- UTILS ------------------
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
   
   def create_exps (n:String, t:Type) : Seq[Expression] =
      create_exps(WRef(n,t,ExpKind(),UNKNOWNGENDER))
   def create_exps (e:Expression) : Seq[Expression] = {
      e match {
         case (e:Mux) => {
            val e1s = create_exps(e.tval)
            val e2s = create_exps(e.fval)
            (e1s, e2s).zipped.map { 
               (e1,e2) => Mux(e.cond,e1,e2,mux_type_and_widths(e1,e2))
            }
         }
         case (e:ValidIf) => {
            create_exps(e.value).map {
               e1 => ValidIf(e.cond,e1,tpe(e1))
            }
         }
         case (e) => {
            tpe(e) match {
               case (t:UIntType) => Seq(e)
               case (t:SIntType) => Seq(e)
               case (t:ClockType) => Seq(e)
               case (t:BundleType) => {
                  t.fields.flatMap {
                     f => create_exps(WSubField(e,f.name,f.tpe,times(gender(e), f.flip)))
                  }
               }
               case (t:VectorType) => {
                  (0 until t.size).flatMap {
                     i => create_exps(WSubIndex(e,i,t.tpe,gender(e)))
                  }
               }
            }
         }
      }
   }
   
   //---------------- Pass ---------------------
   
   def expandConnects (c:Circuit) : Circuit = { 
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

  /** INFER TYPES
   *
   *  This pass infers the type field in all IR nodes by updating
   *    and passing an environment to all statements in pre-order
   *    traversal, and resolving types in expressions in post-
   *    order traversal.
   *  Type propagation for primary ops are defined in Primops.scala.
   *  Type errors are not checked in this pass, as this is
   *    postponed for a later/earlier pass.
   */
  // input -> flip
  //private type TypeMap = Map[String, Type]
  //private val TypeMap = Map[String, Type]().withDefaultValue(UnknownType)
  //private def getBundleSubtype(t: Type, name: String): Type = {
  //  t match {
  //    case b: BundleType => {
  //      val tpe = b.fields.find( _.name == name )
  //      if (tpe.isEmpty) UnknownType
  //      else tpe.get.tpe
  //    }
  //    case _ => UnknownType
  //  }
  //}
  //private def getVectorSubtype(t: Type): Type = t.getType // Added for clarity
  //// TODO Add genders
  //private def inferExpTypes(typeMap: TypeMap)(exp: Expression): Expression = {
  //  logger.debug(s"inferTypes called on ${exp.getClass.getSimpleName}")
  //  exp.map(inferExpTypes(typeMap)) match {
  //    case e: UIntValue => e
  //    case e: SIntValue => e
  //    case e: Ref => Ref(e.name, typeMap(e.name))
  //    case e: SubField => SubField(e.exp, e.name, getBundleSubtype(e.exp.getType, e.name))
  //    case e: SubIndex => SubIndex(e.exp, e.value, getVectorSubtype(e.exp.getType))
  //    case e: SubAccess => SubAccess(e.exp, e.index, getVectorSubtype(e.exp.getType))
  //    case e: DoPrim => lowerAndTypePrimOp(e)
  //    case e: Expression => e
  //  }
  //}
  //private def inferTypes(typeMap: TypeMap, stmt: Stmt): (Stmt, TypeMap) = {
  //  logger.debug(s"inferTypes called on ${stmt.getClass.getSimpleName} ")
  //  stmt.map(inferExpTypes(typeMap)) match {
  //    case b: Begin => {
  //      var tMap = typeMap
  //      // TODO FIXME is map correctly called in sequential order
  //      val body = b.stmts.map { s =>
  //        val ret = inferTypes(tMap, s)
  //        tMap = ret._2
  //        ret._1
  //      }
  //      (Begin(body), tMap)
  //    }
  //    case s: DefWire => (s, typeMap ++ Map(s.name -> s.tpe))
  //    case s: DefRegister => (s, typeMap ++ Map(s.name -> s.tpe))
  //    case s: DefMemory => (s, typeMap ++ Map(s.name -> s.dataType))
  //    case s: DefInstance => (s, typeMap ++ Map(s.name -> typeMap(s.module)))
  //    case s: DefNode => (s, typeMap ++ Map(s.name -> s.value.getType))
  //    case s: Conditionally => { // TODO Check: Assuming else block won't see when scope
  //      val (conseq, cMap) = inferTypes(typeMap, s.conseq)
  //      val (alt, aMap) = inferTypes(typeMap, s.alt)
  //      (Conditionally(s.info, s.pred, conseq, alt), cMap ++ aMap)
  //    }
  //    case s: Stmt => (s, typeMap)
  //  }
  //}
  //private def inferTypes(typeMap: TypeMap, m: Module): Module = {
  //  logger.debug(s"inferTypes called on module ${m.name}")

  //  val pTypeMap = m.ports.map( p => p.name -> p.tpe ).toMap

  //  Module(m.info, m.name, m.ports, inferTypes(typeMap ++ pTypeMap, m.stmt)._1)
  //}
  //def inferTypes(c: Circuit): Circuit = {
  //  logger.debug(s"inferTypes called on circuit ${c.name}")

  //  // initialize typeMap with each module of circuit mapped to their bundled IO (ports converted to fields)
  //  val typeMap = c.modules.map(m => m.name -> BundleType(m.ports.map(toField(_)))).toMap

  //  //val typeMap = c.modules.flatMap(buildTypeMap).toMap
  //  Circuit(c.info, c.name, c.modules.map(inferTypes(typeMap, _)))
  //}

  //def renameall(s : String)(implicit map : Map[String,String]) : String =
  //  map getOrElse (s, s)

  //def renameall(e : Expression)(implicit map : Map[String,String]) : Expression = {
  //  logger.debug(s"renameall called on expression ${e.toString}")
  //  e match {
  //    case p : Ref =>
  //      Ref(renameall(p.name), p.tpe)
  //    case p : SubField =>
  //      SubField(renameall(p.exp), renameall(p.name), p.tpe)
  //    case p : SubIndex =>
  //      SubIndex(renameall(p.exp), p.value, p.tpe)
  //    case p : SubAccess =>
  //      SubAccess(renameall(p.exp), renameall(p.index), p.tpe)
  //    case p : Mux =>
  //      Mux(renameall(p.cond), renameall(p.tval), renameall(p.fval), p.tpe)
  //    case p : ValidIf =>
  //      ValidIf(renameall(p.cond), renameall(p.value), p.tpe)
  //    case p : DoPrim =>
  //      println( p.args.map(x => renameall(x)) )
  //      DoPrim(p.op, p.args.map(renameall), p.consts, p.tpe)
  //    case p : Expression => p
  //  }
  //}

  //def renameall(s : Stmt)(implicit map : Map[String,String]) : Stmt = {
  //  logger.debug(s"renameall called on statement ${s.toString}")

  //  s match {
  //    case p : DefWire =>
  //      DefWire(p.info, renameall(p.name), p.tpe)
  //    case p: DefRegister =>
  //      DefRegister(p.info, renameall(p.name), p.tpe, p.clock, p.reset, p.init)
  //    case p : DefMemory =>
  //      DefMemory(p.info, renameall(p.name), p.dataType, p.depth, p.writeLatency, p.readLatency, 
  //                p.readers, p.writers, p.readwriters)
  //    case p : DefInstance =>
  //      DefInstance(p.info, renameall(p.name), renameall(p.module))
  //    case p : DefNode =>
  //      DefNode(p.info, renameall(p.name), renameall(p.value))
  //    case p : Connect =>
  //      Connect(p.info, renameall(p.loc), renameall(p.exp))
  //    case p : BulkConnect =>
  //      BulkConnect(p.info, renameall(p.loc), renameall(p.exp))
  //    case p : IsInvalid =>
  //      IsInvalid(p.info, renameall(p.exp))
  //    case p : Stop =>
  //      Stop(p.info, p.ret, renameall(p.clk), renameall(p.en))
  //    case p : Print =>
  //      Print(p.info, p.string, p.args.map(renameall), renameall(p.clk), renameall(p.en))
  //    case p : Conditionally =>
  //      Conditionally(p.info, renameall(p.pred), renameall(p.conseq), renameall(p.alt))
  //    case p : Begin =>
  //      Begin(p.stmts.map(renameall))
  //    case p : Stmt => p
  //  }
  //}

  //def renameall(p : Port)(implicit map : Map[String,String]) : Port = {
  //  logger.debug(s"renameall called on port ${p.name}")
  //  Port(p.info, renameall(p.name), p.dir, p.tpe)
  //}

  //def renameall(m : Module)(implicit map : Map[String,String]) : Module = {
  //  logger.debug(s"renameall called on module ${m.name}")
  //  Module(m.info, renameall(m.name), m.ports.map(renameall(_)), renameall(m.stmt))
  //}

  //def renameall(map : Map[String,String]) : Circuit => Circuit = {
  //  c => {
  //    implicit val imap = map
  //    logger.debug(s"renameall called on circuit ${c.name} with %{renameto}")
  //    Circuit(c.info, renameall(c.name), c.modules.map(renameall(_)))
  //  }
  //}
}
