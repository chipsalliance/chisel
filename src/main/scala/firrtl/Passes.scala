
package firrtl

import com.typesafe.scalalogging.LazyLogging
import scala.collection.mutable.HashMap

import Utils._
import DebugUtils._
import PrimOps._

object Passes extends LazyLogging {

   // TODO Perhaps we should get rid of Logger since this map would be nice
   ////private val defaultLogger = Logger()
   //private def mapNameToPass = Map[String, Circuit => Circuit] (
   //  "infer-types" -> inferTypes
   //)
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
       case Input  => Field(p.name, REVERSE, p.tpe)
       case Output => Field(p.name, DEFAULT, p.tpe)
     }
   }
   // ============== RESOLVE ALL ===================
   def resolve (c:Circuit) = {
      val passes = Seq(
         toWorkingIr _,
         resolveKinds _,
         inferTypes _,
         resolveGenders _)
      val names = Seq(
         "To Working IR",
         "Resolve Kinds",
         "Infer Types",
         "Resolve Genders")
      var c_BANG = c
      (names, passes).zipped.foreach { 
         (n,p) => {
            println("Starting " + n)
            c_BANG = p(c_BANG)
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
        m match {
           case m:InModule => InModule(m.info,m.name, m.ports, toStmt(m.body))
           case m:ExModule => m
        }
      }
      println("Before To Working IR")
      println(c.serialize())
     val x = Circuit(c.info,modulesx,c.main)
      println("After To Working IR")
     println(x.serialize())
     x
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
            println(kinds)
            m match {
               case m:InModule => find_stmt(m.body)
               case m:ExModule => false
            }
         }
       
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
      println("Before Resolve Kinds")
      println(c.serialize())
      val x = Circuit(c.info,modulesx,c.main)
      println("After Resolve Kinds")
      println(x.serialize())
      x
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
 
         m.ports.foreach(p => types += (p.name -> p.tpe))
         m match {
            case m:InModule => InModule(m.info,m.name,m.ports,infer_types_s(m.body))
            case m:ExModule => m
         }
       }
 
   
      // MAIN
      val modulesx = c.modules.map { 
         m => {
            val portsx = m.ports.map(p => Port(p.info,p.name,p.direction,remove_unknowns(p.tpe)))
            m match {
               case m:InModule => InModule(m.info,m.name,portsx,m.body)
               case m:ExModule => ExModule(m.info,m.name,portsx)
            }
         }
      }
   
      modulesx.foreach(m => module_types += (m.name -> module_type(m)))
      println("Before Infer Types")
      println(c.serialize())
      val x = Circuit(c.info,modulesx.map(m => infer_types(m)) , c.main )
      println("After Infer Types")
      println(x.serialize())
      x
   }

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
