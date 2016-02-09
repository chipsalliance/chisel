
package firrtl

import com.typesafe.scalalogging.LazyLogging
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer

import Utils._
import DebugUtils._
import PrimOps._


//@deprecated("This object will be replaced with package firrtl.passes")
//object Passes extends LazyLogging {
//
//
//   // TODO Perhaps we should get rid of Logger since this map would be nice
//   ////private val defaultLogger = Logger()
//   //private def mapNameToPass = Map[String, Circuit => Circuit] (
//   //  "infer-types" -> inferTypes
//   //)
//   var mname = ""
//   //def nameToPass(name: String): Circuit => Circuit = {
//     //mapNameToPass.getOrElse(name, throw new Exception("No Standard FIRRTL Pass of name " + name))
//     //name match {
//       //case "to-working-ir" => toWorkingIr
//       //case "infer-types" => inferTypes
//         // errrrrrrrrrr...
//       //case "renameall" => renameall(Map())
//     //}
//   //}
// 
//   private def toField(p: Port): Field = {
//     logger.debug(s"toField called on port ${p.serialize}")
//     p.direction match {
//       case INPUT  => Field(p.name, REVERSE, p.tpe)
//       case OUTPUT => Field(p.name, DEFAULT, p.tpe)
//     }
//   }
//   // ============== RESOLVE ALL ===================
//   def resolve (c:Circuit) = {c
//      //val passes = Seq(
//      //   toWorkingIr _,
//      //   resolveKinds _,
//      //   inferTypes _,
//      //   resolveGenders _,
//      //   pullMuxes _,
//      //   expandConnects _,
//      //   removeAccesses _)
//      //val names = Seq(
//      //   "To Working IR",
//      //   "Resolve Kinds",
//      //   "Infer Types",
//      //   "Resolve Genders",
//      //   "Pull Muxes",
//      //   "Expand Connects",
//      //   "Remove Accesses")
//      //var c_BANG = c
//      //(names, passes).zipped.foreach { 
//      //   (n,p) => {
//      //      println("Starting " + n)
//      //      c_BANG = p(c_BANG)
//      //      println(c_BANG.serialize())
//      //      println("Finished " + n)
//      //   }
//      //}
//      //c_BANG
//   }
// 
//
//  // ============== RESOLVE KINDS ==================
//  // ===============================================
//
//  // ============== INFER TYPES ==================
//
//  // ------------------ Utils -------------------------
//
//
//// =================== RESOLVE GENDERS =======================
//  // ===============================================
//
//  // =============== PULL MUXES ====================
//  // ===============================================
//
//
//
//   // ============ EXPAND CONNECTS ==================
//   // ---------------- UTILS ------------------
//   
//   
//   //---------------- Pass ---------------------
//   
//  // ===============================================
//
//
//
//   // ============ REMOVE ACCESSES ==================
//   // ---------------- UTILS ------------------
//
//
//  /** INFER TYPES
//   *
//   *  This pass infers the type field in all IR nodes by updating
//   *    and passing an environment to all statements in pre-order
//   *    traversal, and resolving types in expressions in post-
//   *    order traversal.
//   *  Type propagation for primary ops are defined in Primops.scala.
//   *  Type errors are not checked in this pass, as this is
//   *    postponed for a later/earlier pass.
//   */
//  // input -> flip
//  //private type TypeMap = Map[String, Type]
//  //private val TypeMap = Map[String, Type]().withDefaultValue(UnknownType)
//  //private def getBundleSubtype(t: Type, name: String): Type = {
//  //  t match {
//  //    case b: BundleType => {
//  //      val tpe = b.fields.find( _.name == name )
//  //      if (tpe.isEmpty) UnknownType
//  //      else tpe.get.tpe
//  //    }
//  //    case _ => UnknownType
//  //  }
//  //}
//  //private def getVectorSubtype(t: Type): Type = t.getType // Added for clarity
//  //// TODO Add genders
//  //private def inferExpTypes(typeMap: TypeMap)(exp: Expression): Expression = {
//  //  logger.debug(s"inferTypes called on ${exp.getClass.getSimpleName}")
//  //  exp.map(inferExpTypes(typeMap)) match {
//  //    case e: UIntValue => e
//  //    case e: SIntValue => e
//  //    case e: Ref => Ref(e.name, typeMap(e.name))
//  //    case e: SubField => SubField(e.exp, e.name, getBundleSubtype(e.exp.getType, e.name))
//  //    case e: SubIndex => SubIndex(e.exp, e.value, getVectorSubtype(e.exp.getType))
//  //    case e: SubAccess => SubAccess(e.exp, e.index, getVectorSubtype(e.exp.getType))
//  //    case e: DoPrim => lowerAndTypePrimOp(e)
//  //    case e: Expression => e
//  //  }
//  //}
//  //private def inferTypes(typeMap: TypeMap, stmt: Stmt): (Stmt, TypeMap) = {
//  //  logger.debug(s"inferTypes called on ${stmt.getClass.getSimpleName} ")
//  //  stmt.map(inferExpTypes(typeMap)) match {
//  //    case b: Begin => {
//  //      var tMap = typeMap
//  //      // TODO FIXME is map correctly called in sequential order
//  //      val body = b.stmts.map { s =>
//  //        val ret = inferTypes(tMap, s)
//  //        tMap = ret._2
//  //        ret._1
//  //      }
//  //      (Begin(body), tMap)
//  //    }
//  //    case s: DefWire => (s, typeMap ++ Map(s.name -> s.tpe))
//  //    case s: DefRegister => (s, typeMap ++ Map(s.name -> s.tpe))
//  //    case s: DefMemory => (s, typeMap ++ Map(s.name -> s.dataType))
//  //    case s: DefInstance => (s, typeMap ++ Map(s.name -> typeMap(s.module)))
//  //    case s: DefNode => (s, typeMap ++ Map(s.name -> s.value.getType))
//  //    case s: Conditionally => { // TODO Check: Assuming else block won't see when scope
//  //      val (conseq, cMap) = inferTypes(typeMap, s.conseq)
//  //      val (alt, aMap) = inferTypes(typeMap, s.alt)
//  //      (Conditionally(s.info, s.pred, conseq, alt), cMap ++ aMap)
//  //    }
//  //    case s: Stmt => (s, typeMap)
//  //  }
//  //}
//  //private def inferTypes(typeMap: TypeMap, m: Module): Module = {
//  //  logger.debug(s"inferTypes called on module ${m.name}")
//
//  //  val pTypeMap = m.ports.map( p => p.name -> p.tpe ).toMap
//
//  //  Module(m.info, m.name, m.ports, inferTypes(typeMap ++ pTypeMap, m.stmt)._1)
//  //}
//  //def inferTypes(c: Circuit): Circuit = {
//  //  logger.debug(s"inferTypes called on circuit ${c.name}")
//
//  //  // initialize typeMap with each module of circuit mapped to their bundled IO (ports converted to fields)
//  //  val typeMap = c.modules.map(m => m.name -> BundleType(m.ports.map(toField(_)))).toMap
//
//  //  //val typeMap = c.modules.flatMap(buildTypeMap).toMap
//  //  Circuit(c.info, c.name, c.modules.map(inferTypes(typeMap, _)))
//  //}
//
//  //def renameall(s : String)(implicit map : Map[String,String]) : String =
//  //  map getOrElse (s, s)
//
//  //def renameall(e : Expression)(implicit map : Map[String,String]) : Expression = {
//  //  logger.debug(s"renameall called on expression ${e.toString}")
//  //  e match {
//  //    case p : Ref =>
//  //      Ref(renameall(p.name), p.tpe)
//  //    case p : SubField =>
//  //      SubField(renameall(p.exp), renameall(p.name), p.tpe)
//  //    case p : SubIndex =>
//  //      SubIndex(renameall(p.exp), p.value, p.tpe)
//  //    case p : SubAccess =>
//  //      SubAccess(renameall(p.exp), renameall(p.index), p.tpe)
//  //    case p : Mux =>
//  //      Mux(renameall(p.cond), renameall(p.tval), renameall(p.fval), p.tpe)
//  //    case p : ValidIf =>
//  //      ValidIf(renameall(p.cond), renameall(p.value), p.tpe)
//  //    case p : DoPrim =>
//  //      println( p.args.map(x => renameall(x)) )
//  //      DoPrim(p.op, p.args.map(renameall), p.consts, p.tpe)
//  //    case p : Expression => p
//  //  }
//  //}
//
//  //def renameall(s : Stmt)(implicit map : Map[String,String]) : Stmt = {
//  //  logger.debug(s"renameall called on statement ${s.toString}")
//
//  //  s match {
//  //    case p : DefWire =>
//  //      DefWire(p.info, renameall(p.name), p.tpe)
//  //    case p: DefRegister =>
//  //      DefRegister(p.info, renameall(p.name), p.tpe, p.clock, p.reset, p.init)
//  //    case p : DefMemory =>
//  //      DefMemory(p.info, renameall(p.name), p.dataType, p.depth, p.writeLatency, p.readLatency, 
//  //                p.readers, p.writers, p.readwriters)
//  //    case p : DefInstance =>
//  //      DefInstance(p.info, renameall(p.name), renameall(p.module))
//  //    case p : DefNode =>
//  //      DefNode(p.info, renameall(p.name), renameall(p.value))
//  //    case p : Connect =>
//  //      Connect(p.info, renameall(p.loc), renameall(p.exp))
//  //    case p : BulkConnect =>
//  //      BulkConnect(p.info, renameall(p.loc), renameall(p.exp))
//  //    case p : IsInvalid =>
//  //      IsInvalid(p.info, renameall(p.exp))
//  //    case p : Stop =>
//  //      Stop(p.info, p.ret, renameall(p.clk), renameall(p.en))
//  //    case p : Print =>
//  //      Print(p.info, p.string, p.args.map(renameall), renameall(p.clk), renameall(p.en))
//  //    case p : Conditionally =>
//  //      Conditionally(p.info, renameall(p.pred), renameall(p.conseq), renameall(p.alt))
//  //    case p : Begin =>
//  //      Begin(p.stmts.map(renameall))
//  //    case p : Stmt => p
//  //  }
//  //}
//
//  //def renameall(p : Port)(implicit map : Map[String,String]) : Port = {
//  //  logger.debug(s"renameall called on port ${p.name}")
//  //  Port(p.info, renameall(p.name), p.dir, p.tpe)
//  //}
//
//  //def renameall(m : Module)(implicit map : Map[String,String]) : Module = {
//  //  logger.debug(s"renameall called on module ${m.name}")
//  //  Module(m.info, renameall(m.name), m.ports.map(renameall(_)), renameall(m.stmt))
//  //}
//
//  //def renameall(map : Map[String,String]) : Circuit => Circuit = {
//  //  c => {
//  //    implicit val imap = map
//  //    logger.debug(s"renameall called on circuit ${c.name} with %{renameto}")
//  //    Circuit(c.info, renameall(c.name), c.modules.map(renameall(_)))
//  //  }
//  //}
//}
