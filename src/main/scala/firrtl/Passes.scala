
package firrtl

import com.typesafe.scalalogging.LazyLogging

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
      case "infer-types" => inferTypes
        // errrrrrrrrrr...
      case "renameall" => renameall(Map())
    }
  }

  private def toField(p: Port): Field = {
    logger.debug(s"toField called on port ${p.serialize}")
    p.dir match {
      case Input  => Field(p.name, Reverse, p.tpe)
      case Output => Field(p.name, Default, p.tpe)
    }
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
  private def getBundleSubtype(t: Type, name: String): Type = {
    t match {
      case b: BundleType => {
        val tpe = b.fields.find( _.name == name )
        if (tpe.isEmpty) UnknownType
        else tpe.get.tpe
      }
      case _ => UnknownType
    }
  }
  private def getVectorSubtype(t: Type): Type = t.getType // Added for clarity
  private type TypeMap = Map[String, Type]
  /*def inferTypes(c: Circuit): Circuit = {
    val moduleTypeMap = Map[String, Type]().withDefaultValue(UnknownType)
    def inferTypes(m: Module): Module = {
      val typeMap = Map[String, Type]().withDefaultValue(UnknownType)
      def inferExpTypes(exp: Expression): Expression = {
        //logger.debug(s"inferTypes called on ${exp.getClass.getSimpleName}")
        exp.map(inferExpTypes) match {
          case e: UIntValue => e
          case e: SIntValue => e
          case e: Ref => Ref(e.name, typeMap(e.name))
          case e: SubField => SubField(e.exp, e.name, getBundleSubtype(e.exp.getType, e.name))
          case e: SubIndex => SubIndex(e.exp, e.value, getVectorSubtype(e.exp.getType))
          case e: SubAccess => SubAccess(e.exp, e.index, getVectorSubtype(e.exp.getType))
          case e: DoPrim => lowerAndTypePrimOp(e)
          case e: Expression => e
        }
      }
      def inferStmtTypes(stmt: Stmt): (Stmt) = {
        //logger.debug(s"inferStmtTypes called on ${stmt.getClass.getSimpleName} ")
        stmt match {
          case s: DefWire => 
             typeMap(s.name) = s.tpe
             s
          case s: DefRegister => 
             typeMap(s.name) = get_tpe(s)
             s
          case s: DefMemory => 
             typeMap(s.name) = get_tpe(s)
             s
          case s: DefInstance => (s, typeMap ++ Map(s.name -> typeMap(s.module)))
          case s: DefNode => (s, typeMap ++ Map(s.name -> s.value.getType))
          case s: s.map(inferStmtTypes)
        }.map(inferExpTypes)
      }
      //logger.debug(s"inferTypes called on module ${m.name}")

      m.ports.for( p => typeMap(p.name) = p.tpe)
      Module(m.info, m.name, m.ports, inferStmtTypes(m.stmt))
    }
    //logger.debug(s"inferTypes called on circuit ${c.name}")

    // initialize typeMap with each module of circuit mapped to their bundled IO (ports converted to fields)
    val typeMap = c.modules.map(m => m.name -> BundleType(m.ports.map(toField(_)))).toMap
    Circuit(c.info, c.name, c.modules.map(inferTypes(typeMap, _)))
  }*/

  def renameall(s : String)(implicit map : Map[String,String]) : String =
    map getOrElse (s, s)

  def renameall(e : Expression)(implicit map : Map[String,String]) : Expression = {
    logger.debug(s"renameall called on expression ${e.toString}")
    e match {
      case p : Ref =>
        Ref(renameall(p.name), p.tpe)
      case p : SubField =>
        SubField(renameall(p.exp), renameall(p.name), p.tpe)
      case p : SubIndex =>
        SubIndex(renameall(p.exp), p.value, p.tpe)
      case p : SubAccess =>
        SubAccess(renameall(p.exp), renameall(p.index), p.tpe)
      case p : Mux =>
        Mux(renameall(p.cond), renameall(p.tval), renameall(p.fval), p.tpe)
      case p : ValidIf =>
        ValidIf(renameall(p.cond), renameall(p.value), p.tpe)
      case p : DoPrim =>
        println( p.args.map(x => renameall(x)) )
        DoPrim(p.op, p.args.map(renameall), p.consts, p.tpe)
      case p : Expression => p
    }
  }

  def renameall(s : Stmt)(implicit map : Map[String,String]) : Stmt = {
    logger.debug(s"renameall called on statement ${s.toString}")

    s match {
      case p : DefWire =>
        DefWire(p.info, renameall(p.name), p.tpe)
      case p: DefRegister =>
        DefRegister(p.info, renameall(p.name), p.tpe, p.clock, p.reset, p.init)
      case p : DefMemory =>
        DefMemory(p.info, renameall(p.name), p.dataType, p.depth, p.writeLatency, p.readLatency, 
                  p.readers, p.writers, p.readwriters)
      case p : DefInstance =>
        DefInstance(p.info, renameall(p.name), renameall(p.module))
      case p : DefNode =>
        DefNode(p.info, renameall(p.name), renameall(p.value))
      case p : Connect =>
        Connect(p.info, renameall(p.loc), renameall(p.exp))
      case p : BulkConnect =>
        BulkConnect(p.info, renameall(p.loc), renameall(p.exp))
      case p : IsInvalid =>
        IsInvalid(p.info, renameall(p.exp))
      case p : Stop =>
        Stop(p.info, p.ret, renameall(p.clk), renameall(p.en))
      case p : Print =>
        Print(p.info, p.string, p.args.map(renameall), renameall(p.clk), renameall(p.en))
      case p : Conditionally =>
        Conditionally(p.info, renameall(p.pred), renameall(p.conseq), renameall(p.alt))
      case p : Begin =>
        Begin(p.stmts.map(renameall))
      case p : Stmt => p
    }
  }

  def renameall(p : Port)(implicit map : Map[String,String]) : Port = {
    logger.debug(s"renameall called on port ${p.name}")
    Port(p.info, renameall(p.name), p.dir, p.tpe)
  }

  def renameall(m : Module)(implicit map : Map[String,String]) : Module = {
    logger.debug(s"renameall called on module ${m.name}")
    Module(m.info, renameall(m.name), m.ports.map(renameall(_)), renameall(m.stmt))
  }

  def renameall(map : Map[String,String]) : Circuit => Circuit = {
    c => {
      implicit val imap = map
      logger.debug(s"renameall called on circuit ${c.name} with %{renameto}")
      Circuit(c.info, renameall(c.name), c.modules.map(renameall(_)))
    }
  }
}
