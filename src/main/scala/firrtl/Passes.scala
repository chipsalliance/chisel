
package firrtl

import com.typesafe.scalalogging.LazyLogging

import Utils._
import DebugUtils._
import Primops._

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
  private type TypeMap = Map[String, Type]
  private val TypeMap = Map[String, Type]().withDefaultValue(UnknownType)
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
  // TODO Add genders
  private def inferExpTypes(typeMap: TypeMap)(exp: Exp): Exp = {
    logger.debug(s"inferTypes called on ${exp.getClass.getSimpleName}")
    exp.map(inferExpTypes(typeMap)) match {
      case e: UIntValue => e
      case e: SIntValue => e
      case e: Ref => Ref(e.name, typeMap(e.name))
      case e: Subfield => Subfield(e.exp, e.name, getBundleSubtype(e.exp.getType, e.name))
      case e: Index => Index(e.exp, e.value, getVectorSubtype(e.exp.getType))
      case e: DoPrimop => lowerAndTypePrimop(e)
      case e: Exp => e
    }
  }
  private def inferTypes(typeMap: TypeMap, stmt: Stmt): (Stmt, TypeMap) = {
    logger.debug(s"inferTypes called on ${stmt.getClass.getSimpleName} ")
    stmt.map(inferExpTypes(typeMap)) match {
      case b: Block => {
        var tMap = typeMap
        // TODO FIXME is map correctly called in sequential order
        val body = b.stmts.map { s =>
          val ret = inferTypes(tMap, s)
          tMap = ret._2
          ret._1
        }
        (Block(body), tMap)
      }
      case s: DefWire => (s, typeMap ++ Map(s.name -> s.tpe))
      case s: DefReg => (s, typeMap ++ Map(s.name -> s.tpe))
      case s: DefMemory => (s, typeMap ++ Map(s.name -> s.tpe))
      case s: DefInst => (s, typeMap ++ Map(s.name -> s.module.getType))
      case s: DefNode => (s, typeMap ++ Map(s.name -> s.value.getType))
      case s: DefPoison => (s, typeMap ++ Map(s.name -> s.tpe))
      case s: DefAccessor => (s, typeMap ++ Map(s.name -> getVectorSubtype(s.source.getType)))
      case s: When => { // TODO Check: Assuming else block won't see when scope
        val (conseq, cMap) = inferTypes(typeMap, s.conseq)
        val (alt, aMap) = inferTypes(typeMap, s.alt)
        (When(s.info, s.pred, conseq, alt), cMap ++ aMap)
      }
      case s: Stmt => (s, typeMap)
    }
  }
  private def inferTypes(typeMap: TypeMap, m: Module): Module = {
    logger.debug(s"inferTypes called on module ${m.name}")

    val pTypeMap = m.ports.map( p => p.name -> p.tpe ).toMap

    Module(m.info, m.name, m.ports, inferTypes(typeMap ++ pTypeMap, m.stmt)._1)
  }
  def inferTypes(c: Circuit): Circuit = {
    logger.debug(s"inferTypes called on circuit ${c.name}")

    // initialize typeMap with each module of circuit mapped to their bundled IO (ports converted to fields)
    val typeMap = c.modules.map(m => m.name -> BundleType(m.ports.map(toField(_)))).toMap

    //val typeMap = c.modules.flatMap(buildTypeMap).toMap
    Circuit(c.info, c.name, c.modules.map(inferTypes(typeMap, _)))
  }

  def renameall(s : String)(implicit map : Map[String,String]) : String =
    map getOrElse (s, s)

  def renameall(e : Exp)(implicit map : Map[String,String]) : Exp = {
    logger.debug(s"renameall called on expression ${e.toString}")
    e match {
      case p : Ref =>
        Ref(renameall(p.name), p.tpe)
      case p : Subfield =>
        Subfield(renameall(p.exp), renameall(p.name), p.tpe)
      case p : Index =>
        Index(renameall(p.exp), p.value, p.tpe)
      case p : DoPrimop =>
        println( p.args.map(x => renameall(x)) )
        DoPrimop(p.op, p.args.map(x => renameall(x)), p.consts, p.tpe)
      case p : Exp => p
    }
  }

  def renameall(s : Stmt)(implicit map : Map[String,String]) : Stmt = {
    logger.debug(s"renameall called on statement ${s.toString}")

    s match {
      case p : DefWire =>
        DefWire(p.info, renameall(p.name), p.tpe)
      case p: DefReg =>
        DefReg(p.info, renameall(p.name), p.tpe, p.clock, p.reset)
      case p : DefMemory =>
        DefMemory(p.info, renameall(p.name), p.seq, p.tpe, p.clock)
      case p : DefInst =>
        DefInst(p.info, renameall(p.name), renameall(p.module))
      case p : DefNode =>
        DefNode(p.info, renameall(p.name), renameall(p.value))
      case p : DefPoison =>
        DefPoison(p.info, renameall(p.name), p.tpe)
      case p : DefAccessor =>
        DefAccessor(p.info, renameall(p.name), p.dir, renameall(p.source), renameall(p.index))
      case p : OnReset =>
        OnReset(p.info, renameall(p.lhs), renameall(p.rhs))
      case p : Connect =>
        Connect(p.info, renameall(p.lhs), renameall(p.rhs))
      case p : BulkConnect =>
        BulkConnect(p.info, renameall(p.lhs), renameall(p.rhs))
      case p : When =>
        When(p.info, renameall(p.pred), renameall(p.conseq), renameall(p.alt))
      case p : Assert =>
        Assert(p.info, renameall(p.pred))
      case p : Block =>
        Block(p.stmts.map(renameall))
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
