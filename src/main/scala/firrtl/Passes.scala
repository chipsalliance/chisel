
package firrtl

import Utils._
import DebugUtils._
import Primops._

object Passes {

  private def toField(p: Port)(implicit logger: Logger): Field = {
    logger.trace(s"toField called on port ${p.serialize}")
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
   *  Type propagation for primary ops are defined here.
   *  Notable cases: LetRec requires updating environment before
   *    resolving the subexpressions in its elements.
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
  private def inferExpTypes(typeMap: TypeMap)(exp: Exp)(implicit logger: Logger): Exp = {
    logger.trace(s"inferTypes called on ${exp.getClass.getSimpleName}")
    exp.map(inferExpTypes(typeMap)) match {
      case e: UIntValue => e
      case e: SIntValue => e
      case e: Ref => Ref(e.name, typeMap(e.name))
      case e: Subfield => Subfield(e.exp, e.name, getBundleSubtype(e.exp.getType, e.name))
      case e: Index => Index(e.exp, e.value, getVectorSubtype(e.exp.getType))
      case e: DoPrimOp => lowerAndTypePrimop(e)
      case e: Exp => e
    }
  }
  private def inferTypes(typeMap: TypeMap, stmt: Stmt)(implicit logger: Logger): (Stmt, TypeMap) = {
    logger.trace(s"inferTypes called on ${stmt.getClass.getSimpleName} ")
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
  private def inferTypes(typeMap: TypeMap, m: Module)(implicit logger: Logger): Module = {
    logger.trace(s"inferTypes called on module ${m.name}")

    val pTypeMap = m.ports.map( p => p.name -> p.tpe ).toMap

    Module(m.info, m.name, m.ports, inferTypes(typeMap ++ pTypeMap, m.stmt)._1)
  }
  def inferTypes(c: Circuit)(implicit logger: Logger): Circuit = {
    logger.trace(s"inferTypes called on circuit ${c.name}")

    // initialize typeMap with each module of circuit mapped to their bundled IO (ports converted to fields)
    val typeMap = c.modules.map(m => m.name -> BundleType(m.ports.map(toField(_)))).toMap

    //val typeMap = c.modules.flatMap(buildTypeMap).toMap
    Circuit(c.info, c.name, c.modules.map(inferTypes(typeMap, _)))
  }

}
