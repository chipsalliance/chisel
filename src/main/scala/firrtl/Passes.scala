
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
      case e: DoPrimop => lowerAndTypePrimop(e)
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

  /** FAME-1
   *
   *  This pass takes a lowered-to-ground circuit and performs a 
   *    FAME-1 (Decoupled) transformation to the circuit
   *
   *  TODO
   *   - SWITCH TO USING HIGH-LEVEL FIRRTL SO WE CAN MAINTAIN STRUCTURE OF BUNDLES
   *   - Add midas$fire : indicates when the module can operate
   *   - Add transform on each assignment to inputs/outputs to assign to data part of bundle
   *   - Add enable logic for each register
   *      * This should just be a when not(midas$fire) : reg := reg
   *        At bottom of module
   *   - QUESTIONS
   *      * Should we have Reset be a special Type?
   *
   *  NOTES
   *    - How do output consumes tie in to MIDAS fire? If all of our outputs are not consumed
   *      in a given cycle, do we block midas$fire on the next cycle? Perhaps there should be 
   *      a register for not having consumed all outputs last cycle
   *    - If our outputs are not consumed we also need to be sure not to consume out inputs,
   *      so the logic for this must depend on the previous cycle being consumed as well
   *    - We also need a way to determine the difference between the MIDAS modules and their
   *      connecting Queues, perhaps they should be MIDAS queues, which then perhaps prints
   *      out a listing of all queues so that they can be properly transformed
   *        * What do these MIDAS queues look like since we're enforcing true decoupled 
   *          interfaces?
   */
  private type PortMap = Map[String, Port]
  //private val PortMap = Map[String, Type]().withDefaultValue(UnknownType)
  private val f1TAvail = Field("avail", Default, UIntType(IntWidth(1)))
  private val f1TConsume = Field("consume", Reverse, UIntType(IntWidth(1)))
  private def fame1Transform(p: Port): Port = {
    if( p.name == "reset" ) p // omit reset
    else {
      p.tpe match {
        case ClockType => p // Omit clocktype
        case t: BundleType => throw new Exception("Bundle Types not supported in FAME-1 Transformation!")
        case t: VectorType => throw new Exception("Vector Types not supported in FAME-1 Transformation!")
        case t: Type => {
            Port(p.info, p.name, p.dir, BundleType(
                  Seq(f1TAvail, f1TConsume, Field("data", Default, t)))
                )
        }
      }
    }
  }
  private def fame1Transform(m: Module): Module = {
    println("fame1Transform called on module " + m.name)
    val ports = m.ports.map(fame1Transform)
    val portMap = Map(ports.map(p => (p.name, p)))
    println(portMap)
    Module(m.info, m.name, ports, m.stmt)
  }
  def fame1Transform(c: Circuit): Circuit = {
    Circuit(c.info, c.name, c.modules.map(fame1Transform))
  }

}
