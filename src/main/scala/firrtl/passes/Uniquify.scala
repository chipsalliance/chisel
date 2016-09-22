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
import scala.annotation.tailrec

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import MemPortUtils.memType

/** Resolve name collisions that would occur in [[LowerTypes]]
  *
  *  @note Must be run after [[InferTypes]] because [[ir.DefNode]]s need type
  *  @example
  *  {{{
  *      wire a = { b, c }[2]
  *      wire a_0
  *  }}}
  *    This lowers to:
  *  {{{
  *      wire a__0_b
  *      wire a__0_c
  *      wire a__1_b
  *      wire a__1_c
  *      wire a_0
  *  }}}
  *    There wouldn't be a collision even if we didn't map a -> a_, but
  *      there WOULD be collisions in references a[0] and a_0 so we still have
  *      to rename a
  */
object Uniquify extends Pass {
  def name = "Uniquify Identifiers"

  private case class UniquifyException(msg: String) extends FIRRTLException(msg)
  private def error(msg: String)(implicit sinfo: Info, mname: String) =
    throw new UniquifyException(s"$sinfo: [module $mname] $msg")

  // For creation of rename map
  private case class NameMapNode(name: String, elts: Map[String, NameMapNode])

  // Appends delim to prefix until no collisions of prefix + elts in names
  // We don't add an _ in the collision check because elts could be Seq("")
  //   In this case, we're just really checking if prefix itself collides
  @tailrec
  private def findValidPrefix(
      prefix: String,
      elts: Seq[String],
      namespace: collection.mutable.HashSet[String]): String = {
    elts find (elt => namespace.contains(prefix + elt)) match {
      case Some(_) => findValidPrefix(prefix + "_", elts, namespace)
      case None => prefix
    }
  }

  // Enumerates all possible names for a given type
  //   eg. foo : { bar : { a, b }[2], c }
  //   => foo, foo bar, foo bar 0, foo bar 1, foo bar 0 a, foo bar 0 b,
  //      foo bar 1 a, foo bar 1 b, foo c
  private def enumerateNames(tpe: Type): Seq[Seq[String]] = tpe match {
    case t: BundleType =>
      t.fields flatMap { f =>
        (enumerateNames(f.tpe) map (f.name +: _)) ++ Seq(Seq(f.name))
      }
    case t: VectorType =>
      ((0 until t.size) map (i => Seq(i.toString))) ++
      ((0 until t.size) flatMap { i =>
        enumerateNames(t.tpe) map (i.toString +: _)
      })
    case _ => Seq()
  }

  // Accepts a Type and an initial namespace
  // Returns new Type with uniquified names
  private def uniquifyNames(
      t: BundleType,
      namespace: collection.mutable.HashSet[String])
      (implicit sinfo: Info, mname: String): BundleType = {
    def recUniquifyNames(t: Type, namespace: collection.mutable.HashSet[String]): Type = t match {
      case t: BundleType =>
        // First add everything
        val newFields = t.fields map { f =>
          val newName = findValidPrefix(f.name, Seq(""), namespace)
          namespace += newName
          Field(newName, f.flip, f.tpe)
        } map { f => f.tpe match {
          case _: GroundType => f
          case _ =>
            val tpe = recUniquifyNames(f.tpe, collection.mutable.HashSet())
            val elts = enumerateNames(tpe)
            // Need leading _ for findValidPrefix, it doesn't add _ for checks
            val eltsNames = elts map (e => "_" + LowerTypes.loweredName(e))
            val prefix = findValidPrefix(f.name, eltsNames, namespace)
            // We added f.name in previous map, delete if we change it
            if (prefix != f.name) {
              namespace -= f.name
              namespace += prefix
            }
            namespace ++= (elts map (e => LowerTypes.loweredName(prefix +: e)))
            Field(prefix, f.flip, tpe)
          }
        }
        BundleType(newFields)
      case t: VectorType =>
        VectorType(recUniquifyNames(t.tpe, namespace), t.size)
      case t => t
    }
    recUniquifyNames(t, namespace) match {
      case t: BundleType => t
      case t => error("Shouldn't be here")
    }
  }

  // Creates a mapping from flattened references to members of $from ->
  //   flattened references to members of $to
  private def createNameMapping(
      from: Type,
      to: Type)
      (implicit sinfo: Info, mname: String): Map[String, NameMapNode] = {
    (from, to) match {
      case (from: BundleType, to: BundleType) =>
        (from.fields zip to.fields flatMap { case (f, t) =>
          val eltsMap = createNameMapping(f.tpe, t.tpe)
          if ((f.name != t.name) || eltsMap.nonEmpty) {
            Map(f.name -> NameMapNode(t.name, eltsMap))
          } else {
            Map[String, NameMapNode]()
          }
        }).toMap
      case (from: VectorType, to: VectorType) =>
        createNameMapping(from.tpe, to.tpe)
      case (from, to) =>
        if (from.getClass == to.getClass) Map()
        else error("Types to map between do not match!")
    }
  }

  // Maps names in expression to new uniquified names
  private def uniquifyNamesExp(
      exp: Expression,
      map: Map[String, NameMapNode])
      (implicit sinfo: Info, mname: String): Expression = {
    // Recursive Helper
    def rec(exp: Expression, m: Map[String, NameMapNode]):
        (Expression, Map[String, NameMapNode]) = exp match {
      case e: WRef =>
        if (m.contains(e.name)) {
          val node = m(e.name)
          (WRef(node.name, e.tpe, e.kind, e.gender), node.elts)
        }
        else (e, Map())
      case e: WSubField =>
        val (subExp, subMap) = rec(e.exp, m)
        val (retName, retMap) =
          if (subMap.contains(e.name)) {
            val node = subMap(e.name)
            (node.name, node.elts)
          } else {
            (e.name, Map[String, NameMapNode]())
          }
        (WSubField(subExp, retName, e.tpe, e.gender), retMap)
      case e: WSubIndex =>
        val (subExp, subMap) = rec(e.exp, m)
        (WSubIndex(subExp, e.value, e.tpe, e.gender), subMap)
      case e: WSubAccess =>
        val (subExp, subMap) = rec(e.exp, m)
        val index = uniquifyNamesExp(e.index, map)
        (WSubAccess(subExp, index, e.tpe, e.gender), subMap)
      case (_: UIntLiteral | _: SIntLiteral) => (exp, m)
      case (_: Mux | _: ValidIf | _: DoPrim) =>
        (exp map ((e: Expression) => uniquifyNamesExp(e, map)), m)
    }
    rec(exp, map)._1
  }

  // Uses map to recursively rename fields of tpe
  private def uniquifyNamesType(
      tpe: Type,
      map: Map[String, NameMapNode])
      (implicit sinfo: Info, mname: String): Type = tpe match {
    case t: BundleType =>
      val newFields = t.fields map { f =>
        if (map.contains(f.name)) {
          val node = map(f.name)
          Field(node.name, f.flip, uniquifyNamesType(f.tpe, node.elts))
        } else {
          f
        }
      }
      BundleType(newFields)
    case t: VectorType =>
      VectorType(uniquifyNamesType(t.tpe, map), t.size)
    case t => t
  }

  // Creates a Bundle Type from a Stmt
  def stmtToType(s: Statement)(implicit sinfo: Info, mname: String): BundleType = {
    // Recursive helper
    def recStmtToType(s: Statement): Seq[Field] = s match {
      case s: DefWire => Seq(Field(s.name, Default, s.tpe))
      case s: DefRegister => Seq(Field(s.name, Default, s.tpe))
      case s: WDefInstance => Seq(Field(s.name, Default, s.tpe))
      case s: DefMemory => s.dataType match {
        case (_: UIntType | _: SIntType) =>
          Seq(Field(s.name, Default, memType(s)))
        case tpe: BundleType =>
          val newFields = tpe.fields map ( f =>
            DefMemory(s.info, f.name, f.tpe, s.depth, s.writeLatency,
              s.readLatency, s.readers, s.writers, s.readwriters)
          ) flatMap (recStmtToType)
          Seq(Field(s.name, Default, BundleType(newFields)))
        case tpe: VectorType =>
          val newFields = (0 until tpe.size) map ( i =>
            s.copy(name = i.toString, dataType = tpe.tpe)
          ) flatMap (recStmtToType)
          Seq(Field(s.name, Default, BundleType(newFields)))
      }
      case s: DefNode => Seq(Field(s.name, Default, s.value.tpe))
      case s: Conditionally => recStmtToType(s.conseq) ++ recStmtToType(s.alt)
      case s: Block => (s.stmts map (recStmtToType)).flatten
      case s => Seq()
    }
    BundleType(recStmtToType(s))
  }

  // Everything wrapped in run so that it's thread safe
  def run(c: Circuit): Circuit = {
    // Debug state
    implicit var mname: String = ""
    implicit var sinfo: Info = NoInfo
    // Global state
    val portNameMap = collection.mutable.HashMap[String, Map[String, NameMapNode]]()
    val portTypeMap = collection.mutable.HashMap[String, Type]()

    def uniquifyModule(m: DefModule): DefModule = {
      val namespace = collection.mutable.HashSet[String]()
      val nameMap = collection.mutable.HashMap[String, NameMapNode]()

      def uniquifyExp(e: Expression): Expression = e match {
        case (_: WRef | _: WSubField | _: WSubIndex | _: WSubAccess ) =>
          uniquifyNamesExp(e, nameMap.toMap)
        case e: Mux => e map (uniquifyExp)
        case e: ValidIf => e map (uniquifyExp)
        case (_: UIntLiteral | _: SIntLiteral) => e
        case e: DoPrim => e map (uniquifyExp)
      }

      def uniquifyStmt(s: Statement): Statement = {
        s map uniquifyStmt map uniquifyExp match {
          case s: DefWire =>
            sinfo = s.info
            if (nameMap.contains(s.name)) {
              val node = nameMap(s.name)
              DefWire(s.info, node.name, uniquifyNamesType(s.tpe, node.elts))
            } else {
              s
            }
          case s: DefRegister =>
            sinfo = s.info
            if (nameMap.contains(s.name)) {
              val node = nameMap(s.name)
              DefRegister(s.info, node.name, uniquifyNamesType(s.tpe, node.elts),
                          s.clock, s.reset, s.init)
            } else {
              s
            }
          case s: WDefInstance =>
            sinfo = s.info
            if (nameMap.contains(s.name)) {
              val node = nameMap(s.name)
              WDefInstance(s.info, node.name, s.module, s.tpe)
            } else {
              s
            }
          case s: DefMemory =>
            sinfo = s.info
            if (nameMap.contains(s.name)) {
              val node = nameMap(s.name)
              val dataType = uniquifyNamesType(s.dataType, node.elts)
              val mem = s.copy(name = node.name, dataType = dataType)
              // Create new mapping to handle references to memory data fields
              val uniqueMemMap = createNameMapping(memType(s), memType(mem))
              nameMap(s.name) = NameMapNode(node.name, node.elts ++ uniqueMemMap)
              mem
            } else {
              s
            }
          case s: DefNode =>
            sinfo = s.info
            if (nameMap.contains(s.name)) {
              val node = nameMap(s.name)
              DefNode(s.info, node.name, s.value)
            } else {
              s
            }
          case s => s
        }
      }

      def uniquifyBody(s: Statement): Statement = {
        val bodyType = stmtToType(s)
        val uniqueBodyType = uniquifyNames(bodyType, namespace)
        val localMap = createNameMapping(bodyType, uniqueBodyType)
        nameMap ++= localMap

        uniquifyStmt(s)
      }

      // uniquify ports and expand aggregate types
      sinfo = m.info
      mname = m.name
      m match {
        case m: ExtModule => m
        case m: Module =>
          // Adds port names to namespace and namemap
          nameMap ++= portNameMap(m.name)
          namespace ++= create_exps("", portTypeMap(m.name)) map
                        (LowerTypes.loweredName) map (_.tail)
          m.copy(body = uniquifyBody(m.body) )
      }
    }

    def uniquifyPorts(m: DefModule): DefModule = {
      def uniquifyPorts(ports: Seq[Port]): Seq[Port] = {
        val portsType = BundleType(ports map {
          case Port(_, name, dir, tpe) => Field(name, to_flip(dir), tpe)
        })
        val uniquePortsType = uniquifyNames(portsType, collection.mutable.HashSet())
        val localMap = createNameMapping(portsType, uniquePortsType)
        portNameMap += (m.name -> localMap)
        portTypeMap += (m.name -> uniquePortsType)

        ports zip uniquePortsType.fields map { case (p, f) =>
          Port(p.info, f.name, p.direction, f.tpe)
        }
      }

      sinfo = m.info
      mname = m.name
      m match {
        case m: ExtModule => m.copy(ports = uniquifyPorts(m.ports))
        case m: Module => m.copy(ports = uniquifyPorts(m.ports))
      }
    }

    sinfo = c.info
    Circuit(c.info, c.modules map uniquifyPorts map uniquifyModule, c.main)
  }
}

