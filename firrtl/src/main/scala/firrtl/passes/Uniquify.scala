// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import scala.annotation.tailrec
import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.options.Dependency

import MemPortUtils.memType

/** Resolve name collisions that would occur in the old [[LowerTypes]] pass
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
@deprecated("Uniquify is now part of LowerTypes", "FIRRTL 1.4.0")
object Uniquify extends Transform with DependencyAPIMigration {

  override def prerequisites = firrtl.stage.Forms.MinimalHighForm

  override def invalidates(a: Transform): Boolean = false

  private case class UniquifyException(msg: String) extends FirrtlInternalException(msg)
  private def error(msg: String)(implicit sinfo: Info, mname: String) =
    throw new UniquifyException(s"$sinfo: [moduleOpt $mname] $msg")

  // For creation of rename map
  private case class NameMapNode(name: String, elts: Map[String, NameMapNode])

  /** Appends delim to prefix until no collisions of prefix + elts in names We don't add an _ in the collision check
    * because elts could be Seq("") In this case, we're just really checking if prefix itself collides
    */
  @deprecated("Use firrtl.Namespace.findValidPrefix", "FIRRTL 1.4.0")
  def findValidPrefix(
    prefix:    String,
    elts:      Seq[String],
    namespace: collection.mutable.HashSet[String]
  ): String = Namespace.findValidPrefix(prefix, elts, namespace)

  /** Creates a Bundle Type from a Stmt */
  @deprecated("Use firrtl.Utils.stmtToType", "FIRRTL 1.4.0")
  def stmtToType(s: Statement)(implicit sinfo: Info, mname: String): BundleType =
    Utils.stmtToType(s)

  // Accepts a Type and an initial namespace
  // Returns new Type with uniquified names
  private def uniquifyNames(
    t:         BundleType,
    namespace: collection.mutable.HashSet[String]
  )(
    implicit sinfo: Info,
    mname:          String
  ): BundleType = {
    def recUniquifyNames(t: Type, namespace: collection.mutable.HashSet[String]): (Type, Seq[String]) = t match {
      case tx: BundleType =>
        // First add everything
        val newFieldsAndElts = tx.fields.map { f =>
          val newName = Namespace.findValidPrefix(f.name, Seq(""), namespace)
          namespace += newName
          Field(newName, f.flip, f.tpe)
        }.map { f =>
          f.tpe match {
            case _: GroundType => (f, Seq[String](f.name))
            case _ =>
              val (tpe, eltsx) = recUniquifyNames(f.tpe, collection.mutable.HashSet())
              // Need leading _ for findValidPrefix, it doesn't add _ for checks
              val eltsNames: Seq[String] = eltsx.map(e => "_" + e)
              val prefix = Namespace.findValidPrefix(f.name, eltsNames, namespace)
              // We added f.name in previous map, delete if we change it
              if (prefix != f.name) {
                namespace -= f.name
                namespace += prefix
              }
              val newElts: Seq[String] = eltsx.map(e => LowerTypes.loweredName(prefix +: Seq(e)))
              namespace ++= newElts
              (Field(prefix, f.flip, tpe), prefix +: newElts)
          }
        }
        val (newFields, elts) = newFieldsAndElts.unzip
        (BundleType(newFields), elts.flatten)
      case tx: VectorType =>
        val (tpe, elts) = recUniquifyNames(tx.tpe, namespace)
        val newElts = ((0 until tx.size).map(i => i.toString)) ++
          ((0 until tx.size).flatMap { i =>
            elts.map(e => LowerTypes.loweredName(Seq(i.toString, e)))
          })
        (VectorType(tpe, tx.size), newElts)
      case tx => (tx, Nil)
    }
    val (tpe, _) = recUniquifyNames(t, namespace)
    tpe match {
      case tx: BundleType => tx
      case tx => throwInternalError(s"uniquifyNames: shouldn't be here - $tx")
    }
  }

  // Creates a mapping from flattened references to members of $from ->
  //   flattened references to members of $to
  private def createNameMapping(
    from: Type,
    to:   Type
  )(
    implicit sinfo: Info,
    mname:          String
  ): Map[String, NameMapNode] = {
    (from, to) match {
      case (fromx: BundleType, tox: BundleType) =>
        (fromx.fields
          .zip(tox.fields)
          .flatMap {
            case (f, t) =>
              val eltsMap = createNameMapping(f.tpe, t.tpe)
              if ((f.name != t.name) || eltsMap.nonEmpty) {
                Map(f.name -> NameMapNode(t.name, eltsMap))
              } else {
                Map[String, NameMapNode]()
              }
          })
          .toMap
      case (fromx: VectorType, tox: VectorType) =>
        createNameMapping(fromx.tpe, tox.tpe)
      case (fromx, tox) =>
        if (fromx.getClass == tox.getClass) Map()
        else error("Types to map between do not match!")
    }
  }

  // Maps names in expression to new uniquified names
  private def uniquifyNamesExp(
    exp: Expression,
    map: Map[String, NameMapNode]
  )(
    implicit sinfo: Info,
    mname:          String
  ): Expression = {
    // Recursive Helper
    def rec(exp: Expression, m: Map[String, NameMapNode]): (Expression, Map[String, NameMapNode]) = exp match {
      case e: WRef =>
        if (m.contains(e.name)) {
          val node = m(e.name)
          (WRef(node.name, e.tpe, e.kind, e.flow), node.elts)
        } else (e, Map())
      case e: WSubField =>
        val (subExp, subMap) = rec(e.expr, m)
        val (retName, retMap) =
          if (subMap.contains(e.name)) {
            val node = subMap(e.name)
            (node.name, node.elts)
          } else {
            (e.name, Map[String, NameMapNode]())
          }
        (WSubField(subExp, retName, e.tpe, e.flow), retMap)
      case e: WSubIndex =>
        val (subExp, subMap) = rec(e.expr, m)
        (WSubIndex(subExp, e.value, e.tpe, e.flow), subMap)
      case e: WSubAccess =>
        val (subExp, subMap) = rec(e.expr, m)
        val index = uniquifyNamesExp(e.index, map)
        (WSubAccess(subExp, index, e.tpe, e.flow), subMap)
      case (_: UIntLiteral | _: SIntLiteral) => (exp, m)
      case (_: Mux | _: ValidIf | _: DoPrim) =>
        (exp.map((e: Expression) => uniquifyNamesExp(e, map)), m)
    }
    rec(exp, map)._1
  }

  // Uses map to recursively rename fields of tpe
  private def uniquifyNamesType(
    tpe: Type,
    map: Map[String, NameMapNode]
  )(
    implicit sinfo: Info,
    mname:          String
  ): Type = tpe match {
    case t: BundleType =>
      val newFields = t.fields.map { f =>
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

  // Everything wrapped in run so that it's thread safe
  @deprecated(
    "The functionality of Uniquify is now part of LowerTypes." +
      "Please file an issue with firrtl if you use Uniquify outside of the context of LowerTypes.",
    "Firrtl 1.4"
  )
  def execute(state: CircuitState): CircuitState = {
    val c = state.circuit
    val renames = RenameMap()
    renames.setCircuit(c.main)
    // Debug state
    implicit var mname: String = ""
    implicit var sinfo: Info = NoInfo
    // Global state
    val portNameMap = collection.mutable.HashMap[String, Map[String, NameMapNode]]()
    val portTypeMap = collection.mutable.HashMap[String, Type]()

    def uniquifyModule(renames: RenameMap)(m: DefModule): DefModule = {
      renames.setModule(m.name)
      val namespace = collection.mutable.HashSet[String]()
      val nameMap = collection.mutable.HashMap[String, NameMapNode]()

      def uniquifyExp(e: Expression): Expression = e match {
        case (_: WRef | _: WSubField | _: WSubIndex | _: WSubAccess) =>
          uniquifyNamesExp(e, nameMap.toMap)
        case e: Mux     => e.map(uniquifyExp)
        case e: ValidIf => e.map(uniquifyExp)
        case (_: UIntLiteral | _: SIntLiteral) => e
        case e: DoPrim => e.map(uniquifyExp)
      }

      def uniquifyStmt(s: Statement): Statement = {
        s.map(uniquifyStmt).map(uniquifyExp) match {
          case sx: DefWire =>
            sinfo = sx.info
            if (nameMap.contains(sx.name)) {
              val node = nameMap(sx.name)
              val newType = uniquifyNamesType(sx.tpe, node.elts)
              (Utils.create_exps(sx.name, sx.tpe).zip(Utils.create_exps(node.name, newType))).foreach {
                case (from, to) => renames.rename(from.serialize, to.serialize)
              }
              DefWire(sx.info, node.name, newType)
            } else {
              sx
            }
          case sx: DefRegister =>
            sinfo = sx.info
            if (nameMap.contains(sx.name)) {
              val node = nameMap(sx.name)
              val newType = uniquifyNamesType(sx.tpe, node.elts)
              (Utils.create_exps(sx.name, sx.tpe).zip(Utils.create_exps(node.name, newType))).foreach {
                case (from, to) => renames.rename(from.serialize, to.serialize)
              }
              DefRegister(sx.info, node.name, newType, sx.clock, sx.reset, sx.init)
            } else {
              sx
            }
          case sx: WDefInstance =>
            sinfo = sx.info
            if (nameMap.contains(sx.name)) {
              val node = nameMap(sx.name)
              val newType = portTypeMap(sx.module)
              (Utils.create_exps(sx.name, sx.tpe).zip(Utils.create_exps(node.name, newType))).foreach {
                case (from, to) => renames.rename(from.serialize, to.serialize)
              }
              WDefInstance(sx.info, node.name, sx.module, newType)
            } else {
              sx
            }
          case sx: DefMemory =>
            sinfo = sx.info
            if (nameMap.contains(sx.name)) {
              val node = nameMap(sx.name)
              val dataType = uniquifyNamesType(sx.dataType, node.elts)
              val mem = sx.copy(name = node.name, dataType = dataType)
              // Create new mapping to handle references to memory data fields
              val uniqueMemMap = createNameMapping(memType(sx), memType(mem))
              (Utils.create_exps(sx.name, memType(sx)).zip(Utils.create_exps(node.name, memType(mem)))).foreach {
                case (from, to) => renames.rename(from.serialize, to.serialize)
              }
              nameMap(sx.name) = NameMapNode(node.name, node.elts ++ uniqueMemMap)
              mem
            } else {
              sx
            }
          case sx: DefNode =>
            sinfo = sx.info
            if (nameMap.contains(sx.name)) {
              val node = nameMap(sx.name)
              (Utils
                .create_exps(sx.name, s.asInstanceOf[DefNode].value.tpe)
                .zip(Utils.create_exps(node.name, sx.value.tpe)))
                .foreach {
                  case (from, to) => renames.rename(from.serialize, to.serialize)
                }
              DefNode(sx.info, node.name, sx.value)
            } else {
              sx
            }
          case sx => sx
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
        case m: Module    =>
          // Adds port names to namespace and namemap
          nameMap ++= portNameMap(m.name)
          namespace ++= create_exps("", portTypeMap(m.name)).map(LowerTypes.loweredName).map(_.tail)
          m.copy(body = uniquifyBody(m.body))
      }
    }

    def uniquifyPorts(renames: RenameMap)(m: DefModule): DefModule = {
      renames.setModule(m.name)
      def uniquifyPorts(ports: Seq[Port]): Seq[Port] = {
        val portsType = BundleType(ports.map {
          case Port(_, name, dir, tpe) => Field(name, to_flip(dir), tpe)
        })
        val uniquePortsType = uniquifyNames(portsType, collection.mutable.HashSet())
        val localMap = createNameMapping(portsType, uniquePortsType)
        portNameMap += (m.name -> localMap)
        portTypeMap += (m.name -> uniquePortsType)

        ports.zip(uniquePortsType.fields).map {
          case (p, f) =>
            (Utils.create_exps(p.name, p.tpe).zip(Utils.create_exps(f.name, f.tpe))).foreach {
              case (from, to) => renames.rename(from.serialize, to.serialize)
            }
            Port(p.info, f.name, p.direction, f.tpe)
        }
      }

      sinfo = m.info
      mname = m.name
      m match {
        case m: ExtModule => m.copy(ports = uniquifyPorts(m.ports))
        case m: Module    => m.copy(ports = uniquifyPorts(m.ports))
      }
    }

    sinfo = c.info
    val result = Circuit(c.info, c.modules.map(uniquifyPorts(renames)).map(uniquifyModule(renames)), c.main)
    state.copy(circuit = result, renames = Some(renames))
  }
}
