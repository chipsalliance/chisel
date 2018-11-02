// See LICENSE for license details.

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
object Uniquify extends Transform {
  def inputForm = UnknownForm
  def outputForm = UnknownForm
  private case class UniquifyException(msg: String) extends FIRRTLException(msg)
  private def error(msg: String)(implicit sinfo: Info, mname: String) =
    throw new UniquifyException(s"$sinfo: [moduleOpt $mname] $msg")

  // For creation of rename map
  private case class NameMapNode(name: String, elts: Map[String, NameMapNode])

  // Appends delim to prefix until no collisions of prefix + elts in names
  // We don't add an _ in the collision check because elts could be Seq("")
  //   In this case, we're just really checking if prefix itself collides
  @tailrec
  def findValidPrefix(
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
  private [firrtl] def enumerateNames(tpe: Type): Seq[Seq[String]] = tpe match {
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
      case tx: BundleType =>
        // First add everything
        val newFields = tx.fields map { f =>
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
      case tx: VectorType =>
        VectorType(recUniquifyNames(tx.tpe, namespace), tx.size)
      case tx => tx
    }
    recUniquifyNames(t, namespace) match {
      case tx: BundleType => tx
      case tx => throwInternalError(s"uniquifyNames: shouldn't be here - $tx")
    }
  }

  // Creates a mapping from flattened references to members of $from ->
  //   flattened references to members of $to
  private def createNameMapping(
      from: Type,
      to: Type)
      (implicit sinfo: Info, mname: String): Map[String, NameMapNode] = {
    (from, to) match {
      case (fromx: BundleType, tox: BundleType) =>
        (fromx.fields zip tox.fields flatMap { case (f, t) =>
          val eltsMap = createNameMapping(f.tpe, t.tpe)
          if ((f.name != t.name) || eltsMap.nonEmpty) {
            Map(f.name -> NameMapNode(t.name, eltsMap))
          } else {
            Map[String, NameMapNode]()
          }
        }).toMap
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
        val (subExp, subMap) = rec(e.expr, m)
        val (retName, retMap) =
          if (subMap.contains(e.name)) {
            val node = subMap(e.name)
            (node.name, node.elts)
          } else {
            (e.name, Map[String, NameMapNode]())
          }
        (WSubField(subExp, retName, e.tpe, e.gender), retMap)
      case e: WSubIndex =>
        val (subExp, subMap) = rec(e.expr, m)
        (WSubIndex(subExp, e.value, e.tpe, e.gender), subMap)
      case e: WSubAccess =>
        val (subExp, subMap) = rec(e.expr, m)
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
      case sx: DefWire => Seq(Field(sx.name, Default, sx.tpe))
      case sx: DefRegister => Seq(Field(sx.name, Default, sx.tpe))
      case sx: WDefInstance => Seq(Field(sx.name, Default, sx.tpe))
      case sx: DefMemory => sx.dataType match {
        case (_: UIntType | _: SIntType | _: FixedType) =>
          Seq(Field(sx.name, Default, memType(sx)))
        case tpe: BundleType =>
          val newFields = tpe.fields map ( f =>
            DefMemory(sx.info, f.name, f.tpe, sx.depth, sx.writeLatency,
              sx.readLatency, sx.readers, sx.writers, sx.readwriters)
          ) flatMap recStmtToType
          Seq(Field(sx.name, Default, BundleType(newFields)))
        case tpe: VectorType =>
          val newFields = (0 until tpe.size) map ( i =>
            sx.copy(name = i.toString, dataType = tpe.tpe)
          ) flatMap recStmtToType
          Seq(Field(sx.name, Default, BundleType(newFields)))
      }
      case sx: DefNode => Seq(Field(sx.name, Default, sx.value.tpe))
      case sx: Conditionally => recStmtToType(sx.conseq) ++ recStmtToType(sx.alt)
      case sx: Block => (sx.stmts map recStmtToType).flatten
      case sx => Seq()
    }
    BundleType(recStmtToType(s))
  }

  // Everything wrapped in run so that it's thread safe
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
        case (_: WRef | _: WSubField | _: WSubIndex | _: WSubAccess ) =>
          uniquifyNamesExp(e, nameMap.toMap)
        case e: Mux => e map uniquifyExp
        case e: ValidIf => e map uniquifyExp
        case (_: UIntLiteral | _: SIntLiteral) => e
        case e: DoPrim => e map uniquifyExp
      }

      def uniquifyStmt(s: Statement): Statement = {
        s map uniquifyStmt map uniquifyExp match {
          case sx: DefWire =>
            sinfo = sx.info
            if (nameMap.contains(sx.name)) {
              val node = nameMap(sx.name)
              val newType = uniquifyNamesType(sx.tpe, node.elts)
              (Utils.create_exps(sx.name, sx.tpe) zip Utils.create_exps(node.name, newType)) foreach { 
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
              (Utils.create_exps(sx.name, sx.tpe) zip Utils.create_exps(node.name, newType)) foreach {
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
              val newType = portTypeMap(m.name)
              (Utils.create_exps(sx.name, sx.tpe) zip Utils.create_exps(node.name, newType)) foreach {
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
              (Utils.create_exps(sx.name, memType(sx)) zip Utils.create_exps(node.name, memType(mem))) foreach {
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
              (Utils.create_exps(sx.name, s.asInstanceOf[DefNode].value.tpe) zip Utils.create_exps(node.name, sx.value.tpe)) foreach {
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
        case m: Module =>
          // Adds port names to namespace and namemap
          nameMap ++= portNameMap(m.name)
          namespace ++= create_exps("", portTypeMap(m.name)) map
                        LowerTypes.loweredName map (_.tail)
          m.copy(body = uniquifyBody(m.body) )
      }
    }

    def uniquifyPorts(renames: RenameMap)(m: DefModule): DefModule = {
      renames.setModule(m.name)
      def uniquifyPorts(ports: Seq[Port]): Seq[Port] = {
        val portsType = BundleType(ports map {
          case Port(_, name, dir, tpe) => Field(name, to_flip(dir), tpe)
        })
        val uniquePortsType = uniquifyNames(portsType, collection.mutable.HashSet())
        val localMap = createNameMapping(portsType, uniquePortsType)
        portNameMap += (m.name -> localMap)
        portTypeMap += (m.name -> uniquePortsType)

        ports zip uniquePortsType.fields map { case (p, f) =>
          (Utils.create_exps(p.name, p.tpe) zip Utils.create_exps(f.name, f.tpe)) foreach {
            case (from, to) => renames.rename(from.serialize, to.serialize)
          }
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
    val result = Circuit(c.info, c.modules map uniquifyPorts(renames) map uniquifyModule(renames), c.main)
    CircuitState(result, outputForm, state.annotations, Some(renames))
  }
}

