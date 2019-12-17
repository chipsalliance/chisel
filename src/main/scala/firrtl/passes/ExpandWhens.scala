// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.PrimOps._
import firrtl.WrappedExpression._
import firrtl.options.Dependency

import annotation.tailrec
import collection.mutable

/** Expand Whens
  *
  * This pass does the following things:
  * $ - Remove last connect semantics
  * $ - Remove conditional blocks
  * $ - Eliminate concept of scoping
  * $ - Consolidate attaches
  *
  * @note Assumes bulk connects and isInvalids have been expanded
  * @note Assumes all references are declared
  */
object ExpandWhens extends Pass {

  override val prerequisites =
    Seq( Dependency(PullMuxes),
         Dependency(ReplaceAccesses),
         Dependency(ExpandConnects),
         Dependency(RemoveAccesses),
         Dependency(Uniquify) ) ++ firrtl.stage.Forms.Resolved

  override def invalidates(a: Transform): Boolean = a match {
    case CheckInitialization | ResolveKinds | InferTypes => true
    case _ => false
  }

  /** Returns circuit with when and last connection semantics resolved */
  def run(c: Circuit): Circuit = {
    val modulesx = c.modules map {
      case m: ExtModule => m
      case m: Module =>
      val (netlist, simlist, attaches, bodyx, sourceInfoMap) = expandWhens(m)
      val attachedAnalogs = attaches.flatMap(_.exprs.map(we)).toSet
      val newBody = Block(Seq(squashEmpty(bodyx)) ++ expandNetlist(netlist, attachedAnalogs, sourceInfoMap) ++
                              combineAttaches(attaches) ++ simlist)
      Module(m.info, m.name, m.ports, newBody)
    }
    Circuit(c.info, modulesx, c.main)
  }

  /** Maps an expression to a declared node name. Used to memoize predicates */
  type NodeMap = mutable.HashMap[MemoizedHash[Expression], String]

  /** Maps a reference to whatever connects to it. Used to resolve last connect semantics */
  type Netlist = mutable.LinkedHashMap[WrappedExpression, Expression]

  /** Collects Info data serialized names for nodes, aggregating into MultiInfo when necessary */
  class InfoMap extends mutable.HashMap[String, Info] {
    override def default(key: String): Info = {
      val x = NoInfo
      this(key) = x
      x
    }
  }

  /** Contains all simulation constructs */
  type Simlist = mutable.ArrayBuffer[Statement]

  /** List of all netlists of each declared scope, ordered from closest to farthest
    * @note Note immutable.Map because conversion from mutable.LinkedHashMap to mutable.Map is VERY slow
    */
  type Defaults = Seq[mutable.Map[WrappedExpression, Expression]]


  /** Expands a module's when statements
    * @param m Module to expand
    * @note Netlist maps a reference to whatever connects to it
    * @note Simlist contains all simulation constructs in m
    * @note Seq[Attach] contains all Attach statements (unsimplified)
    * @note Statement contains all declarations in the module (including DefNode's)
    */
  def expandWhens(m: Module): (Netlist, Simlist, Seq[Attach], Statement, InfoMap) = {
    val namespace = Namespace(m)
    val simlist = new Simlist

    // Memoizes if an expression contains any WVoids inserted in this pass
    val memoizedVoid = new mutable.HashSet[MemoizedHash[Expression]] += WVoid

    // Memoizes the node that holds a particular expression, if any
    val nodes = new NodeMap

    // Seq of attaches in order
    lazy val attaches = mutable.ArrayBuffer.empty[Attach]

    val infoMap: InfoMap = new InfoMap

    /* Adds into into map, aggregates info into MultiInfo where necessary
     * @param key  serialized name of node
     * @param info info being recorded
     */
    def saveInfo(key: String, info: Info): Unit = {
      infoMap(key) = infoMap(key) ++ info
    }

    /* Removes connections/attaches from the statement
     * Mutates namespace, simlist, nodes, attaches
     * Mutates input netlist
     * @param netlist maps references to their values for a given immediate scope
     * @param defaults sequence of netlists of surrouding scopes, ordered closest to farthest
     * @param p predicate so far, used to update simulation constructs
     * @param s statement to expand
     */
    def expandWhens(netlist: Netlist,
                    defaults: Defaults,
                    p: Expression)
                    (s: Statement): Statement = s match {
      // For each non-register declaration, update netlist with value WVoid for each female reference
      // Return self, unchanged
      case stmt @ (_: DefNode | EmptyStmt) => stmt
      case w: DefWire =>
        netlist ++= (getFemaleRefs(w.name, w.tpe, DuplexFlow) map (ref => we(ref) -> WVoid))
        w
      case w: DefMemory =>
        netlist ++= (getFemaleRefs(w.name, MemPortUtils.memType(w), SourceFlow) map (ref => we(ref) -> WVoid))
        w
      case w: WDefInstance =>
        netlist ++= (getFemaleRefs(w.name, w.tpe, SourceFlow).map(ref => we(ref) -> WVoid))
        w
      // Update netlist with self reference for each female reference
      // Return self, unchanged
      case r: DefRegister =>
        netlist ++= (getFemaleRefs(r.name, r.tpe, DuplexFlow) map (ref => we(ref) -> ref))
        r
      // For value assignments, update netlist/attaches and return EmptyStmt
      case c: Connect =>
        saveInfo(c.loc.serialize, c.info)
        netlist(c.loc) = c.expr
        EmptyStmt
      case c: IsInvalid =>
        netlist(c.expr) = WInvalid
        EmptyStmt
      case a: Attach =>
        attaches += a
        EmptyStmt
      // For simulation constructs, update simlist with predicated statement and return EmptyStmt
      case sx: Print =>
        simlist += (if (weq(p, one)) sx else Print(sx.info, sx.string, sx.args, sx.clk, AND(p, sx.en)))
        EmptyStmt
      case sx: Stop =>
        simlist += (if (weq(p, one)) sx else Stop(sx.info, sx.ret, sx.clk, AND(p, sx.en)))
        EmptyStmt
      // Expand conditionally, see comments below
      case sx: Conditionally =>
        /* 1) Recurse into conseq and alt with empty netlist, updated defaults, updated predicate
         * 2) For each assigned reference (lvalue) in either conseq or alt, get merged value
         *   a) Find default value from defaults
         *   b) Create Mux, ValidIf or WInvalid, depending which (or both) conseq/alt assigned lvalue
         * 3) If a merged value has been memoized, update netlist. Otherwise, memoize then update netlist.
         * 4) Return conseq and alt declarations, followed by memoized nodes
         */
        val conseqNetlist = new Netlist
        val altNetlist = new Netlist
        val conseqStmt = expandWhens(conseqNetlist, netlist +: defaults, AND(p, sx.pred))(sx.conseq)
        val altStmt = expandWhens(altNetlist, netlist +: defaults, AND(p, NOT(sx.pred)))(sx.alt)

        // Process combined maps because we only want to create 1 mux for each node
        //   present in the conseq and/or alt
        val memos = (conseqNetlist ++ altNetlist) map { case (lvalue, _) =>
          // Defaults in netlist get priority over those in defaults
          val default = netlist get lvalue match {
            case Some(v) => Some(v)
            case None => getDefault(lvalue, defaults)
          }
          val res = default match {
            case Some(defaultValue) =>
              val trueValue = conseqNetlist getOrElse (lvalue, defaultValue)
              val falseValue = altNetlist getOrElse (lvalue, defaultValue)
              (trueValue, falseValue) match {
                case (WInvalid, WInvalid) => WInvalid
                case (WInvalid, fv) => ValidIf(NOT(sx.pred), fv, fv.tpe)
                case (tv, WInvalid) => ValidIf(sx.pred, tv, tv.tpe)
                case (tv, fv) => Mux(sx.pred, tv, fv, mux_type_and_widths(tv, fv)) //Muxing clocks will be checked during type checking
              }
            case None =>
              // Since not in netlist, lvalue must be declared in EXACTLY one of conseq or alt
              conseqNetlist getOrElse (lvalue, altNetlist(lvalue))
          }

          // Does an expression contain WVoid inserted in this pass?
          def containsVoid(e: Expression): Boolean = e match {
            case WVoid => true
            case ValidIf(_, value, _) => memoizedVoid(value)
            case Mux(_, tv, fv, _) => memoizedVoid(tv) || memoizedVoid(fv)
            case _ => false
          }

          res match {
            // Don't create a node to hold mux trees with void values
            // "Idiomatic" emission of these muxes isn't a concern because they represent bad code (latches)
            case e if containsVoid(e) =>
              netlist(lvalue) = e
              memoizedVoid += e   // remember that this was void
              EmptyStmt
            case _: ValidIf | _: Mux | _: DoPrim => nodes get res match {
              case Some(name) =>
                netlist(lvalue) = WRef(name, res.tpe, NodeKind, SourceFlow)
                EmptyStmt
              case None =>
                val name = namespace.newTemp
                nodes(res) = name
                netlist(lvalue) = WRef(name, res.tpe, NodeKind, SourceFlow)
                DefNode(sx.info, name, res)
            }
            case _ =>
              netlist(lvalue) = res
              EmptyStmt
          }
        }
        Block(Seq(conseqStmt, altStmt) ++ memos)
      case block: Block => block map expandWhens(netlist, defaults, p)
      case _ => throwInternalError()
    }
    val netlist = new Netlist
    // Add ports to netlist
    netlist ++= (m.ports flatMap { case Port(_, name, dir, tpe) =>
      getFemaleRefs(name, tpe, to_flow(dir)) map (ref => we(ref) -> WVoid)
    })
    val bodyx = expandWhens(netlist, Seq(netlist), one)(m.body)
    (netlist, simlist, attaches, bodyx, infoMap)
  }


  /** Returns all references to all Female leaf subcomponents of a reference */
  private def getFemaleRefs(n: String, t: Type, g: Flow): Seq[Expression] = {
    val exps = create_exps(WRef(n, t, ExpKind, g))
    exps.flatMap { case exp =>
      exp.tpe match {
        case AnalogType(w) => None
        case _ => flow(exp) match {
          case (DuplexFlow | SinkFlow) => Some(exp)
          case _ => None
        }
      }
    }
  }

  /** Returns all connections/invalidations in the circuit
    * @todo Preserve Info
    * @note Remove IsInvalids on attached Analog-typed components
    */
  private def expandNetlist(netlist: Netlist, attached: Set[WrappedExpression], sourceInfoMap: InfoMap) =
    netlist map {
      case (k, WInvalid) => // Remove IsInvalids on attached Analog types
        if (attached.contains(k)) EmptyStmt else IsInvalid(NoInfo, k.e1)
      case (k, v) =>
        val info = sourceInfoMap(k.e1.serialize)
        Connect(info, k.e1, v)
    }

  /** Returns new sequence of combined Attaches
    * @todo Preserve Info
    */
  private def combineAttaches(attaches: Seq[Attach]): Seq[Attach] = {
    // Helper type to add an ordering index to attached Expressions
    case class AttachAcc(exprs: Seq[WrappedExpression], idx: Int)
    // Map from every attached expression to its corresponding AttachAcc
    //   (many keys will point to same value)
    val attachMap = mutable.LinkedHashMap.empty[WrappedExpression, AttachAcc]
    for (Attach(_, es) <- attaches) {
      val exprs = es.map(we(_))
      val acc = exprs.map(attachMap.get(_)).flatten match {
        case Seq() => // None of these expressions is present in the attachMap
          AttachAcc(exprs, attachMap.size)
        case accs => // At least one expression present in the attachMap
          val sorted = accs sortBy (_.idx)
          AttachAcc((sorted.map(_.exprs) :+ exprs).flatten.distinct, sorted.head.idx)
      }
      attachMap ++= acc.exprs.map(_ -> acc)
    }
    attachMap.values.toList.distinct.map(acc => Attach(NoInfo, acc.exprs.map(_.e1)))
  }
  // Searches nested scopes of defaults for lvalue
  // defaults uses mutable Map because we are searching LinkedHashMaps and conversion to immutable is VERY slow
  @tailrec
  private def getDefault(lvalue: WrappedExpression, defaults: Defaults): Option[Expression] = {
    defaults match {
      case Nil => None
      case head :: tail => head get lvalue match {
        case Some(p) => Some(p)
        case None => getDefault(lvalue, tail)
      }
    }
  }

  private def AND(e1: Expression, e2: Expression) =
    DoPrim(And, Seq(e1, e2), Nil, BoolType)
  private def NOT(e: Expression) =
    DoPrim(Eq, Seq(e, zero), Nil, BoolType)
}

class ExpandWhensAndCheck extends SeqTransform {

  override val prerequisites =
    Seq( Dependency(PullMuxes),
         Dependency(ReplaceAccesses),
         Dependency(ExpandConnects),
         Dependency(RemoveAccesses),
         Dependency(Uniquify) ) ++ firrtl.stage.Forms.Deduped

  override def invalidates(a: Transform): Boolean = a match {
    case ResolveKinds | InferTypes | ResolveFlows | _: InferWidths => true
    case _ => false
  }

  override def inputForm = UnknownForm
  override def outputForm = UnknownForm

  override val transforms = Seq(ExpandWhens, CheckInitialization)

}
