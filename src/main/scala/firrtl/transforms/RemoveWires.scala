// See LICENSE for license details.

package firrtl
package transforms

import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.traversals.Foreachers._
import firrtl.WrappedExpression._
import firrtl.graph.{MutableDiGraph, CyclicException}
import firrtl.options.{Dependency, PreservesAll}

import scala.collection.mutable
import scala.util.{Try, Success, Failure}

/** Replace wires with nodes in a legal, flow-forward order
  *
  *  This pass must run after LowerTypes because Aggregate-type
  *  wires have multiple connections that may be impossible to order in a
  *  flow-foward way
  */
class RemoveWires extends Transform with PreservesAll[Transform] {
  def inputForm = LowForm
  def outputForm = LowForm

  override val prerequisites = firrtl.stage.Forms.MidForm ++
    Seq( Dependency(passes.LowerTypes),
         Dependency(passes.Legalize),
         Dependency(transforms.RemoveReset),
         Dependency[transforms.CheckCombLoops] )

  override val optionalPrerequisites = Seq(Dependency[checks.CheckResets])

  override val dependents = Seq.empty

  // Extract all expressions that are references to a Node, Wire, or Reg
  // Since we are operating on LowForm, they can only be WRefs
  private def extractNodeWireRegRefs(expr: Expression): Seq[WRef] = {
    val refs = mutable.ArrayBuffer.empty[WRef]
    def rec(e: Expression): Expression = {
      e match {
        case ref @ WRef(_,_, WireKind | NodeKind | RegKind, _) => refs += ref
        case nested @ (_: Mux | _: DoPrim | _: ValidIf) => nested.foreach(rec)
        case _ => // Do nothing
      }
      e
    }
    rec(expr)
    refs
  }

  // Transform netlist into DefNodes
  private def getOrderedNodes(
    netlist: mutable.LinkedHashMap[WrappedExpression, (Seq[Expression], Info)],
    regInfo: mutable.Map[WrappedExpression, DefRegister]): Try[Seq[Statement]] = {
    val digraph = new MutableDiGraph[WrappedExpression]
    for ((sink, (exprs, _)) <- netlist) {
      digraph.addVertex(sink)
      for (expr <- exprs) {
        for (source <- extractNodeWireRegRefs(expr)) {
          digraph.addPairWithEdge(sink, source)
        }
      }
    }

    // We could reverse edge directions and not have to do this reverse, but doing it this way does
    // a MUCH better job of preserving the logic order as expressed by the designer
    // See RemoveWireTests for illustration
    Try {
      val ordered = digraph.linearize.reverse
      ordered.map { key =>
        val WRef(name, _, kind, _) = key.e1
        kind match {
          case RegKind => regInfo(key)
          case WireKind | NodeKind =>
            val (Seq(rhs), info) = netlist(key)
            DefNode(info, name, rhs)
        }
      }
    }
  }

  private def onModule(m: DefModule): DefModule = {
    // Store all non-node declarations here (like reg, inst, and mem)
    val decls = mutable.ArrayBuffer.empty[Statement]
    // Store all "other" statements here, non-wire, non-node connections, printfs, etc.
    val otherStmts = mutable.ArrayBuffer.empty[Statement]
    // Add nodes and wire connection here
    val netlist = mutable.LinkedHashMap.empty[WrappedExpression, (Seq[Expression], Info)]
    // Info at definition of wires for combining into node
    val wireInfo = mutable.HashMap.empty[WrappedExpression, Info]
    // Additional info about registers
    val regInfo = mutable.HashMap.empty[WrappedExpression, DefRegister]

    def onStmt(stmt: Statement): Statement = {
      stmt match {
        case node: DefNode =>
          netlist(we(WRef(node))) = (Seq(node.value), node.info)
        case wire: DefWire if !wire.tpe.isInstanceOf[AnalogType] => // Remove all non-Analog wires
          wireInfo(WRef(wire)) = wire.info
        case reg: DefRegister =>
          val resetDep = reg.reset.tpe match {
            case AsyncResetType => Some(reg.reset)
            case _ => None
          }
          val initDep = Some(reg.init).filter(we(WRef(reg)) != we(_)) // Dependency exists IF reg doesn't init itself
          regInfo(we(WRef(reg))) = reg
          netlist(we(WRef(reg))) = (Seq(reg.clock) ++ resetDep ++ initDep, reg.info)
        case decl: IsDeclaration => // Keep all declarations except for nodes and non-Analog wires
          decls += decl
        case con @ Connect(cinfo, lhs, rhs) => kind(lhs) match {
          case WireKind =>
            // Be sure to pad the rhs since nodes get their type from the rhs
            val paddedRhs = ConstantPropagation.pad(rhs, lhs.tpe)
            val dinfo = wireInfo(lhs)
            netlist(we(lhs)) = (Seq(paddedRhs), MultiInfo(dinfo, cinfo))
          case _ => otherStmts += con // Other connections just pass through
        }
        case invalid @ IsInvalid(info, expr) =>
          kind(expr) match {
            case WireKind =>
              val width = expr.tpe match { case GroundType(width) => width } // LowFirrtl
              netlist(we(expr)) = (Seq(ValidIf(Utils.zero, UIntLiteral(BigInt(0), width), expr.tpe)), info)
            case _ => otherStmts += invalid
          }
        case other @ (_: Print | _: Stop | _: Attach) =>
          otherStmts += other
        case EmptyStmt => // Dont bother keeping EmptyStmts around
        case block: Block => block.foreach(onStmt)
        case _ => throwInternalError()
      }
      stmt
    }

    m match {
      case mod @ Module(info, name, ports, body) =>
        onStmt(body)
        getOrderedNodes(netlist, regInfo) match {
          case Success(logic) =>
            Module(info, name, ports, Block(decls ++ logic ++ otherStmts))
          // If we hit a CyclicException, just abort removing wires
          case Failure(c: CyclicException) =>
            val problematicNode = c.node
            logger.warn(s"Cycle found in module $name, " +
              s"wires will not be removed which can prevent optimizations! Problem node: $problematicNode")
            mod
          case Failure(other) => throw other
        }
      case m: ExtModule => m
    }
  }

  /* @todo move ResolveKinds outside */
  private val cleanup = Seq(
    passes.ResolveKinds
  )

  def execute(state: CircuitState): CircuitState = {
    val result = state.copy(circuit = state.circuit.map(onModule))
    cleanup.foldLeft(result) { case (in, xform) => xform.execute(in) }
  }
}
