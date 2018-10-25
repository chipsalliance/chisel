// See LICENSE for license details.

package firrtl
package transforms

import firrtl._
import firrtl.annotations._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.PrimOps._
import firrtl.graph.DiGraph
import firrtl.WrappedExpression.weq
import firrtl.analyses.InstanceGraph
import firrtl.annotations.TargetToken.Ref

import annotation.tailrec
import collection.mutable

object ConstantPropagation {
  private def asUInt(e: Expression, t: Type) = DoPrim(AsUInt, Seq(e), Seq(), t)

  /** Pads e to the width of t */
  def pad(e: Expression, t: Type) = (bitWidth(e.tpe), bitWidth(t)) match {
    case (we, wt) if we < wt => DoPrim(Pad, Seq(e), Seq(wt), t)
    case (we, wt) if we == wt => e
  }

  def constPropBitExtract(e: DoPrim) = {
    val arg = e.args.head
    val (hi, lo) = e.op match {
      case Bits => (e.consts.head.toInt, e.consts(1).toInt)
      case Tail => ((bitWidth(arg.tpe) - 1 - e.consts.head).toInt, 0)
      case Head => ((bitWidth(arg.tpe) - 1).toInt, (bitWidth(arg.tpe) - e.consts.head).toInt)
    }

    arg match {
      case lit: Literal =>
        require(hi >= lo)
        UIntLiteral((lit.value >> lo) & ((BigInt(1) << (hi - lo + 1)) - 1), getWidth(e.tpe))
      case x if bitWidth(e.tpe) == bitWidth(x.tpe) => x.tpe match {
        case t: UIntType => x
        case _ => asUInt(x, e.tpe)
      }
      case _ => e
    }
  }
}

class ConstantPropagation extends Transform with ResolvedAnnotationPaths {
  import ConstantPropagation._
  def inputForm = LowForm
  def outputForm = LowForm

  override val annotationClasses: Traversable[Class[_]] = Seq(classOf[DontTouchAnnotation])

  trait FoldCommutativeOp {
    def fold(c1: Literal, c2: Literal): Expression
    def simplify(e: Expression, lhs: Literal, rhs: Expression): Expression

    def apply(e: DoPrim): Expression = (e.args.head, e.args(1)) match {
      case (lhs: Literal, rhs: Literal) => fold(lhs, rhs)
      case (lhs: Literal, rhs) => pad(simplify(e, lhs, rhs), e.tpe)
      case (lhs, rhs: Literal) => pad(simplify(e, rhs, lhs), e.tpe)
      case _ => e
    }
  }

  object FoldADD extends FoldCommutativeOp {
    def fold(c1: Literal, c2: Literal) = (c1, c2) match {
      case (_: UIntLiteral, _: UIntLiteral) => UIntLiteral(c1.value + c2.value, (c1.width max c2.width) + IntWidth(1))
      case (_: SIntLiteral, _: SIntLiteral) => SIntLiteral(c1.value + c2.value, (c1.width max c2.width) + IntWidth(1))
    }
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, w) if v == BigInt(0) => rhs
      case SIntLiteral(v, w) if v == BigInt(0) => rhs
      case _ => e
    }
  }

  object FoldAND extends FoldCommutativeOp {
    def fold(c1: Literal, c2: Literal) = UIntLiteral(c1.value & c2.value, c1.width max c2.width)
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, w) if v == BigInt(0) => UIntLiteral(0, w)
      case SIntLiteral(v, w) if v == BigInt(0) => UIntLiteral(0, w)
      case UIntLiteral(v, IntWidth(w)) if v == (BigInt(1) << bitWidth(rhs.tpe).toInt) - 1 => rhs
      case _ => e
    }
  }

  object FoldOR extends FoldCommutativeOp {
    def fold(c1: Literal, c2: Literal) = UIntLiteral(c1.value | c2.value, c1.width max c2.width)
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, _) if v == BigInt(0) => rhs
      case SIntLiteral(v, _) if v == BigInt(0) => asUInt(rhs, e.tpe)
      case UIntLiteral(v, IntWidth(w)) if v == (BigInt(1) << bitWidth(rhs.tpe).toInt) - 1 => lhs
      case _ => e
    }
  }

  object FoldXOR extends FoldCommutativeOp {
    def fold(c1: Literal, c2: Literal) = UIntLiteral(c1.value ^ c2.value, c1.width max c2.width)
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, _) if v == BigInt(0) => rhs
      case SIntLiteral(v, _) if v == BigInt(0) => asUInt(rhs, e.tpe)
      case _ => e
    }
  }

  object FoldEqual extends FoldCommutativeOp {
    def fold(c1: Literal, c2: Literal) = UIntLiteral(if (c1.value == c2.value) 1 else 0, IntWidth(1))
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, IntWidth(w)) if v == BigInt(1) && w == BigInt(1) && bitWidth(rhs.tpe) == BigInt(1) => rhs
      case _ => e
    }
  }

  object FoldNotEqual extends FoldCommutativeOp {
    def fold(c1: Literal, c2: Literal) = UIntLiteral(if (c1.value != c2.value) 1 else 0, IntWidth(1))
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, IntWidth(w)) if v == BigInt(0) && w == BigInt(1) && bitWidth(rhs.tpe) == BigInt(1) => rhs
      case _ => e
    }
  }

  private def foldConcat(e: DoPrim) = (e.args.head, e.args(1)) match {
    case (UIntLiteral(xv, IntWidth(xw)), UIntLiteral(yv, IntWidth(yw))) => UIntLiteral(xv << yw.toInt | yv, IntWidth(xw + yw))
    case _ => e
  }

  private def foldShiftLeft(e: DoPrim) = e.consts.head.toInt match {
    case 0 => e.args.head
    case x => e.args.head match {
      case UIntLiteral(v, IntWidth(w)) => UIntLiteral(v << x, IntWidth(w + x))
      case SIntLiteral(v, IntWidth(w)) => SIntLiteral(v << x, IntWidth(w + x))
      case _ => e
    }
  }

  private def foldShiftRight(e: DoPrim) = e.consts.head.toInt match {
    case 0 => e.args.head
    case x => e.args.head match {
      // TODO when amount >= x.width, return a zero-width wire
      case UIntLiteral(v, IntWidth(w)) => UIntLiteral(v >> x, IntWidth((w - x) max 1))
      // take sign bit if shift amount is larger than arg width
      case SIntLiteral(v, IntWidth(w)) => SIntLiteral(v >> x, IntWidth((w - x) max 1))
      case _ => e
    }
  }

  private def foldComparison(e: DoPrim) = {
    def foldIfZeroedArg(x: Expression): Expression = {
      def isUInt(e: Expression): Boolean = e.tpe match {
        case UIntType(_) => true
        case _ => false
      }
      def isZero(e: Expression) = e match {
          case UIntLiteral(value, _) => value == BigInt(0)
          case SIntLiteral(value, _) => value == BigInt(0)
          case _ => false
        }
      x match {
        case DoPrim(Lt,  Seq(a,b),_,_) if isUInt(a) && isZero(b) => zero
        case DoPrim(Leq, Seq(a,b),_,_) if isZero(a) && isUInt(b) => one
        case DoPrim(Gt,  Seq(a,b),_,_) if isZero(a) && isUInt(b) => zero
        case DoPrim(Geq, Seq(a,b),_,_) if isUInt(a) && isZero(b) => one
        case ex => ex
      }
    }

    def foldIfOutsideRange(x: Expression): Expression = {
      //Note, only abides by a partial ordering
      case class Range(min: BigInt, max: BigInt) {
        def === (that: Range) =
          Seq(this.min, this.max, that.min, that.max)
            .sliding(2,1)
            .map(x => x.head == x(1))
            .reduce(_ && _)
        def > (that: Range) = this.min > that.max
        def >= (that: Range) = this.min >= that.max
        def < (that: Range) = this.max < that.min
        def <= (that: Range) = this.max <= that.min
      }
      def range(e: Expression): Range = e match {
        case UIntLiteral(value, _) => Range(value, value)
        case SIntLiteral(value, _) => Range(value, value)
        case _ => e.tpe match {
          case SIntType(IntWidth(width)) => Range(
            min = BigInt(0) - BigInt(2).pow(width.toInt - 1),
            max = BigInt(2).pow(width.toInt - 1) - BigInt(1)
          )
          case UIntType(IntWidth(width)) => Range(
            min = BigInt(0),
            max = BigInt(2).pow(width.toInt) - BigInt(1)
          )
        }
      }
      // Calculates an expression's range of values
      x match {
        case ex: DoPrim =>
          def r0 = range(ex.args.head)
          def r1 = range(ex.args(1))
          ex.op match {
            // Always true
            case Lt  if r0 < r1 => one
            case Leq if r0 <= r1 => one
            case Gt  if r0 > r1 => one
            case Geq if r0 >= r1 => one
            // Always false
            case Lt  if r0 >= r1 => zero
            case Leq if r0 > r1 => zero
            case Gt  if r0 <= r1 => zero
            case Geq if r0 < r1 => zero
            case _ => ex
          }
        case ex => ex
      }
    }
    foldIfZeroedArg(foldIfOutsideRange(e))
  }

  private def constPropPrim(e: DoPrim): Expression = e.op match {
    case Shl => foldShiftLeft(e)
    case Shr => foldShiftRight(e)
    case Cat => foldConcat(e)
    case Add => FoldADD(e)
    case And => FoldAND(e)
    case Or => FoldOR(e)
    case Xor => FoldXOR(e)
    case Eq => FoldEqual(e)
    case Neq => FoldNotEqual(e)
    case (Lt | Leq | Gt | Geq) => foldComparison(e)
    case Not => e.args.head match {
      case UIntLiteral(v, IntWidth(w)) => UIntLiteral(v ^ ((BigInt(1) << w.toInt) - 1), IntWidth(w))
      case _ => e
    }
    case AsUInt => e.args.head match {
      case SIntLiteral(v, IntWidth(w)) => UIntLiteral(v + (if (v < 0) BigInt(1) << w.toInt else 0), IntWidth(w))
      case u: UIntLiteral => u
      case _ => e
    }
    case AsSInt => e.args.head match {
      case UIntLiteral(v, IntWidth(w)) => SIntLiteral(v - ((v >> (w.toInt-1)) << w.toInt), IntWidth(w))
      case s: SIntLiteral => s
      case _ => e
    }
    case Pad => e.args.head match {
      case UIntLiteral(v, IntWidth(w)) => UIntLiteral(v, IntWidth(e.consts.head max w))
      case SIntLiteral(v, IntWidth(w)) => SIntLiteral(v, IntWidth(e.consts.head max w))
      case _ if bitWidth(e.args.head.tpe) >= e.consts.head => e.args.head
      case _ => e
    }
    case (Bits | Head | Tail) => constPropBitExtract(e)
    case _ => e
  }

  private def constPropMuxCond(m: Mux) = m.cond match {
    case UIntLiteral(c, _) => pad(if (c == BigInt(1)) m.tval else m.fval, m.tpe)
    case _ => m
  }

  private def constPropMux(m: Mux): Expression = (m.tval, m.fval) match {
    case _ if m.tval == m.fval => m.tval
    case (t: UIntLiteral, f: UIntLiteral) =>
      if (t.value == BigInt(1) && f.value == BigInt(0) && bitWidth(m.tpe) == BigInt(1)) m.cond
      else constPropMuxCond(m)
    case _ => constPropMuxCond(m)
  }

  private def constPropNodeRef(r: WRef, e: Expression) = e match {
    case _: UIntLiteral | _: SIntLiteral | _: WRef => e
    case _ => r
  }

  // Is "a" a "better name" than "b"?
  private def betterName(a: String, b: String): Boolean = (a.head != '_') && (b.head == '_')

  def optimize(e: Expression): Expression = constPropExpression(new NodeMap(), Map.empty[String, String], Map.empty[String, Map[String, Literal]])(e)
  def optimize(e: Expression, nodeMap: NodeMap): Expression = constPropExpression(nodeMap, Map.empty[String, String], Map.empty[String, Map[String, Literal]])(e)
  private def constPropExpression(nodeMap: NodeMap, instMap: Map[String, String], constSubOutputs: Map[String, Map[String, Literal]])(e: Expression): Expression = {
    val old = e map constPropExpression(nodeMap, instMap, constSubOutputs)
    val propagated = old match {
      case p: DoPrim => constPropPrim(p)
      case m: Mux => constPropMux(m)
      case ref @ WRef(rname, _,_, MALE) if nodeMap.contains(rname) =>
        constPropNodeRef(ref, nodeMap(rname))
      case ref @ WSubField(WRef(inst, _, InstanceKind, _), pname, _, MALE) =>
        val module = instMap(inst)
        // Check constSubOutputs to see if the submodule is driving a constant
        constSubOutputs.get(module).flatMap(_.get(pname)).getOrElse(ref)
      case x => x
    }
    propagated
  }

  /** Constant propagate a Module
    *
    * Two pass process
    * 1. Propagate constants in expressions and forward propagate references
    * 2. Propagate references again for backwards reference (Wires)
    * TODO Replacing all wires with nodes makes the second pass unnecessary
    *   However, preserving decent names DOES require a second pass
    *   Replacing all wires with nodes makes it unnecessary for preserving decent names to trigger an
    *   extra iteration though
    *
    * @param m the Module to run constant propagation on
    * @param dontTouches names of components local to m that should not be propagated across
    * @param instMap map of instance names to Module name
    * @param constInputs map of names of m's input ports to literal driving it (if applicable)
    * @param constSubOutputs Map of Module name to Map of output port name to literal driving it
    * @return (Constpropped Module, Map of output port names to literal value,
    *   Map of submodule modulenames to Map of input port names to literal values)
    */
  @tailrec
  private def constPropModule(
      m: Module,
      dontTouches: Set[String],
      instMap: Map[String, String],
      constInputs: Map[String, Literal],
      constSubOutputs: Map[String, Map[String, Literal]]
    ): (Module, Map[String, Literal], Map[String, Map[String, Seq[Literal]]]) = {

    var nPropagated = 0L
    val nodeMap = new NodeMap()
    // For cases where we are trying to constprop a bad name over a good one, we swap their names
    // during the second pass
    val swapMap = mutable.HashMap.empty[String, String]
    // Keep track of any outputs we drive with a constant
    val constOutputs = mutable.HashMap.empty[String, Literal]
    // Keep track of any submodule inputs we drive with a constant
    // (can have more than 1 of the same submodule)
    val constSubInputs = mutable.HashMap.empty[String, mutable.HashMap[String, Seq[Literal]]]

    // Copy constant mapping for constant inputs (except ones marked dontTouch!)
    nodeMap ++= constInputs.filterNot { case (pname, _) => dontTouches.contains(pname) }

    // Note that on back propagation we *only* worry about swapping names and propagating references
    // to constant wires, we don't need to worry about propagating primops or muxes since we'll do
    // that on the next iteration if necessary
    def backPropExpr(expr: Expression): Expression = {
      val old = expr map backPropExpr
      val propagated = old match {
        // When swapping, we swap both rhs and lhs
        case ref @ WRef(rname, _,_,_) if swapMap.contains(rname) =>
          ref.copy(name = swapMap(rname))
        // Only const prop on the rhs
        case ref @ WRef(rname, _,_, MALE) if nodeMap.contains(rname) =>
          constPropNodeRef(ref, nodeMap(rname))
        case x => x
      }
      if (old ne propagated) {
        nPropagated += 1
      }
      propagated
    }

    def backPropStmt(stmt: Statement): Statement = stmt map backPropExpr match {
      case decl: IsDeclaration if swapMap.contains(decl.name) =>
        val newName = swapMap(decl.name)
        nPropagated += 1
        decl match {
          case node: DefNode => node.copy(name = newName)
          case wire: DefWire => wire.copy(name = newName)
          case reg: DefRegister => reg.copy(name = newName)
          case other => throwInternalError()
        }
      case other => other map backPropStmt
    }


    // When propagating a reference, check if we want to keep the name that would be deleted
    def propagateRef(lname: String, value: Expression): Unit = {
      value match {
        case WRef(rname,_,kind,_) if betterName(lname, rname) && !swapMap.contains(rname) && kind != PortKind =>
          assert(!swapMap.contains(lname)) // <- Shouldn't be possible because lname is either a
          // node declaration or the single connection to a wire or register
          swapMap += (lname -> rname, rname -> lname)
        case _ =>
      }
      nodeMap(lname) = value
    }

    def constPropStmt(s: Statement): Statement = {
      val stmtx = s map constPropStmt map constPropExpression(nodeMap, instMap, constSubOutputs)
      // Record things that should be propagated
      stmtx match {
        case x: DefNode if !dontTouches.contains(x.name) => propagateRef(x.name, x.value)
        case Connect(_, WRef(wname, wtpe, WireKind, _), expr: Literal) if !dontTouches.contains(wname) =>
          val exprx = constPropExpression(nodeMap, instMap, constSubOutputs)(pad(expr, wtpe))
          propagateRef(wname, exprx)
        // Record constants driving outputs
        case Connect(_, WRef(pname, ptpe, PortKind, _), lit: Literal) if !dontTouches.contains(pname) =>
          val paddedLit = constPropExpression(nodeMap, instMap, constSubOutputs)(pad(lit, ptpe)).asInstanceOf[Literal]
          constOutputs(pname) = paddedLit
        // Const prop registers that are fed only a constant or a mux between and constant and the
        // register itself
        // This requires that reset has been made explicit
        case Connect(_, lref @ WRef(lname, ltpe, RegKind, _), expr) if !dontTouches.contains(lname) => expr match {
          case lit: Literal =>
            nodeMap(lname) = constPropExpression(nodeMap, instMap, constSubOutputs)(pad(lit, ltpe))
          case Mux(_, tval: WRef, fval: Literal, _) if weq(lref, tval) =>
            nodeMap(lname) = constPropExpression(nodeMap, instMap, constSubOutputs)(pad(fval, ltpe))
          case Mux(_, tval: Literal, fval: WRef, _) if weq(lref, fval) =>
            nodeMap(lname) = constPropExpression(nodeMap, instMap, constSubOutputs)(pad(tval, ltpe))
          case WRef(`lname`, _,_,_) => // If a register is connected to itself, propagate zero
            val zero = passes.RemoveValidIf.getGroundZero(ltpe)
            nodeMap(lname) = constPropExpression(nodeMap, instMap, constSubOutputs)(pad(zero, ltpe))
          case _ =>
        }
        // Mark instance inputs connected to a constant
        case Connect(_, lref @ WSubField(WRef(inst, _, InstanceKind, _), port, ptpe, _), lit: Literal) =>
          val paddedLit = constPropExpression(nodeMap, instMap, constSubOutputs)(pad(lit, ptpe)).asInstanceOf[Literal]
          val module = instMap(inst)
          val portsMap = constSubInputs.getOrElseUpdate(module, mutable.HashMap.empty)
          portsMap(port) = paddedLit +: portsMap.getOrElse(port, List.empty)
        case _ =>
      }
      // Actually transform some statements
      stmtx match {
        // Propagate connections to references
        case Connect(info, lhs, rref @ WRef(rname, _, NodeKind, _)) if !dontTouches.contains(rname) =>
          Connect(info, lhs, nodeMap(rname))
        // If an Attach has at least 1 port, any wires are redundant and can be removed
        case Attach(info, exprs) if exprs.exists(kind(_) == PortKind) =>
          Attach(info, exprs.filterNot(kind(_) == WireKind))
        case other => other
      }
    }

    val modx = m.copy(body = backPropStmt(constPropStmt(m.body)))

    // When we call this function again, constOutputs and constSubInputs are reconstructed and
    // strictly a superset of the versions here
    if (nPropagated > 0) constPropModule(modx, dontTouches, instMap, constInputs, constSubOutputs)
    else (modx, constOutputs.toMap, constSubInputs.mapValues(_.toMap).toMap)
  }

  // Unify two maps using f to combine values of duplicate keys
  private def unify[K, V](a: Map[K, V], b: Map[K, V])(f: (V, V) => V): Map[K, V] =
    b.foldLeft(a) { case (acc, (k, v)) =>
      acc + (k -> acc.get(k).map(f(_, v)).getOrElse(v))
    }

  private def run(c: Circuit, dontTouchMap: Map[String, Set[String]]): Circuit = {
    val iGraph = (new InstanceGraph(c)).graph
    val moduleDeps = iGraph.getEdgeMap.map({ case (mod, children) =>
      mod.module -> children.map(i => i.name -> i.module).toMap
    })

    // This is a *relative* instance count, ie. how many there are when you visit each Module once
    // (even if it is instantiated multiple times)
    val instCount: Map[String, Int] = iGraph.getEdgeMap.foldLeft(Map(c.main -> 1)) {
      case (cs, (_, values)) => values.foldLeft(cs) {
        case (counts, value) => counts.updated(value.module, counts.getOrElse(value.module, 0) + 1)
      }
    }

    // DiGraph using Module names as nodes, destination of edge is a parent Module
    val parentGraph: DiGraph[String] = iGraph.reverse.transformNodes(_.module)

    // This outer loop works by applying constant propagation to the modules in a topologically
    // sorted order from leaf to root
    // Modules will register any outputs they drive with a constant in constOutputs which is then
    // checked by later modules in the same iteration (since we iterate from leaf to root)
    // Since Modules can be instantiated multiple times, for inputs we must check that all instances
    // are driven with the same constant value. Then, if we find a Module input where each instance
    // is driven with the same constant (and not seen in a previous iteration), we iterate again
    @tailrec
    def iterate(toVisit: Set[String],
            modules: Map[String, Module],
            constInputs: Map[String, Map[String, Literal]]): Map[String, DefModule] = {
      if (toVisit.isEmpty) modules
      else {
        // Order from leaf modules to root so that any module driving an output
        // with a constant will be visible to modules that instantiate it
        // TODO Generating order as we execute constant propagation on each module would be faster
        val order = parentGraph.subgraph(toVisit).linearize
        // Execute constant propagation on each module in order
        // Aggreagte Module outputs that are driven constant for use by instaniating Modules
        // Aggregate submodule inputs driven constant for checking later
        val (modulesx, _, constInputsx) =
          order.foldLeft((modules,
                          Map[String, Map[String, Literal]](),
                          Map[String, Map[String, Seq[Literal]]]())) {
            case ((mmap, constOutputs, constInputsAcc), mname) =>
              val dontTouches = dontTouchMap.getOrElse(mname, Set.empty)
              val (mx, mco, mci) = constPropModule(modules(mname), dontTouches, moduleDeps(mname),
                                                   constInputs.getOrElse(mname, Map.empty), constOutputs)
              // Accumulate all Literals used to drive a particular Module port
              val constInputsx = unify(constInputsAcc, mci)((a, b) => unify(a, b)((c, d) => c ++ d))
              (mmap + (mname -> mx), constOutputs + (mname -> mco), constInputsx)
          }
        // Determine which module inputs have all of the same, new constants driving them
        val newProppedInputs = constInputsx.flatMap { case (mname, ports) =>
          val portsx = ports.flatMap { case (pname, lits) =>
            val newPort = !constInputs.get(mname).map(_.contains(pname)).getOrElse(false)
            val isModule = modules.contains(mname) // ExtModules are not contained in modules
            val allSameConst = lits.size == instCount(mname) && lits.toSet.size == 1
            if (isModule && newPort && allSameConst) Some(pname -> lits.head)
            else None
          }
          if (portsx.nonEmpty) Some(mname -> portsx) else None
        }
        val modsWithConstInputs = newProppedInputs.keySet
        val newToVisit = modsWithConstInputs ++
                         modsWithConstInputs.flatMap(parentGraph.reachableFrom)
        // Combine const inputs (there can't be duplicate values in the inner maps)
        val nextConstInputs = unify(constInputs, newProppedInputs)((a, b) => a ++ b)
        iterate(newToVisit.toSet, modulesx, nextConstInputs)
      }
    }

    val modulesx = {
      val nameMap = c.modules.collect { case m: Module => m.name -> m }.toMap
      // We only pass names of Modules, we can't apply const prop to ExtModules
      val mmap = iterate(nameMap.keySet, nameMap, Map.empty)
      c.modules.map(m => mmap.getOrElse(m.name, m))
    }


    Circuit(c.info, modulesx, c.main)
  }

  def execute(state: CircuitState): CircuitState = {
    val dontTouches: Seq[(String, String)] = state.annotations.collect {
      case DontTouchAnnotation(Target(_, Some(m), Seq(Ref(c)))) => m -> c
    }
    // Map from module name to component names
    val dontTouchMap: Map[String, Set[String]] =
      dontTouches.groupBy(_._1).mapValues(_.map(_._2).toSet)

    state.copy(circuit = run(state.circuit, dontTouchMap))
  }
}
