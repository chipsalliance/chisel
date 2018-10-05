// See LICENSE for license details.

package firrtl.passes

import com.typesafe.scalalogging.LazyLogging
import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.PrimOps._
import firrtl.transforms.ConstantPropagation
import firrtl.annotations.{Named, CircuitName, ModuleName, ComponentName}
import firrtl.analyses.InstanceGraph

import scala.collection.mutable

/** [[Pass]] is simple transform that is generally part of a larger [[Transform]]
  * Has an [[UnknownForm]], because larger [[Transform]] should specify form
  */
trait Pass extends Transform {
  def inputForm: CircuitForm = UnknownForm
  def outputForm: CircuitForm = UnknownForm
  def run(c: Circuit): Circuit
  def execute(state: CircuitState): CircuitState = {
    val result = (state.form, inputForm) match {
      case (_, UnknownForm) => run(state.circuit)
      case (UnknownForm, _) => run(state.circuit)
      case (x, y) if x > y =>
        error(s"[$name]: Input form must be lower or equal to $inputForm. Got ${state.form}")
      case _ => run(state.circuit)
    }
    CircuitState(result, outputForm, state.annotations, state.renames)
  }
}

// Error handling
class PassException(message: String) extends Exception(message)
class PassExceptions(exceptions: Seq[PassException]) extends Exception("\n" + exceptions.mkString("\n"))
class Errors {
  val errors = collection.mutable.ArrayBuffer[PassException]()
  def append(pe: PassException) = errors.append(pe)
  def trigger() = errors.size match {
    case 0 =>
    case 1 => throw errors.head
    case _ =>
      append(new PassException(s"${errors.length} errors detected!"))
      throw new PassExceptions(errors)
  }
}

// These should be distributed into separate files
object ToWorkingIR extends Pass {
  def toExp(e: Expression): Expression = e map toExp match {
    case ex: Reference => WRef(ex.name, ex.tpe, NodeKind, UNKNOWNGENDER)
    case ex: SubField => WSubField(ex.expr, ex.name, ex.tpe, UNKNOWNGENDER)
    case ex: SubIndex => WSubIndex(ex.expr, ex.value, ex.tpe, UNKNOWNGENDER)
    case ex: SubAccess => WSubAccess(ex.expr, ex.index, ex.tpe, UNKNOWNGENDER)
    case ex => ex // This might look like a case to use case _ => e, DO NOT!
  }

  def toStmt(s: Statement): Statement = s map toExp match {
    case sx: DefInstance => WDefInstance(sx.info, sx.name, sx.module, UnknownType)
    case sx => sx map toStmt
  }

  def run (c:Circuit): Circuit =
    c copy (modules = c.modules map (_ map toStmt))
}

object PullMuxes extends Pass {
   def run(c: Circuit): Circuit = {
     def pull_muxes_e(e: Expression): Expression = e map pull_muxes_e match {
       case ex: WSubField => ex.expr match {
         case exx: Mux => Mux(exx.cond,
           WSubField(exx.tval, ex.name, ex.tpe, ex.gender),
           WSubField(exx.fval, ex.name, ex.tpe, ex.gender), ex.tpe)
         case exx: ValidIf => ValidIf(exx.cond,
           WSubField(exx.value, ex.name, ex.tpe, ex.gender), ex.tpe)
         case _ => ex  // case exx => exx causes failed tests
       }
       case ex: WSubIndex => ex.expr match {
         case exx: Mux => Mux(exx.cond,
           WSubIndex(exx.tval, ex.value, ex.tpe, ex.gender),
           WSubIndex(exx.fval, ex.value, ex.tpe, ex.gender), ex.tpe)
         case exx: ValidIf => ValidIf(exx.cond,
           WSubIndex(exx.value, ex.value, ex.tpe, ex.gender), ex.tpe)
         case _ => ex  // case exx => exx causes failed tests
       }
       case ex: WSubAccess => ex.expr match {
         case exx: Mux => Mux(exx.cond,
           WSubAccess(exx.tval, ex.index, ex.tpe, ex.gender),
           WSubAccess(exx.fval, ex.index, ex.tpe, ex.gender), ex.tpe)
         case exx: ValidIf => ValidIf(exx.cond,
           WSubAccess(exx.value, ex.index, ex.tpe, ex.gender), ex.tpe)
         case _ => ex  // case exx => exx causes failed tests
       }
       case ex => ex
     }
     def pull_muxes(s: Statement): Statement = s map pull_muxes map pull_muxes_e
     val modulesx = c.modules.map {
       case (m:Module) => Module(m.info, m.name, m.ports, pull_muxes(m.body))
       case (m:ExtModule) => m
     }
     Circuit(c.info, modulesx, c.main)
   }
}

object ExpandConnects extends Pass {
  def run(c: Circuit): Circuit = {
    def expand_connects(m: Module): Module = {
      val genders = collection.mutable.LinkedHashMap[String,Gender]()
      def expand_s(s: Statement): Statement = {
        def set_gender(e: Expression): Expression = e map set_gender match {
          case ex: WRef => WRef(ex.name, ex.tpe, ex.kind, genders(ex.name))
          case ex: WSubField =>
            val f = get_field(ex.expr.tpe, ex.name)
            val genderx = times(gender(ex.expr), f.flip)
            WSubField(ex.expr, ex.name, ex.tpe, genderx)
          case ex: WSubIndex => WSubIndex(ex.expr, ex.value, ex.tpe, gender(ex.expr))
          case ex: WSubAccess => WSubAccess(ex.expr, ex.index, ex.tpe, gender(ex.expr))
          case ex => ex
        }
        s match {
          case sx: DefWire => genders(sx.name) = BIGENDER; sx
          case sx: DefRegister => genders(sx.name) = BIGENDER; sx
          case sx: WDefInstance => genders(sx.name) = MALE; sx
          case sx: DefMemory => genders(sx.name) = MALE; sx
          case sx: DefNode => genders(sx.name) = MALE; sx
          case sx: IsInvalid =>
            val invalids = (create_exps(sx.expr) foldLeft Seq[Statement]())(
               (invalids,  expx) => gender(set_gender(expx)) match {
                  case BIGENDER => invalids :+ IsInvalid(sx.info, expx)
                  case FEMALE => invalids :+ IsInvalid(sx.info, expx)
                  case _ => invalids
               }
            )
            invalids.size match {
               case 0 => EmptyStmt
               case 1 => invalids.head
               case _ => Block(invalids)
            }
          case sx: Connect =>
            val locs = create_exps(sx.loc)
            val exps = create_exps(sx.expr)
            Block((locs zip exps).zipWithIndex map {case ((locx, expx), i) =>
               get_flip(sx.loc.tpe, i, Default) match {
                  case Default => Connect(sx.info, locx, expx)
                  case Flip => Connect(sx.info, expx, locx)
               }
            })
          case sx: PartialConnect =>
            val ls = get_valid_points(sx.loc.tpe, sx.expr.tpe, Default, Default)
            val locs = create_exps(sx.loc)
            val exps = create_exps(sx.expr)
            val stmts = ls map { case (x, y) =>
              locs(x).tpe match {
                case AnalogType(_) => Attach(sx.info, Seq(locs(x), exps(y)))
                case _ =>
                  get_flip(sx.loc.tpe, x, Default) match {
                    case Default => Connect(sx.info, locs(x), exps(y))
                    case Flip => Connect(sx.info, exps(y), locs(x))
                  }
              }
            }
            Block(stmts)
          case sx => sx map expand_s
        }
      }

      m.ports.foreach { p => genders(p.name) = to_gender(p.direction) }
      Module(m.info, m.name, m.ports, expand_s(m.body))
    }

    val modulesx = c.modules.map {
       case (m: ExtModule) => m
       case (m: Module) => expand_connects(m)
    }
    Circuit(c.info, modulesx, c.main)
  }
}


// Replace shr by amount >= arg width with 0 for UInts and MSB for SInts
// TODO replace UInt with zero-width wire instead
object Legalize extends Pass {
  private def legalizeShiftRight(e: DoPrim): Expression = {
    require(e.op == Shr)
    val amount = e.consts.head.toInt
    val width = bitWidth(e.args.head.tpe)
    lazy val msb = width - 1
    if (amount >= width) {
      e.tpe match {
        case UIntType(_) => zero
        case SIntType(_) =>
          val bits = DoPrim(Bits, e.args, Seq(msb, msb), BoolType)
          DoPrim(AsSInt, Seq(bits), Seq.empty, SIntType(IntWidth(1)))
        case t => error(s"Unsupported type $t for Primop Shift Right")
      }
    } else {
      e
    }
  }
  private def legalizeBitExtract(expr: DoPrim): Expression = {
    expr.args.head match {
      case _: UIntLiteral | _: SIntLiteral => ConstantPropagation.constPropBitExtract(expr)
      case _ => expr
    }
  }
  private def legalizePad(expr: DoPrim): Expression = expr.args.head match {
    case UIntLiteral(value, IntWidth(width)) if width < expr.consts.head =>
      UIntLiteral(value, IntWidth(expr.consts.head))
    case SIntLiteral(value, IntWidth(width)) if width < expr.consts.head =>
      SIntLiteral(value, IntWidth(expr.consts.head))
    case _ => expr
  }
  private def legalizeConnect(c: Connect): Statement = {
    val t = c.loc.tpe
    val w = bitWidth(t)
    if (w >= bitWidth(c.expr.tpe)) {
      c
    } else {
      val bits = DoPrim(Bits, Seq(c.expr), Seq(w - 1, 0), UIntType(IntWidth(w)))
      val expr = t match {
        case UIntType(_) => bits
        case SIntType(_) => DoPrim(AsSInt, Seq(bits), Seq(), SIntType(IntWidth(w)))
        case FixedType(_, IntWidth(p)) => DoPrim(AsFixedPoint, Seq(bits), Seq(p), t)
      }
      Connect(c.info, c.loc, expr)
    }
  }
  def run (c: Circuit): Circuit = {
    def legalizeE(expr: Expression): Expression = expr map legalizeE match {
      case prim: DoPrim => prim.op match {
        case Shr => legalizeShiftRight(prim)
        case Pad => legalizePad(prim)
        case Bits | Head | Tail => legalizeBitExtract(prim)
        case _ => prim
      }
      case e => e // respect pre-order traversal
    }
    def legalizeS (s: Statement): Statement = {
      val legalizedStmt = s match {
        case c: Connect => legalizeConnect(c)
        case _ => s
      }
      legalizedStmt map legalizeS map legalizeE
    }
    c copy (modules = c.modules map (_ map legalizeS))
  }
}

/** Transform that removes collisions with Verilog keywords
  * @define implicitRename @param renames the [[RenameMap]] to query when renaming
  * @define implicitNamespace @param ns an encolosing [[Namespace]] with which new names must not conflict
  * @define implicitScope @param scope the enclosing scope of this name. If [[None]], then this is a [[Circuit]] name
  */
object VerilogRename extends Transform {
  def inputForm: CircuitForm = LowForm
  def outputForm: CircuitForm = LowForm
  type Renames = mutable.HashMap[String, String]
  private val inlineDelim = "_"

  /** Generate a new name, by appending underscores, that will not conflict with the existing namespace
    * @param n a name
    * @param ns a [[Namespace]]
    * @return a conflict-free name
    * @note prefix uniqueness is not respected
    */
  private def safeName(n: String, ns: Namespace): String =
    Uniquify.findValidPrefix(n + inlineDelim, Seq(""), ns.cloneUnderlying ++ v_keywords)

  /** Modify a name to not conflict with a Verilog keywords while respecting existing renames and a namespace
    * @param n the name to rename
    * @param renames the [[RenameMap]] to query when renaming
    * $implicitRename
    * $implicitNamespace
    * $implicitScope
    * @return a name without keyword conflicts
    */
  private def onName(n: String)(implicit renames: RenameMap, ns: Namespace, scope: Option[Named]): String = {

    // Convert a [[String]] into [[Named]] based on the provided scope.
    def wrap(name: String, scope: Option[Named]): Named = scope match {
      case None                     => CircuitName(name)
      case Some(cir: CircuitName)   => ModuleName(name, cir)
      case Some(mod: ModuleName)    => ComponentName(name, mod)
      case Some(com: ComponentName) => ComponentName(s"${com.name}.$name", com.module)
    }

    val named = wrap(n, scope)

    // If this has already been renamed use that name. If it conflicts with a keyword, determine a new, safe name and
    // update the renames. Otherwise, leave it alone.
    val namedx: Seq[Named] = renames.get(named) match {
      case Some(x) => x
      case None if v_keywords(n) =>
        val sn = wrap(safeName(n, ns), scope)
        renames.rename(named, sn)
        Seq(sn)
      case _ => Seq(wrap(n, scope))
    }

    namedx match {
      case ComponentName(n, _) :: Nil => n
      case ModuleName(n, _)    :: Nil => n
      case CircuitName(n)      :: Nil => n
      case x => throw new PassException(
        s"Verilog renaming shouldn't result in multiple renames, but found '$named -> $namedx'")
    }
  }

  /** Rename the fields of a [[Type]] to match the ports of an instance
    * @param t the type to rename
    * $implicitRename
    * $implicitNamespace
    * $implicitScope
    * @return a [[Type]] with updated names
    * @note This is not intended for fixing arbitrary types, only [[BundleType]] in instance [[WRef]]s
    */
  private def onType(t: Type)
                    (implicit renames: RenameMap, ns: Namespace, scope: Option[ModuleName]): Type = t match {
    case b: BundleType => b.copy(fields = b.fields.map(f => f.copy(name = onName(f.name))))
    case _             => t
  }

  /** Rename an [[Expression]] to respect existing renames and avoid keyword collisions
    * @param e the [[Expression]] to rename
    * $implicitRename
    * $implicitNamespace
    * $implicitScope
    * @return an [[Expression]] without keyword conflicts
    */
  private def onExpression(e: Expression)
                          (implicit renames: RenameMap, ns: Namespace, scope: Option[ModuleName],
                           iToM: mutable.Map[ComponentName, ModuleName]): Expression = e match {
    case wsf@ WSubField(wr@ WRef(name, _, InstanceKind, _), port, _, _) =>
      val subInst = ComponentName(name, scope.get)
      val subModule = iToM(subInst)
      val subPort = ComponentName(port, subModule)

      val wrx = wr.copy(
        name = renames.get(subInst).orElse(Some(Seq(subInst))).get.head.name,
        tpe = onType(wr.tpe)(renames, ns, Some(subModule)))

      wsf.copy(
        expr = wrx,
        name = renames.get(subPort).orElse(Some(Seq(subPort))).get.head.name)
    case wr: WRef => wr.copy(name=onName(wr.name))
    case ex       => ex.map(onExpression)
  }

  /** Rename a [[Statement]] to respect existing renames and avoid keyword collisions
    * $implicitRename
    * $implicitNamespace
    * $implicitScope
    * @return a [[Statement]] without keyword conflicts
    */
  private def onStatement(s: Statement)
                         (implicit renames: RenameMap, ns: Namespace, scope: Option[ModuleName],
                          iToM: mutable.Map[ComponentName, ModuleName]): Statement = s match {
    case wdi: WDefInstance =>
      val subModule = ModuleName(wdi.module, scope.get.circuit)
      val modulex = renames.get(subModule).orElse(Some(Seq(subModule))).get.head.name
      val wdix = wdi.copy(module = modulex,
                          name = onName(wdi.name),
                          tpe = onType(wdi.tpe)(renames, ns, Some(ModuleName(modulex, scope.get.circuit))))
      iToM(ComponentName(wdi.name, scope.get)) = ModuleName(wdix.module, scope.get.circuit)
      wdix
    case _ => s
        .map(onStatement)
        .map(onExpression)
        .map(onName)
  }

  /** Rename a [[Port]] to avoid keyword collisions
    * $implicitRename
    * $implicitNamespace
    * $implicitScope
    * @return a [[Port]] without keyword conflicts
    */
  private def onPort(p: Port)(implicit renames: RenameMap, ns: Namespace, scope: Option[ModuleName]): Port =
    p.copy(name = onName(p.name))

  /** Rename a [[DefModule]] and it's internals (ports and statements) to fix keyword collisions and update instance
    * references to respect previous renames
    * @param renames a [[RenameMap]]
    * @param circuit the enclosing [[CircuitName]]
    * @return a [[DefModule]] without keyword conflicts
    */
  private def onModule(renames: RenameMap, circuit: CircuitName)(m: DefModule): DefModule = {
    implicit val moduleNamespace: Namespace = Namespace(m)
    implicit val scope: Option[ModuleName] = Some(ModuleName(m.name, circuit))
    implicit val r: RenameMap = renames

    // Store local renames of refs to instances to their renamed modules. This is needed when renaming port connections
    // on subfields where only the local instance name is available.
    implicit val iToM: mutable.Map[ComponentName, ModuleName] = mutable.Map.empty

    m
      .map(onPort)
      .map(onStatement)
      .map(onName(_: String)(renames, moduleNamespace, Some(circuit)))
  }

  /** Fix any Verilog keyword collisions in a [[Circuit]]
    * @param c a [[Circuit]] with possible name collisions
    * @param renames a [[RenameMap]] to update. If you don't want to propagate renames, this can be ignored.
    * @return a [[Circuit]] without keyword conflicts
    */
  def run(c: Circuit, renames: RenameMap = RenameMap()): Circuit = {
    implicit val circuitNamespace: Namespace = Namespace(c)
    implicit val scope: Option[CircuitName] = Some(CircuitName(c.main))

    // Rename all modules from leafs to root in one pass while updating a shared rename map. Going from leafs to roots
    // ensures that the rename map is safe for parents to blindly consult.
    val modulesx: Map[ModuleName, Seq[DefModule]] = new InstanceGraph(c).moduleOrder.reverse
      .map(onModule(renames, scope.get))
      .groupBy(m => ModuleName(m.name, scope.get))

    // Reorder the renamed modules into the original circuit order.
    val modulesxx: Seq[DefModule] = c.modules.flatMap{ orig =>
      val named = ModuleName(orig.name, scope.get)
      modulesx(renames.get(named).orElse(Some(Seq(named))).get.head)
    }

    // Rename the circuit if the top module was renamed
    val mainx = renames.get(ModuleName(c.main, CircuitName(c.main))) match {
      case Some(ModuleName(m, _) :: Nil) =>
        renames.rename(CircuitName(c.main), CircuitName(m))
        m
      case x@ Some(car :: cdr) => throw new PassException(
        s"Verilog renaming shouldn't result in multiple renames, but found '${c.main} -> $x'")
      case None =>
        c.main
    }

    // Apply all updates
    c.copy(modules = modulesxx, main = mainx)
  }

  /** Fix any Verilog keyword name collisions in a [[CircuitState]] while propagating renames
    * @param state the [[CircuitState]] with possible name collisions
    * @return a [[CircuitState]] without name collisions
    */
  def execute(state: CircuitState): CircuitState = {
    val renames = RenameMap()
    renames.setCircuit(state.circuit.main)
    state.copy(circuit = run(state.circuit, renames), renames = Some(renames))
  }
}

/** Makes changes to the Firrtl AST to make Verilog emission easier
  *
  * - For each instance, adds wires to connect to each port
  *   - Note that no Namespace is required because Uniquify ensures that there will be no
  *     collisions with the lowered names of instance ports
  * - Also removes Attaches where a single Port OR Wire connects to 1 or more instance ports
  *   - These are expressed in the portCons of WDefInstConnectors
  *
  * @note The result of this pass is NOT legal Firrtl
  */
object VerilogPrep extends Pass {

  type AttachSourceMap = Map[WrappedExpression, Expression]

  // Finds attaches with only a single source (Port or Wire)
  //   - Creates a map of attached expressions to their source
  //   - Removes the Attach
  private def collectAndRemoveAttach(m: DefModule): (DefModule, AttachSourceMap) = {
    val sourceMap = mutable.HashMap.empty[WrappedExpression, Expression]
    lazy val namespace = Namespace(m)

    def onStmt(stmt: Statement): Statement = stmt map onStmt match {
      case attach: Attach =>
        val wires = attach.exprs groupBy kind
        val sources = wires.getOrElse(PortKind, Seq.empty) ++ wires.getOrElse(WireKind, Seq.empty)
        val instPorts = wires.getOrElse(InstanceKind, Seq.empty)
        // Sanity check (Should be caught by CheckTypes)
        assert(sources.size + instPorts.size == attach.exprs.size)

        sources match {
          case Seq() => // Zero sources, can add a wire to connect and remove
            val name = namespace.newTemp
            val wire = DefWire(NoInfo, name, instPorts.head.tpe)
            val ref = WRef(wire)
            for (inst <- instPorts) sourceMap(inst) = ref
            wire // Replace the attach with new source wire definition
          case Seq(source) => // One source can be removed
            assert(!sourceMap.contains(source)) // should have been merged
            for (inst <- instPorts) sourceMap(inst) = source
            EmptyStmt
          case moreThanOne =>
            attach
        }
      case s => s
    }

    (m map onStmt, sourceMap.toMap)
  }

  def run(c: Circuit): Circuit = {
    def lowerE(e: Expression): Expression = e match {
      case (_: WRef | _: WSubField) if kind(e) == InstanceKind =>
        WRef(LowerTypes.loweredName(e), e.tpe, kind(e), gender(e))
      case _ => e map lowerE
    }

    def lowerS(attachMap: AttachSourceMap)(s: Statement): Statement = s match {
      case WDefInstance(info, name, module, tpe) =>
        val portRefs = create_exps(WRef(name, tpe, ExpKind, MALE))
        val (portCons, wires) = portRefs.map { p =>
          attachMap.get(p) match {
            // If it has a source in attachMap use that
            case Some(ref) => (p -> ref, None)
            // If no source, create a wire corresponding to the port and connect it up
            case None =>
              val wire = DefWire(info, LowerTypes.loweredName(p), p.tpe)
              (p -> WRef(wire), Some(wire))
          }
        }.unzip
        val newInst = WDefInstanceConnector(info, name, module, tpe, portCons)
        Block(wires.flatten :+ newInst)
      case other => other map lowerS(attachMap) map lowerE
    }

    val modulesx = c.modules map { mod =>
      val (modx, attachMap) = collectAndRemoveAttach(mod)
      modx map lowerS(attachMap)
    }
    c.copy(modules = modulesx)
  }
}
