// See LICENSE for license details.

package firrtl
package transforms

import firrtl.{Utils}


import firrtl.PrimOps._
import firrtl.ir._
import firrtl.ir.{AsyncResetType}
import firrtl.annotations._
import firrtl.options.{Dependency, PreservesAll}

import scala.collection.mutable

object PropagatePresetAnnotations {
  val advice = "Please Note that a Preset-annotated AsyncReset shall NOT be casted to other types with any of the following functions: asInterval, asUInt, asSInt, asClock, asFixedPoint, asAsyncReset."
  case class TreeCleanUpOrphanException(message: String) extends FirrtlUserException(s"Node left an orphan during tree cleanup: $message $advice")
}

/** Propagate PresetAnnotations to all children of targeted AsyncResets
  * Leaf Registers are annotated with PresetRegAnnotation
  * All wires, nodes and connectors along the way are suppressed
  *
  * Processing of multiples targets are NOT isolated from one another as the expected outcome does not differ
  * Annotations of leaf registers, wires, nodes & connectors does indeed not depend on the initial AsyncReset reference
  * The set of created annotation based on multiple initial AsyncReset PresetAnnotation
  *
  * This transform consists of 2 successive walk of the AST
  * I./ Propagate
  *    - 1./ Create all AsyncResetTrees
  *    - 2./ Leverage them to annotate register for specialized emission & PresetTree for cleanUp
  * II./ CleanUpTree
  *    - clean up all the intermediate nodes (replaced with EmptyStmt)
  *    - raise Error on orphans (typically cast of Annotated Reset)
  *    - disconnect Registers from their reset nodes (replaced with UInt(0))
  *
  * Thanks to the clean-up phase, this transform does not rely on DCE
  *
  * @note This pass must run before InlineCastsTransform
  */
class PropagatePresetAnnotations extends Transform with PreservesAll[Transform] {
  def inputForm = UnknownForm
  def outputForm = UnknownForm

  override val prerequisites = firrtl.stage.Forms.LowFormMinimumOptimized ++
    Seq( Dependency[BlackBoxSourceHelper],
         Dependency[FixAddingNegativeLiterals],
         Dependency[ReplaceTruncatingArithmetic])

  override val optionalPrerequisites = firrtl.stage.Forms.LowFormOptimized

  override val dependents = Seq.empty


  import PropagatePresetAnnotations._

  private type TargetSet = mutable.HashSet[ReferenceTarget]
  private type TargetMap = mutable.HashMap[ReferenceTarget,String]
  private type TargetSetMap = mutable.HashMap[ReferenceTarget, TargetSet]

  private val toCleanUp = new TargetSet()

  /**
    * Logic of the propagation, divided in two main phases:
    * 1./ Walk all the Circuit looking for annotated AsyncResets :
    *     - Store all Annotated AsyncReset reference
    *     - Build all AsyncReset Trees (whether annotated or not)
    *     - Store all async-reset-registers (whether annotated or not)
    * 2./ Walk the AsyncReset Tree based on Annotated AsyncReset as entry points
    *     - Annotate all leaf register with PresetRegAnnotation
    *     - Annotate all intermediate wire, node, connect with PresetConnectorAnnotation
    *
    * @param circuit the circuit
    * @param annotations all the annotations
    * @return updated annotations
    */
  private def propagate(cs: CircuitState, presetAnnos: Seq[PresetAnnotation]): AnnotationSeq = {
    val presets = presetAnnos.groupBy(_.target)
    // store all annotated asyncreset references
    val asyncToAnnotate = new TargetSet()
    // store all async-reset-registers
    val asyncRegMap = new TargetSetMap()
    // store async-reset trees
    val asyncCoMap = new TargetSetMap()
    // Annotations to be appended and returned as result of the transform
    val annos = cs.annotations.to[mutable.ArrayBuffer] -- presetAnnos

    val circuitTarget = CircuitTarget(cs.circuit.main)

    /*
    * WALK I PHASE 1 FUNCTIONS
    */

    /**
      * Walk current module
      *  - process ports
      *     - store connections & entry points for PHASE 2
      *  - process statements
      *     - Instances => record local instances for cross module AsyncReset Tree Buidling
      *     - Registers => store AsyncReset bound registers for PHASE 2
      *     - Wire => store AsyncReset Connections & entry points for PHASE 2
      *     - Connect => store AsyncReset Connections & entry points for PHASE 2
      *
      * @param m module
      */
    def processModule(m: DefModule): Unit = {
      val moduleTarget = circuitTarget.module(m.name)
      val localInstances = new TargetMap()

      /**
        * Recursively process a given type
        * Recursive on Bundle and Vector Type only
        * Store Register and Connections for AsyncResetType
        *
        * @param tpe Type to be processed
        * @param target ReferenceTarget associated to the tpe
        * @param all Boolean indicating whether all subelements of the current tpe should also be stored as Annotated AsyncReset entry points
        */
      def processType(tpe: Type, target: ReferenceTarget, all: Boolean): Unit = {
        if(tpe == AsyncResetType){
          asyncRegMap(target) = new TargetSet()
          asyncCoMap(target) = new TargetSet()
          if (presets.contains(target) || all) {
            asyncToAnnotate += target
          }
        } else {
          tpe match {
            case b: BundleType =>
              b.fields.foreach{
                (x: Field) =>
                  val tar = target.field(x.name)
                  processType(x.tpe, tar, (presets.contains(tar) || all))
              }

            case v: VectorType =>
              for(i <- 0 until v.size) {
                val tar = target.index(i)
                processType(v.tpe, tar, (presets.contains(tar) || all))
              }
            case _ =>
          }
        }
      }

      def processWire(w: DefWire): Unit = {
        val target = moduleTarget.ref(w.name)
        processType(w.tpe, target, presets.contains(target))
      }

      /**
        * Recursively search for the ReferenceTarget of a given Expression
        *
        * @param e Targeted Expression
        * @param ta Local ReferenceTarget of the Targeted Expression
        * @return a ReferenceTarget in case of success, a GenericTarget otherwise
        * @throw Internal Error on unexpected recursive path return results
        */
      def getRef(e: Expression, ta: ReferenceTarget, annoCo: Boolean = false) : Target = {
        e match {
          case w: WRef => moduleTarget.ref(w.name)
          case w: WSubField =>
            getRef(w.expr, ta, annoCo) match {
              case rt: ReferenceTarget =>
                if(localInstances.contains(rt)){
                  val remote_ref =  circuitTarget.module(localInstances(rt))
                  if (annoCo)
                    asyncCoMap(ta) += rt.field(w.name)
                  remote_ref.ref(w.name)
                } else {
                  rt.field(w.name)
                }
              case remote_target => remote_target
             }
          case w: WSubIndex =>
            getRef(w.expr, ta, annoCo) match {
              case remote_target: ReferenceTarget =>
                if (annoCo)
                  asyncCoMap(ta) += remote_target
                remote_target.index(w.value)
              case _ => Utils.throwInternalError("Unexpected Reference kind")
            }

          case _ => Target(None, None, Seq.empty)
        }
      }

      def processRegister(r: DefRegister): Unit = {
        getRef(r.reset, moduleTarget.ref(r.name), false) match {
          case rt : ReferenceTarget =>
            if (asyncRegMap.contains(rt)) {
              asyncRegMap(rt) += moduleTarget.ref(r.name)
            }
          case _ =>
        }

      }

      def processConnect(c: Connect): Unit = {
        getRef(c.expr, ReferenceTarget("","", Seq.empty, "", Seq.empty)) match {
          case rhs: ReferenceTarget =>
            if (presets.contains(rhs) || asyncRegMap.contains(rhs)) {
              getRef(c.loc, rhs, true) match {
                case lhs : ReferenceTarget =>
                  if(asyncRegMap.contains(rhs)){
                    asyncRegMap(rhs) += lhs
                  } else {
                    asyncToAnnotate += lhs
                  }
                case _ => //
              }
            }
          case rhs: GenericTarget => //nothing to do
          case _ => Utils.throwInternalError("Unexpected Reference kind")
        }
      }

      def processNode(n: DefNode): Unit = {
        val target = moduleTarget.ref(n.name)
        processType(n.value.tpe, target, presets.contains(target))

        getRef(n.value, ReferenceTarget("","", Seq.empty, "", Seq.empty)) match {
          case rhs: ReferenceTarget =>
            if (presets.contains(rhs) || asyncRegMap.contains(rhs)) {
              if(asyncRegMap.contains(rhs)){
                asyncRegMap(rhs) += target
              } else {
                asyncToAnnotate += target
              }
            }
          case rhs: GenericTarget => //nothing to do
          case _ => Utils.throwInternalError("Unexpected Reference kind")
        }
      }

      def processStatements(statement: Statement): Unit = {
        statement match {
          case i : WDefInstance =>
            localInstances(moduleTarget.ref(i.name)) = i.module
          case r : DefRegister => processRegister(r)
          case w : DefWire     => processWire(w)
          case n : DefNode     => processNode(n)
          case c : Connect     => processConnect(c)
          case s               => s.foreachStmt(processStatements)
        }
      }

      def processPorts(port: Port): Unit = {
        if(port.tpe == AsyncResetType){
          val target = moduleTarget.ref(port.name)
          asyncRegMap(target) = new TargetSet()
          asyncCoMap(target) = new TargetSet()
          if (presets.contains(target)) {
            asyncToAnnotate += target
            toCleanUp += target
          }
        }
      }

      m match {
        case module: firrtl.ir.Module =>
          module.foreachPort(processPorts)
          processStatements(module.body)
        case _ =>
      }
    }

    /*
     * WALK I PHASE 2 FUNCTIONS
     */

    /** Annotate a given target and all its children according to the asyncCoMap */
    def annotateCo(ta: ReferenceTarget){
      if (asyncCoMap.contains(ta)){
        toCleanUp += ta
        asyncCoMap(ta) foreach( (t: ReferenceTarget) => {
          toCleanUp += t
        })
      }
    }

    /** Annotate all registers somehow connected to the orignal annotated async reset */
    def annotateRegSet(set: TargetSet) : Unit = {
      set foreach ( (ta: ReferenceTarget) => {
        annotateCo(ta)
        if (asyncRegMap.contains(ta)) {
          annotateRegSet(asyncRegMap(ta))
        } else {
          annos += new PresetRegAnnotation(ta)
        }
      })
    }

    /**
      * Walk AsyncReset Trees with all Annotated AsyncReset as entry points
      * Annotate all leaf registers and intermediate wires, nodes, connectors along the way
      */
    def annotateAsyncSet(set: TargetSet) : Unit = {
      set foreach ((t: ReferenceTarget) => {
        annotateCo(t)
        if (asyncRegMap.contains(t))
          annotateRegSet(asyncRegMap(t))
      })
    }

    /*
     * MAIN
     */

    cs.circuit.foreachModule(processModule) // PHASE 1 : Initialize
    annotateAsyncSet(asyncToAnnotate)       // PHASE 2 : Annotate
    annos
  }

  /*
   * WALK II FUNCTIONS
   */

  /**
    * Clean-up useless reset tree (not relying on DCE)
    * Disconnect preset registers from their reset tree
    */
  private def cleanUpPresetTree(circuit: Circuit, annos: AnnotationSeq) : Circuit = {
    val presetRegs = annos.collect {case a : PresetRegAnnotation => a}.groupBy(_.target)
    val circuitTarget = CircuitTarget(circuit.main)

    def processModule(m: DefModule): DefModule = {
      val moduleTarget = circuitTarget.module(m.name)
      val localInstances = new TargetMap()

      def getRef(e: Expression) : Target = {
        e match {
          case w: WRef => moduleTarget.ref(w.name)
          case w: WSubField =>
            getRef(w.expr) match {
              case rt: ReferenceTarget =>
                if(localInstances.contains(rt)){
                  circuitTarget.module(localInstances(rt)).ref(w.name)
                } else {
                  rt.field(w.name)
                }
              case remote_target => remote_target
            }
          case w: WSubIndex =>
            getRef(w.expr) match {
              case remote_target: ReferenceTarget => remote_target.index(w.value)
              case _ => Utils.throwInternalError("Unexpected Reference kind")
            }
          case DoPrim(op, args, _, _) =>
            op match {
              case AsInterval | AsUInt | AsSInt | AsClock | AsFixedPoint | AsAsyncReset => getRef(args.head)
              case _ => Target(None, None, Seq.empty)
            }
          case _ => Target(None, None, Seq.empty)
        }
      }


      def processRegister(r: DefRegister) : DefRegister = {
        if (presetRegs.contains(moduleTarget.ref(r.name))) {
          r.copy(reset = UIntLiteral(0))
        } else {
          r
        }
      }

      def processWire(w: DefWire) : Statement = {
        if (toCleanUp.contains(moduleTarget.ref(w.name))) {
          EmptyStmt
        } else {
          w
        }
      }

      def processNode(n: DefNode) : Statement = {
        if (toCleanUp.contains(moduleTarget.ref(n.name))) {
          EmptyStmt
        } else {
          getRef(n.value) match {
            case rt : ReferenceTarget if(toCleanUp.contains(rt)) =>
              throw TreeCleanUpOrphanException(s"Orphan (${moduleTarget.ref(n.name)}) the way.")
            case _ => n
          }
        }
      }

      def processConnect(c: Connect): Statement = {
        getRef(c.expr) match {
          case rhs: ReferenceTarget if (toCleanUp.contains(rhs)) =>
            getRef(c.loc) match {
              case lhs : ReferenceTarget if(!toCleanUp.contains(lhs)) =>
                throw TreeCleanUpOrphanException(s"Orphan ${lhs} connected deleted node $rhs.")
              case _ => EmptyStmt
            }
          case _ => c
        }
      }

      def processInstance(i: WDefInstance) : WDefInstance = {
        localInstances(moduleTarget.ref(i.name)) = i.module
        val tpe = i.tpe match {
          case b: BundleType =>
            val inst = moduleTarget.instOf(i.name, i.module).asReference
            BundleType(b.fields.filterNot(p => toCleanUp.contains(inst.field(p.name))))
          case other => other
        }
        i.copy(tpe = tpe)
      }

      def processStatements(statement: Statement): Statement = {
        statement match {
          case i : WDefInstance => processInstance(i)
          case r : DefRegister  => processRegister(r)
          case w : DefWire      => processWire(w)
          case n : DefNode      => processNode(n)
          case c : Connect      => processConnect(c)
          case s                => s.mapStmt(processStatements)
        }
      }

      m match {
        case module: firrtl.ir.Module =>
          val ports = module.ports.filterNot(p => toCleanUp.contains(moduleTarget.ref(p.name)))
          module.copy(body = processStatements(module.body), ports = ports)
        case _ => m
      }
    }
    circuit.mapModule(processModule)
  }

  def execute(state: CircuitState): CircuitState = {
    // Collect all user-defined PresetAnnotation
    val presets = state.annotations
      .collect{ case m : PresetAnnotation => m }

    // No PresetAnnotation => no need to walk the IR
    if (presets.size == 0){
      state
    } else {
      // PHASE I - Propagate
      val annos = propagate(state, presets)
      // PHASE II - CleanUp
      val cleanCircuit = cleanUpPresetTree(state.circuit, annos)
      // Because toCleanup is a class field, we need to clear it
      // TODO refactor so that toCleanup is not a class field
      toCleanUp.clear()
      state.copy(annotations = annos, circuit = cleanCircuit)
    }
  }
}
