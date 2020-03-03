// See LICENSE for license details.

package chisel3.util.experimental

import chisel3.Bits
import chisel3.experimental.{ChiselAnnotation, RunFirrtlTransform, annotate}
import chisel3.internal.firrtl.Width
import firrtl.passes.{InferTypes, PadWidths, Pass}
import firrtl.annotations.{Annotation, CircuitTarget, ModuleTarget, ReferenceTarget, SingleTargetAnnotation}
import firrtl.{ResolvedAnnotationPaths, Transform}

import scala.collection.mutable


/** Use to overprovision the width of a module port
  *
  * Regardless of the width inferred, this utility will increase the port width to the desired width
  * Any references to the port are properly trimmed to not modify circuit behavior
  * Any assignments to the port are properly padded to the overprovisioned width
  *
  * If the port is deleted (e.g. by dead code elimination), nothing is overprovisioned
  * If you do not want your port deleted, use [[chisel3.dontTouch]] to mark the signal
  *
  * If the same signal is overprovisioned multiple times, the largest width wins
  *
  * This generates errors on the following cases:
  * - overprovision width is smaller than the existing width
  * - port is not a UInt or SInt
  * - port is not an IO
  */
object overprovision {
  def apply(port: Bits, width: Width): Unit = {
    require(width.known, s"You can only overprovision to a known width!")
    annotate(
      new ChiselAnnotation with RunFirrtlTransform {
        override def transformClass: Class[_ <: Transform] = classOf[OverprovisionWidths]
        override def toFirrtl: Annotation = OverprovisionWidthAnnotation(port.toTarget, width.get)
      }
    )
  }
}

/** Annotation to track module ports to be overprovisioned
  * @param target module port to overprovision
  * @param width size of overprovisioning
  */
case class OverprovisionWidthAnnotation(target: ReferenceTarget,
                                        width: Int) extends SingleTargetAnnotation[ReferenceTarget] {
  override def duplicate(n: ReferenceTarget): Annotation = OverprovisionWidthAnnotation(n, width)
}

/** Container of useful functions for overprovisioning widths */
object OverprovisionWidths {
  import firrtl._
  import firrtl.ir._
  import firrtl.Mappers._
  import firrtl.traversals.Foreachers._

  /** Contains a sequence of [[OverprovisionedWidthException]] to be thrown at once
    * @param errors errors to be thrown
    */
  case class MultipleOverprovisionedWidthException(errors: Seq[OverprovisionedWidthException]
                                                  ) extends FirrtlUserException(
    errors.map(_.getMessage).mkString("\n")
  )

  /** An exception indicating a problem overprovisioning a port width
    * @param info Source file location of the port
    * @param target Signal to overprovision
    * @param overWidth Size of the overprovisioning
    * @param reason Explanation of the error
    */
  case class OverprovisionedWidthException(info: Info,
                                           target: ReferenceTarget,
                                           overWidth: Int,
                                           reason: String) extends firrtl.FirrtlUserException(
    s"[${info.serialize}] Overprovisioned width of $overWidth cannot be applied to ${target.prettyPrint()}: $reason"
  )

  /** Overprovisions the width of the provided type, or returns an exception
    * @param info Source file location of the type declaration
    * @param target Signal to overprovision
    * @param t type of the signal
    * @param overWidth size of the overprovisioning
    * @return either the type with overprovisioned width, or an exception
    */
  def overprovisionType(info: Info,
                        target: ReferenceTarget,
                        t: Type,
                        overWidth: Int): Either[GroundType, OverprovisionedWidthException] = {
    t match {
      case UIntType(IntWidth(int)) if int <= overWidth =>
        Left(UIntType(IntWidth(overWidth)))
      case SIntType(IntWidth(int)) if int <= overWidth =>
        Left(SIntType(IntWidth(overWidth)))
      case GroundType(IntWidth(int)) =>
        Right(OverprovisionedWidthException(info, target, overWidth, s"given width $int is too large"))
      case GroundType(UnknownWidth) =>
        Right(OverprovisionedWidthException(info, target, overWidth, s"signal has unknown width"))
      case other =>
        Right(OverprovisionedWidthException(info, target, overWidth, s"signal has incorrect type of ${other.serialize}"))
    }
  }

  /** Indicates whether this expression should be overprovisioned
    * @param ports Maps port names to desired overprovisioned width
    * @param instancePorts Maps (instanceName, portName) to desired overprovisioned width
    * @param e expression to analyze
    * @return Some(overWidth) if expression should be overprovisioned, None otherwise
    */
  def isOverprovisioned(ports: Map[String, Int],
                        instancePorts: Map[(String, String), Int]
                       )(e: Expression): Option[Int] = {
    e match {
      case WRef(name, _, _, _) => ports.get(name)
      case WSubField(WRef(inst, _, _, _), name, _, _) => instancePorts.get((inst, name))
      case other => None
    }
  }

  /** Trims an expression to be the size of the smaller width
    * @param smallerWidth width to trim to, expected to be smaller than the width of expr
    * @param expr a ground typed expression to be trimmed
    * @return the trimmed expression
    */
  def trim(smallerWidth: BigInt, expr: Expression): Expression = {
      val trimmed = PrimOps.set_primop_type(DoPrim(PrimOps.Bits, Seq(expr), Seq(smallerWidth, 0), UnknownType))
      expr.tpe match {
        case _: SIntType => PrimOps.set_primop_type(DoPrim(PrimOps.AsSInt, Seq(trimmed), Nil, UnknownType))
        case _: UIntType => trimmed
      }
  }

  /** Builds a map from instance port to desired overprovisioned width
    * @param module module to build the instance port map for
    * @param modulePorts map of (moduleName -> (map of (portName -> overprovisioned width)))
    * @return Maps (instanceName, portName) -> overprovisioned width
    */
  def buildInstancePortMap(module: DefModule,
                           modulePorts: Map[String, Map[String, Int]]
                          ): Map[(String, String), Int] = {
    val instancePorts = mutable.HashMap[(String, String), Int]()
    def onStmt(s: Statement): Unit = s match {
      case WDefInstance(_, name, module, _) if modulePorts.contains(module) =>
        modulePorts(module).foreach { case (portName, over) => instancePorts((name, portName)) = over }
      case other => other foreach onStmt
    }
    module foreach onStmt
    instancePorts.toMap
  }

  /** Trims any nested expression which has been overprovisioned
    * @param ports maps portName to overprovisioned width
    * @param instancePorts maps (instanceName, portName) to overprovisioned width
    * @param e expression tree to walk and trim as necessary
    * @return
    */
  def trimExpressions(ports: Map[String, Int], instancePorts: Map[(String, String), Int])(e: Expression): Expression = {
    (isOverprovisioned(ports, instancePorts)(e), Utils.flow(e), e.tpe) match {
      case (None, _, _)     => e map trimExpressions(ports, instancePorts)
      case (_, SinkFlow, _) => e map trimExpressions(ports, instancePorts)
      case (Some(over), SourceFlow, GroundType(IntWidth(orig))) => if(orig == over) e else trim(orig, e)
      case other => firrtl.Utils.throwInternalError(s"Unexpected case in match: $other")
    }
  }

  /** Trims all expressions in a module which have been overprovisioned
    * @param module module to trim
    * @param modulePorts map of (moduleName -> (map of (portName -> overprovisioned width)))
    * @return module with trimmed expressions
    */
  def trimModule(module: DefModule, modulePorts: Map[String, Map[String, Int]]): DefModule = {
    val instancePorts = buildInstancePortMap(module, modulePorts)
    val ports = modulePorts.get(module.name)
    def onStatement(s: Statement): Statement = s map onStatement map trimExpressions(ports.get, instancePorts)
    if(ports.nonEmpty) module map onStatement else module
  }

  /** Overprovisions the ports of the module which are contained in targetWidthMap
    * @note The returned module is in an inconsistent state, where the ports have the new width but the expressions
    *       have not had their types updated accordingly! InferTypes must be run later on.
    * @param m module to overprovision
    * @param circuitTarget target of the circuit (helps build error messages)
    * @param targetWidthMap maps signal targets to desired overprovisioned width
    * @return module which is overprovisioned
    */
  def overprovisionModule(m: DefModule,
                          circuitTarget: CircuitTarget,
                          targetWidthMap: mutable.HashMap[ReferenceTarget, Int]
                         ): (DefModule, List[OverprovisionedWidthException]) = {
    val moduleTarget = circuitTarget.module(m.name)
    val errors = mutable.ListBuffer[OverprovisionedWidthException]()
    // Contains references which have been overprovisioned
    val usedRefs = mutable.HashSet[ReferenceTarget]()

    // Overprovisions the port's width, if contained in targetWidthMap
    def overprovisionPort(moduleTarget: ModuleTarget)(p: Port): Port = {
      val ref = moduleTarget.ref(p.name)
      p match {
        case Port(info, name, _, tpe) if targetWidthMap.contains(ref) =>
          overprovisionType(info, ref, tpe, targetWidthMap(moduleTarget.ref(name)) ) match {
            case Left(newTpe) =>
              usedRefs += ref
              p.copy(tpe = newTpe)
            case Right(error) =>
              errors += error
              p
          }
        case other => other
      }
    }

    val retModule = m map overprovisionPort(moduleTarget)

    // Any target which is not used to overprovision is an error, and is handled accordingly
    errors ++= targetWidthMap.keySet.diff(usedRefs).map(
      ref => OverprovisionedWidthException(NoInfo, ref, targetWidthMap(ref), "signal is not a port in its module")
    )

    (retModule, errors.toList)
  }
}

/** Transform which collects all [[OverprovisionWidthAnnotation]] and modifies the circuit to overprovision widths
  * @throws [[OverprovisionWidths.MultipleOverprovisionedWidthException]]
  * @throws [[OverprovisionWidths.OverprovisionedWidthException]]
  */
class OverprovisionWidths extends Transform with ResolvedAnnotationPaths {
  import firrtl._
  import OverprovisionWidths._

  // Run after all aggregate types are expanded
  override val inputForm = LowForm
  override val outputForm = LowForm

  // Ensures instance-specific annotations are resolved to module-specific annotations
  override val annotationClasses: Traversable[Class[_]] = List(classOf[OverprovisionWidthAnnotation])

  override def execute(state: CircuitState): CircuitState = {

    // Maps module to map of port to overprovisioned width
    val moduleToTargetWidthMap = mutable.HashMap[String, mutable.HashMap[ReferenceTarget, Int]]()

    // Populates moduleToTargetWidthMap and returns remaining annotations
    val remainingAnnotations = state.annotations.flatMap {
      case OverprovisionWidthAnnotation(target, width) =>
        require(width > 0, s"Overprovisioned width must be positive! ${target.prettyPrint()}")
        val targetWidthMap = moduleToTargetWidthMap.getOrElseUpdate(target.module, mutable.HashMap[ReferenceTarget, Int]())
        targetWidthMap(target) = math.max(width, targetWidthMap.getOrElse(target, 0))
        Nil
      case other => Seq(other)
    }

    val circuitTarget = CircuitTarget(state.circuit.main)
    val allErrors = mutable.ListBuffer[OverprovisionedWidthException]()

    // Overprovision all ports of all modules, collect errors
    val overprovisionedModules = state.circuit.modules.map {
      m => moduleToTargetWidthMap.get(m.name).map { targetWidthMap =>
        val (newM, errors) = overprovisionModule(m, circuitTarget, targetWidthMap)
        allErrors ++= errors
        newM
      }.getOrElse(m)
    }

    // Build new map for trimming
    val modulePorts = moduleToTargetWidthMap.map{ case (mname, targetMap) =>
      mname -> targetMap.map { case (ref: ReferenceTarget, i: Int) => ref.ref -> i}.toMap
    }.toMap

    val trimmedModules = overprovisionedModules map { m => trimModule(m, modulePorts) }

    val newCircuit = state.circuit.copy(modules = trimmedModules)

    // Run InferTypes to fix types of references to overprovisioned ports
    // Run PadWidths to pad assignments to overprovisioned ports
    val cleanedCircuit = postTransformCleanup.foldLeft(newCircuit){ (c, pass) => pass.run(c) }

    // Error if any errors found, otherwise return updated CircuitState
    allErrors.toList match {
      case Nil => state.copy(circuit = cleanedCircuit, annotations = remainingAnnotations)
      case List(err) => throw err
      case l => throw new MultipleOverprovisionedWidthException(l)
    }

  }

  val postTransformCleanup: Seq[Pass] = Seq(InferTypes, PadWidths)
}
