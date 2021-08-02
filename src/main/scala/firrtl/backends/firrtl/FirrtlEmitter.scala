package firrtl

import java.io.Writer
import firrtl.Utils._
import firrtl.ir._
import firrtl.stage.TransformManager.TransformDependency
import firrtl.traversals.Foreachers._

import scala.collection.mutable

sealed abstract class FirrtlEmitter(form: Seq[TransformDependency], val outputSuffix: String)
    extends Transform
    with Emitter
    with DependencyAPIMigration {
  override def prerequisites = form
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Transform) = false

  private def emitAllModules(circuit: Circuit): Seq[EmittedFirrtlModule] = {
    // For a given module, returns a Seq of all modules instantited inside of it
    def collectInstantiatedModules(mod: Module, map: Map[String, DefModule]): Seq[DefModule] = {
      // Use list instead of set to maintain order
      val modules = mutable.ArrayBuffer.empty[DefModule]
      def onStmt(stmt: Statement): Unit = stmt match {
        case DefInstance(_, _, name, _) => modules += map(name)
        case _: WDefInstanceConnector => throwInternalError(s"unrecognized statement: $stmt")
        case other => other.foreach(onStmt)
      }
      onStmt(mod.body)
      modules.distinct.toSeq
    }
    val modMap = circuit.modules.map(m => m.name -> m).toMap
    // Turn each module into it's own circuit with it as the top and all instantied modules as ExtModules
    circuit.modules.collect {
      case m: Module =>
        val instModules = collectInstantiatedModules(m, modMap)
        val extModules = instModules.map {
          case Module(info, name, ports, _) => ExtModule(info, name, ports, name, Seq.empty)
          case ext: ExtModule => ext
        }
        val newCircuit = Circuit(m.info, extModules :+ m, m.name)
        EmittedFirrtlModule(m.name, newCircuit.serialize, outputSuffix)
    }
  }

  override def execute(state: CircuitState): CircuitState = {
    val newAnnos = state.annotations.flatMap {
      case EmitCircuitAnnotation(a) if this.getClass == a =>
        Seq(
          EmittedFirrtlCircuitAnnotation(
            EmittedFirrtlCircuit(state.circuit.main, state.circuit.serialize, outputSuffix)
          )
        )
      case EmitAllModulesAnnotation(a) if this.getClass == a =>
        emitAllModules(state.circuit).map(EmittedFirrtlModuleAnnotation(_))
      case _ => Seq()
    }
    state.copy(annotations = newAnnos ++ state.annotations)
  }

  // Old style, deprecated
  def emit(state: CircuitState, writer: Writer): Unit = writer.write(state.circuit.serialize)
}

class ChirrtlEmitter extends FirrtlEmitter(stage.Forms.ChirrtlForm, ".fir")
class MinimumHighFirrtlEmitter extends FirrtlEmitter(stage.Forms.MinimalHighForm, ".mhi.fir")
class HighFirrtlEmitter extends FirrtlEmitter(stage.Forms.HighForm, ".hi.fir")
class MiddleFirrtlEmitter extends FirrtlEmitter(stage.Forms.MidForm, ".mid.fir")
class LowFirrtlEmitter extends FirrtlEmitter(stage.Forms.LowForm, ".lo.fir")
object LowFirrtlOptimizedEmitter extends FirrtlEmitter(stage.Forms.LowFormOptimized, ".opt.lo.fir")
