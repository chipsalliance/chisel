// See LICENSE for license details.

package firrtl.annotations.transforms

import firrtl._
import firrtl.annotations.{CircuitTarget, ModuleTarget, MultiTargetAnnotation, ReferenceTarget, SingleTargetAnnotation}
import firrtl.ir
import firrtl.options.{Dependency, PreservesAll}
import firrtl.traversals.Foreachers._

import scala.collection.immutable.{Set => ISet}

/** Replaces all [[firrtl.annotations.ReferenceTarget ReferenceTargets]] pointing at instances with
  * [[firrtl.annotations.InstanceTarget InstanceTargets]].
  *
  * @note This exists because of [[firrtl.annotations.Named Named]] where a [[firrtl.annotations.ComponentName
  * ComponentName]] is the only way to refer to an instance, but this is resolved incorrectly to a
  * [[firrtl.annotations.ReferenceTarget ReferenceTarget]].
  */
class CleanupNamedTargets extends Transform with DependencyAPIMigration {

  override def prerequisites = Seq(Dependency(passes.RemoveCHIRRTL))

  override def optionalPrerequisites = Seq.empty

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Transform) = false

  private def onStatement(statement: ir.Statement)
                         (implicit references: ISet[ReferenceTarget],
                          renameMap: RenameMap,
                          module: ModuleTarget): Unit = statement match {
    case ir.DefInstance(_, a, b, _) if references(module.instOf(a, b).asReference) =>
      renameMap.record(module.instOf(a, b).asReference, module.instOf(a, b))
    case a => statement.foreach(onStatement)
  }

  private def onModule(module: ir.DefModule)
                      (implicit references: ISet[ReferenceTarget],
                       renameMap: RenameMap,
                       circuit: CircuitTarget): Unit = {
    implicit val mTarget = circuit.module(module.name)
    module.foreach(onStatement)
  }

  override protected def execute(state: CircuitState): CircuitState = {

    implicit val rTargets: ISet[ReferenceTarget] = state.annotations.flatMap {
      case a: SingleTargetAnnotation[_] => Some(a.target)
      case a: MultiTargetAnnotation     => a.targets.flatten
      case _                            => None
    }.collect {
      case a: ReferenceTarget => a
    }.toSet

    implicit val renameMap = RenameMap()

    implicit val cTarget = CircuitTarget(state.circuit.main)

    state.circuit.foreach(onModule)

    state.copy(renames = Some(renameMap))
  }

}
