// See LICENSE for license details.

package firrtl.stage.transforms

import firrtl.{CircuitState, Transform, VerilogEmitter}
import firrtl.options.DependencyManagerUtils.CharSet
import firrtl.stage.TransformManager

class Compiler(
  targets: Seq[TransformManager.TransformDependency],
  currentState: Seq[TransformManager.TransformDependency] = Seq.empty,
  knownObjects: Set[Transform] = Set.empty) extends TransformManager(targets, currentState, knownObjects) {

  override val wrappers = Seq(
    (a: Transform) => CatchCustomTransformExceptions(a),
    (a: Transform) => UpdateAnnotations(a)
  )

  override def customPrintHandling(
    tab: String,
    charSet: CharSet,
    size: Int): Option[PartialFunction[(Transform, Int), Seq[String]]] = {

    val (l, n, c) = (charSet.lastNode, charSet.notLastNode, charSet.continuation)
    val last = size - 1

    val f: PartialFunction[(Transform, Int), Seq[String]] = {
      {
        case (a: VerilogEmitter, `last`) =>
          val firstTransforms = a.transforms.dropRight(1)
          val lastTransform = a.transforms.last
          Seq(s"$tab$l ${a.name}") ++
            firstTransforms.map(t => s"""$tab${" " * c.size} $n ${t.name}""") :+
            s"""$tab${" " * c.size} $l ${lastTransform.name}"""
        case (a: VerilogEmitter, _) =>
          val firstTransforms = a.transforms.dropRight(1)
          val lastTransform = a.transforms.last
          Seq(s"$tab$n ${a.name}") ++
            firstTransforms.map(t => s"""$tab$c $n ${t.name}""") :+
            s"""$tab$c $l ${lastTransform.name}"""
      }
    }

    Some(f)
  }

}
