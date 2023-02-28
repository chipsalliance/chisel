// SPDX-License-Identifier: Apache-2.0

package firrtl.transforms

import firrtl._
import firrtl.ir._

/** Adds default names to print, stop and verification statements if their name is empty. */
object EnsureNamedStatements extends Transform with DependencyAPIMigration {
  override def invalidates(a: Transform) = false

  override protected def execute(state: CircuitState): CircuitState = {
    val c = state.circuit.mapModule(onModule)
    state.copy(circuit = c)
  }

  private def onModule(m: DefModule): DefModule = m match {
    case e:   ExtModule => e
    case mod: Module =>
      val namespace = Namespace(mod)
      // Ensure we always start with _0 suffix
      val prefixes = Seq("cover", "assert", "assume", "print", "stop")
      prefixes.filterNot(namespace.contains).foreach(namespace.newName)
      mod.mapStmt(onStmt(namespace))
  }

  private def onStmt(namespace: Namespace)(stmt: Statement): Statement = stmt match {
    case s: Print if s.name.isEmpty => s.withName(namespace.newName("print"))
    case s: Stop if s.name.isEmpty => s.withName(namespace.newName("stop"))
    case s: Verification if s.name.isEmpty =>
      val baseName = s.op match {
        case Formal.Cover  => "cover"
        case Formal.Assert => "assert"
        case Formal.Assume => "assume"
      }
      s.withName(namespace.newName(baseName))
    case other => other.mapStmt(onStmt(namespace))
  }
}
