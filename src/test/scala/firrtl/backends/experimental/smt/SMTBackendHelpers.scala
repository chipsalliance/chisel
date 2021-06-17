// SPDX-License-Identifier: Apache-2.0

package firrtl.backends.experimental.smt

import firrtl.annotations.Annotation
import firrtl.{ir, MemoryInitValue}
import firrtl.stage.{Forms, TransformManager}

private object SMTBackendHelpers {
  private val dependencies = Forms.LowForm ++ FirrtlToTransitionSystem.prerequisites
  private val compiler = new TransformManager(dependencies)

  def compile(src: String, annos: Seq[Annotation] = List()): ir.Circuit = {
    val c = firrtl.Parser.parse(src)
    compiler.runTransform(firrtl.CircuitState(c, annos)).circuit
  }

  def toSys(
    src:        String,
    mod:        String = "m",
    presetRegs: Set[String] = Set(),
    memInit:    Map[String, MemoryInitValue] = Map()
  ): TransitionSystem = {
    val circuit = compile(src)
    val module = circuit.modules.find(_.name == mod).get.asInstanceOf[ir.Module]
    // println(module.serialize)
    new ModuleToTransitionSystem().run(module, presetRegs = presetRegs, memInit = memInit)
  }

  def toBotr2(src: String, mod: String = "m"): Iterable[String] =
    Btor2Serializer.serialize(toSys(src, mod))

  def toBotr2Str(src: String, mod: String = "m"): String =
    toBotr2(src, mod).mkString("\n") + "\n"

  def toSMTLib(src: String, mod: String = "m"): Iterable[String] =
    SMTTransitionSystemEncoder.encode(toSys(src, mod)).map(SMTLibSerializer.serialize)

  def toSMTLibStr(src: String, mod: String = "m"): String =
    toSMTLib(src, mod).mkString("\n") + "\n"
}
