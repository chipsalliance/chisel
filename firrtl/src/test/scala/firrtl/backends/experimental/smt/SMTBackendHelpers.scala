// SPDX-License-Identifier: Apache-2.0

package firrtl.backends.experimental.smt

import firrtl.annotations.Annotation
import firrtl.backends.experimental.smt.random.{InvalidToRandomPass, UndefinedMemoryBehaviorPass}
import firrtl.options.Dependency
import firrtl.{ir, MemoryInitValue}
import firrtl.stage.{Forms, RunFirrtlTransformAnnotation, TransformManager}

private object SMTBackendHelpers {
  private val dependencies = Forms.LowForm ++ FirrtlToTransitionSystem.prerequisites
  private val compiler = new TransformManager(dependencies)
  private val undefCompiler = new TransformManager(
    dependencies ++ Seq(
      Dependency(InvalidToRandomPass),
      Dependency(UndefinedMemoryBehaviorPass)
    )
  )

  def compile(src: String, annos: Seq[Annotation] = List()): ir.Circuit = {
    val c = firrtl.Parser.parse(src)
    compiler.runTransform(firrtl.CircuitState(c, annos)).circuit
  }

  def compileUndef(src: String, annos: Seq[Annotation] = List()): ir.Circuit = {
    val c = firrtl.Parser.parse(src)
    undefCompiler.runTransform(firrtl.CircuitState(c, annos)).circuit
  }

  def toSys(
    src:        String,
    mod:        String = "m",
    presetRegs: Set[String] = Set(),
    memInit:    Map[String, MemoryInitValue] = Map(),
    modelUndef: Boolean = false
  ): TransitionSystem = {
    val circuit = if (modelUndef) compileUndef(src) else compile(src)
    val module = circuit.modules.find(_.name == mod).get.asInstanceOf[ir.Module]
    // println(module.serialize)
    new ModuleToTransitionSystem(presetRegs = presetRegs, memInit = memInit, uninterpreted = Map()).run(module)
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
