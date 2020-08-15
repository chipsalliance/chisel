// See LICENSE for license details.

package firrtl.backends.experimental.smt

import firrtl.annotations.Annotation
import firrtl.{MemoryInitValue, ir}
import firrtl.stage.{Forms, TransformManager}
import org.scalatest.flatspec.AnyFlatSpec

private abstract class SMTBackendBaseSpec extends AnyFlatSpec {
  private val dependencies = Forms.LowForm
  private val compiler = new TransformManager(dependencies)

  protected def compile(src: String, annos: Seq[Annotation] = List()): ir.Circuit = {
    val c = firrtl.Parser.parse(src)
    compiler.runTransform(firrtl.CircuitState(c, annos)).circuit
  }

  protected def toSys(src: String, mod: String = "m", presetRegs: Set[String] = Set(),
                      memInit: Map[String, MemoryInitValue] = Map()): TransitionSystem = {
    val circuit = compile(src)
    val module = circuit.modules.find(_.name == mod).get.asInstanceOf[ir.Module]
    // println(module.serialize)
    new ModuleToTransitionSystem().run(module, presetRegs = presetRegs, memInit = memInit)
  }

  protected def toBotr2(src: String, mod: String = "m"): Iterable[String] =
    Btor2Serializer.serialize(toSys(src, mod))

  protected def toBotr2Str(src: String, mod: String = "m"): String =
    toBotr2(src, mod).mkString("\n") + "\n"

  protected def toSMTLib(src: String, mod: String = "m"): Iterable[String] =
    SMTTransitionSystemEncoder.encode(toSys(src, mod)).map(SMTLibSerializer.serialize)

  protected def toSMTLibStr(src: String, mod: String = "m"): String =
    toSMTLib(src, mod).mkString("\n") + "\n"
}