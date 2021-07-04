// SPDX-License-Identifier: Apache-2.0

package chiselTests.testers

import chisel3._
import chisel3.stage.ChiselCircuitAnnotation
import chisel3.stage.phases.{Elaborate, Emitter}
import firrtl.AnnotationSeq
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.{Dependency, Phase, TargetDirAnnotation, Unserializable}

object TesterDriver extends BackendCompilationUtilities {

  trait Backend extends NoTargetAnnotation with Unserializable {
    def execute(t: () => BasicTester,
                additionalVResources: Seq[String] = Seq(),
                annotations: AnnotationSeq = Seq(),
                nameHint:    Option[String] = None
               ): Boolean
  }

  val defaultBackend: Backend = sys.env.getOrElse("TEST_BACKEND", "verilator") match {
    case "verilator" => VerilatorBackend
    case "treadle" => TreadleBackend
  }

  /** Use this to force a test to be run only with backends that are restricted to verilator backend
    */
  def verilatorOnly: AnnotationSeq = Seq(VerilatorBackend)

  /** Set the target directory to the name of the top module after elaboration */
  final class AddImplicitTesterDirectory extends Phase {
    override def prerequisites = Seq(Dependency[Elaborate])
    override def optionalPrerequisites = Seq.empty
    override def optionalPrerequisiteOf = Seq(Dependency[Emitter])
    override def invalidates(a: Phase) = false

    override def transform(a: AnnotationSeq) = a.flatMap {
      case a@ ChiselCircuitAnnotation(circuit) =>
        Seq(a, TargetDirAnnotation(
              firrtl.util.BackendCompilationUtilities.createTestDirectory(circuit.name)
                .getAbsolutePath
                .toString))
      case a => Seq(a)
    }
  }

  /** For use with modules that should successfully be elaborated by the
    * frontend, and which can be turned into executables with assertions. */
  def execute(t:                    () => BasicTester,
              additionalVResources: Seq[String] = Seq(),
              annotations:          AnnotationSeq = Seq(),
              nameHint:             Option[String] = None): Boolean = {

    val backendAnnotations = annotations.collect { case anno: Backend => anno }
    val backendAnnotation = if (backendAnnotations.length == 1) {
      backendAnnotations.head
    } else if (backendAnnotations.isEmpty) {
      defaultBackend
    } else {
      throw new ChiselException(s"Only one backend annotation allowed, found: ${backendAnnotations.mkString(", ")}")
    }
    backendAnnotation.execute(t, additionalVResources, annotations, nameHint)
  }

  /**
    * Calls the finish method of an BasicTester or a class that extends it.
    * The finish method is a hook for code that augments the circuit built in the constructor.
    */
  def finishWrapper(test: () => BasicTester): () => BasicTester = { () =>
  {
    val tester = test()
    tester.finish()
    tester
  }
  }

}
