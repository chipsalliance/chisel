package chiselTests
package experimental

import chisel3._
import chisel3.experimental._
import chisel3.experimental.hierarchy._

case class DelayHandoff(in: Data, out: Data, n: Int)(implicit clock: Clock, reset: Reset) extends Handoff

@instantiable
class DelayInterface(n: Int, width: Int) extends Module with HasImplementation[DelayHandoff] {
  @public val in = IO(Input(UInt(width.W)))
  @public val out = IO(Output(UInt(width.W)))
  val implementation = DelayInterface
  val handoff = DelayHandoff(in, out, n)(clock, reset)
}

object DelayInterface extends Implementation[DelayHandoff] {
  def build(handoff: DelayHandoff): Unit = {
    import handoff._
    //TODO bugfix: for some reason noPrefix around call to build doesn't stop "delay" from it
    out := (0 until n).foldLeft(in) { case (rPrev, _) =>
      RegNext(rPrev)
    }
  }
}

case class BarHandoff(a: Bool, b: Bool) extends Handoff

@instantiable
class Bar extends ExtModule with HasImplementation[BarHandoff] {
  @public val a = IO(Input(Bool()))
  @public val b = IO(Output(Bool()))
  val implementation = Bar
  val handoff = BarHandoff(a, b)
}

object Bar extends Implementation[BarHandoff] {
  def build(handoff: BarHandoff): Unit = {
    import handoff._
    b := ~a
  }
}


class InterfaceImplementationSpec extends SeparableSpec with Utils {
  describe("(0) Module with implementation") {
    it("(0.a): Use normal Module() to build both interface and implementation") {
      class Top extends Module {
        val delay = Module(new DelayInterface(3, 8))
        delay.in := 1.U
        printf("%d", delay.out)
      }
      containsChirrtl(new Top)(
        "inst delay of DelayInterface",
        "reg out_REG",
        "reg out_REG_1",
        "reg out_REG_2"
      )()
    }
    it("(0.b): Use Definition/Instance to build both interface and implementation") {
      class Top extends Module {
        val delay = Instance(Definition(new DelayInterface(3, 8)))
        delay.in := 1.U
        printf("%d", delay.out)
      }
      containsChirrtl(new Top)(
        "inst delay of DelayInterface",
        "reg out_REG",
        "reg out_REG_1",
        "reg out_REG_2"
      )()
    }
    it("(0.c): Use Definition/Instance + Interface/Implementation to mock chained, but separable compilation") {
      object ComponentShelf {
        val delay = containsChirrtl(new DelayInterface(3, 8))(
          "reg out_REG",
          "reg out_REG_1",
          "reg out_REG_2"
        )()
      }
      class Top extends Module {
        val delay = Instance(ComponentShelf.delay.top)
        delay.in := 1.U
        printf("%d", delay.out)
      }
      containsChirrtl(new Top, ComponentShelf.delay.compiledModules)(
        "inst delay of DelayInterface",
      )( // Should omit
        "reg out_REG",
        "reg out_REG_1",
        "reg out_REG_2"
      )
    }
    it("(0.d): Use Definition/Instance + Interface/Implementation to mock independent but common interface") {
      object InterfaceShelf {
        val delayInterface = containsChirrtl(new DelayInterface(3, 8), Nil, false)(
          "module DelayInterface"
        )( // Should omit
          "reg out_REG",
          "reg out_REG_1",
          "reg out_REG_2"
        )
      }

      // First do the client
      class Top extends Module {
        val delay = Instance(InterfaceShelf.delayInterface.top)
        delay.in := 1.U
        printf("%d", delay.out)
      }
      containsChirrtl(new Top, InterfaceShelf.delayInterface.compiledModules)(
        "module DelayInterface",
        "inst delay of DelayInterface",
      )( // Should omit
        "reg out_REG",
        "reg out_REG_1",
        "reg out_REG_2"
      )

      // Then build the component
      val delay = containsChirrtl(InterfaceShelf.delayInterface.top.buildImplementation)(
        "module DelayInterface",
        "reg out_REG",
        "reg out_REG_1",
        "reg out_REG_2"
      )(
        "inst delay of DelayInterface",
      )
    }
    it("(0.e): Show exact separable example of Foo and Bar") {
      object BarInterface {
        val result = compile(new Bar(), buildImplementation = false)
        val barInterface: Interface[Bar] = result.top
      }
      class Foo extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))
        val bar1 = Instance(BarInterface.barInterface)
        val bar2 = Instance(BarInterface.barInterface)

        bar1.a := a
        bar2.a := bar1.b
        b := bar2.b
      }
      containsChirrtl(new Foo, BarInterface.result.compiledModules)(
        "circuit Foo :",
          "extmodule Bar :",
            "input a : UInt<1>",
            "output b : UInt<1>",
          "module Foo :",
            "input a : UInt<1>",
            "output b : UInt<1>",
            "inst bar1 of Bar",
            "inst bar2 of Bar",
            "bar1.a <= a",
            "bar2.a <= bar1.b",
            "b <= bar2.b",
      )( // Should omit implementation of Bar
        "node _b_T = not(a)",
        "b <= _b_T",
      )
      containsChirrtl(BarInterface.barInterface.buildImplementation)(
        "circuit Bar :",
          "module Bar :",
            "input a : UInt<1>",
            "output b : UInt<1>",
            "node _b_T = not(a)",
            "b <= _b_T",
      )()
    }
  }
}



import org.scalatest.funspec.AnyFunSpec
import chisel3.stage._
import chisel3.experimental.hierarchy.core.ImportDefinitionAnnotation
import chisel3.experimental.hierarchy._
import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase, PhaseManager}

import scala.reflect.runtime.universe.TypeTag

case class CompilationResult[T <: BaseModule](top: Interface[T], compiledModules: Seq[ImportDefinitionAnnotation[_]], output: String) {
  def exportAs[R : TypeTag]: CompilationResult[R with BaseModule] =  {
    CompilationResult(compiledModules.collectFirst {
      case x: ImportDefinitionAnnotation[R with BaseModule] @unchecked if x.definition.isA[R] => x.definition
    }.get, compiledModules, output)
  }
}

abstract class SeparableSpec extends AnyFunSpec {
  def verbosePrinting: Boolean = true
  private def matches(lines: List[String], matchh: String): Option[String] = lines.findLast(_.contains(matchh))
  private def omits(line: String, omit: String): Option[(String, String)] = if (line.contains(omit)) Some((omit, line)) else None
  private def omits(lines: List[String], omit: String): Seq[(String, String)] = lines.flatMap { omits(_, omit) }

  def containsChirrtl[T <: BaseModule](gen: => T, imports: Seq[ImportDefinitionAnnotation[_]] = Nil, buildImplementation: Boolean = true)(matchList: String*)(omitList: String*): CompilationResult[T] = {
    val result = compile(gen, buildImplementation, imports)
    contains(result.output)(matchList: _*)(omitList: _*)
    result
  }
  def contains(output: String)(matchList: String*)(omitList: String*): Unit = {
    val lines = output.split("\n").toList
    val unmatched = matchList.flatMap { m =>
      if (matches(lines, m).nonEmpty) None else Some(m)
    }.map(x => s"  > $x was unmatched")
    val unomitted = omitList.flatMap { o => omits(lines, o) }.map {
      case (o, l) => s"  > $o was not omitted in ($l)"
    }
    val results = unmatched ++ unomitted
    assert(results.isEmpty, results.mkString("\n"))
  }
  def compile[T <: BaseModule](gen: => T, buildImplementation: Boolean = true, imports: Seq[ImportDefinitionAnnotation[_]] = Nil): CompilationResult[T] = {
    val outputAnnotations = getOutputAnnos(Seq(ChiselGeneratorAnnotation(() => gen, buildImplementation), PrintFullStackTraceAnnotation) ++ imports)
    val d = getInterface[T](outputAnnotations)
    val chirrtl = getChirrtl(outputAnnotations)
    if(verbosePrinting) println(chirrtl)
    val exports = allModulesToImportedDefs(outputAnnotations)
    CompilationResult(d, exports, chirrtl)
  }
  def fakeFileRoundTrip[T](obj: T): T = obj
  private def getDesignAnnotation[T <: BaseModule](annos: AnnotationSeq): DesignAnnotation[T] = {
    val designAnnos = annos.flatMap { a =>
      a match {
        case a: DesignAnnotation[T @unchecked] => Some(a)
        case _ => None
      }
    }
    require(designAnnos.length == 1, s"Exactly one DesignAnnotation should exist, but found: $designAnnos.")
    designAnnos.head
  }
  private def getChirrtl[T <: BaseModule](annos: AnnotationSeq): String = {
    annos.collectFirst {
      //case a: Firrtl => CircuitSerializationAnnotation(a.circuit, "", CircuitSerializationAnnotation.FirrtlFileFormat).getBytes
      case a: firrtl.stage.FirrtlCircuitAnnotation => a.circuit.serialize
    }.get
      //.map(_.toChar)
      //.mkString
  }
  private def elaborate(gen: => BaseModule, buildImplementation: Boolean = true, inputAnnotations: AnnotationSeq = Nil): AnnotationSeq = {
    getOutputAnnos(Seq(ChiselGeneratorAnnotation(() => gen, buildImplementation)))
  }
  private def getOutputAnnos(inputAnnotations: AnnotationSeq): AnnotationSeq = {
    import firrtl.options.{Dependency, Phase, PhaseManager}
    val targets = Seq(
      Dependency[chisel3.stage.phases.Checks],
      Dependency[chisel3.stage.phases.AddImplicitOutputFile],
      Dependency[chisel3.stage.phases.AddImplicitOutputAnnotationFile],
      Dependency[chisel3.stage.phases.AddSerializationAnnotations],
      Dependency[chisel3.stage.phases.Convert],
      Dependency[chisel3.stage.phases.MaybeAspectPhase],
      Dependency[chisel3.stage.phases.MaybeInjectingPhase],
    )
    class ChiselPhase extends PhaseManager(targets) {
    
    }
    val phase = new ChiselPhase {
      override val targets = Seq(
        Dependency[chisel3.stage.phases.Checks],
        Dependency[chisel3.stage.phases.Elaborate],
        Dependency[chisel3.stage.phases.AddImplicitOutputFile],
        Dependency[chisel3.stage.phases.AddImplicitOutputAnnotationFile],
        Dependency[chisel3.stage.phases.Convert],
        Dependency[chisel3.stage.phases.MaybeAspectPhase],
        Dependency[chisel3.stage.phases.MaybeInjectingPhase]
      )
    }

    phase.transform(inputAnnotations)
  }

  /** Elaborates [[AddOne]] and returns its [[Definition]]. */
  private def getInterface[T <: BaseModule](dutAnnos: AnnotationSeq): Interface[T] = {
    // Grab DUT definition to pass into testbench
    getDesignAnnotation(dutAnnos).design.asInstanceOf[T].toInterface
  }

  /** Return [[Definition]]s of all modules in a circuit. */
  private def allModulesToImportedDefs(annos: AnnotationSeq): Seq[ImportDefinitionAnnotation[_]] = {
    annos.flatMap { a =>
      a match {
        case a: ChiselCircuitAnnotation =>
          a.circuit.components.map { c => ImportDefinitionAnnotation(c.id.toInterface) }
        case _ => Seq.empty
      }
    }
  }

}