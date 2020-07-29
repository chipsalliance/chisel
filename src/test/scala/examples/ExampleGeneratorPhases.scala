// See LICENSE for license details.

package examples

import chisel3.stage.{ChiselStage, DefaultGeneratorPackage, DefaultGeneratorPackageCreator, MultipleGeneratorPackageCreator, PrintFullStackTraceAnnotation}
import chisel3._
import chisel3.aop.Select
import chisel3.experimental.BaseModule
import firrtl.AnnotationSeq
import firrtl.options.{Dependency, PhaseManager, Stage}
import org.scalatest.{FlatSpec, Matchers}


class Simple(nDelay: Int) extends RawModule {
  val in:  UInt = IO(Input(UInt(3.W)))
  val out: UInt = IO(Output(UInt(3.W)))
  out := in
}

class SimpleTest(template: Template[Simple]) extends MultiIOModule {
  val simple = template.instantiate()
  assert(simple(_.out) === simple(_.in))
}

object SimplePackageCreator extends MultipleGeneratorPackageCreator[Simple] {
  override val packge = "Seq[Simple]"
  override val version = "snapshot"

  override def createGenerators(packages: Seq[GeneratorPackage[BaseModule]]): Seq[() => Simple] = {
    Seq(
      () => new Simple(3),
      () => new Simple(5),
    )
  }

  override def createPackage(top: Simple, annos: AnnotationSeq): DefaultGeneratorPackage[Simple] = {
    DefaultGeneratorPackage[Simple](top, annos, SimplePackageCreator)
  }
}

case class Bar() extends RawModule {
  val in:  UInt = IO(Input(UInt(3.W)))
  val out: UInt = IO(Output(UInt(3.W)))
  val templates = importTemplate[Simple](Some(SimplePackageCreator))
  out := templates.foldLeft(in) { (data, template) =>
    val simple = template.instantiate()
    simple(_.in) := data
    simple(_.out)
  }
}

object BarCreator extends DefaultGeneratorPackageCreator[Bar] {
  val packge = "Bar"
  val version = "snapshot"

  def gen() = new Bar

  override def createPackage(top: Bar, annos: AnnotationSeq) =
    DefaultGeneratorPackage[Bar](top, annos, BarCreator)

  override def prerequisites = Seq(Dependency(SimplePackageCreator))
}

object SimpleTester extends MultipleGeneratorPackageCreator[SimpleTest] {
  val packge = "SimpleTesters"
  val version = "snapshot"

  override def createGenerators(packages: Seq[GeneratorPackage[BaseModule]]): Seq[() => SimpleTest] = {
    val x = packages.flatMap { packge =>
      Select.collectDeep(packge.top) {
        case s: Simple => () => new SimpleTest(Template(s, Some(packge)))
      }
    }
    x
  }
  override def createPackage(top: SimpleTest, annos: AnnotationSeq): DefaultGeneratorPackage[SimpleTest] = {
    DefaultGeneratorPackage[SimpleTest](top, annos, SimpleTester)
  }

  override def prerequisites = Seq(Dependency(BarCreator))

}





class ChiselStageSpec extends FlatSpec with Matchers {

  class Fixture { val stage: Stage = new ChiselStage }

  "BarPhase" should "work as a dependent of SimplePhase" in new Fixture {
    case class ChiselLangRunner() extends PhaseManager( Seq(Dependency(BarCreator)) )
    val x = new ChiselLangRunner().transform(Nil)
    println(x)
  }

  "SimpleTester" should "create all simple tests" in new Fixture {
    case class ChiselLangRunner() extends PhaseManager( Seq(Dependency(SimpleTester)) )
    val x = new ChiselLangRunner().transform(Seq(PrintFullStackTraceAnnotation))
    println(x)
  }

}
