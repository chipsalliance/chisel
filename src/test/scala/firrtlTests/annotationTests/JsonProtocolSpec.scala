// See LICENSE for license details.

package firrtlTests.annotationTests

import firrtl._
import firrtl.annotations.{JsonProtocol, NoTargetAnnotation}
import firrtl.ir._
import firrtl.options.Dependency
import _root_.logger.{LogLevel, LogLevelAnnotation, Logger}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should._

case class AnAnnotation(
  info:       Info,
  cir:        Circuit,
  mod:        DefModule,
  port:       Port,
  statement:  Statement,
  expr:       Expression,
  tpe:        Type,
  groundType: GroundType)
    extends NoTargetAnnotation

class AnnoInjector extends Transform with DependencyAPIMigration {
  override def optionalPrerequisiteOf = Dependency[ChirrtlEmitter] :: Nil
  override def invalidates(a: Transform): Boolean = false
  def execute(state: CircuitState): CircuitState = {
    // Classes defined in method bodies can't be serialized by json4s
    case class MyAnno(x: Int) extends NoTargetAnnotation
    state.copy(annotations = MyAnno(3) +: state.annotations)
  }
}

class JsonProtocolSpec extends AnyFlatSpec with Matchers {
  "JsonProtocol" should "serialize and deserialize FIRRTL types" in {

    val circuit =
      """circuit Top: @[FPU.scala 509:25]
        |  module Top:
        |    input x: UInt
        |    output y: UInt
        |    y <= add(x, x)
        |""".stripMargin
    val cir = Parser.parse(circuit)
    val mod = cir.modules.head
    val port = mod.ports.head
    val stmt = mod.asInstanceOf[Module].body
    val expr = stmt.asInstanceOf[Block].stmts.head.asInstanceOf[Connect].expr
    val tpe = port.tpe
    val groundType = port.tpe.asInstanceOf[GroundType]
    val inputAnnos = Seq(AnAnnotation(cir.info, cir, mod, port, stmt, expr, tpe, groundType))
    val annosString = JsonProtocol.serialize(inputAnnos)
    val outputAnnos = JsonProtocol.deserialize(annosString)
    inputAnnos should be(outputAnnos)
  }

  "Annotation serialization during logging" should "not throw an exception" in {
    val compiler = new firrtl.stage.transforms.Compiler(Seq(Dependency[AnnoInjector]))
    val circuit = Parser.parse("""
                                 |circuit test :
                                 |  module test :
                                 |    output out : UInt<1>
                                 |    out <= UInt(0)
      """.stripMargin)
    Logger.makeScope(LogLevelAnnotation(LogLevel.Trace) :: Nil) {
      compiler.execute(CircuitState(circuit, Nil))
    }
  }
}
