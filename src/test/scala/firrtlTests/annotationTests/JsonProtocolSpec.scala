// See LICENSE for license details.

package firrtlTests.annotationTests

import firrtl.Parser
import firrtl.annotations.{Annotation, JsonProtocol, NoTargetAnnotation}
import firrtl.ir._
import org.scalatest.{FlatSpec, Matchers, PropSpec}

case class AnAnnotation(
    info: Info,
    cir: Circuit,
    mod: DefModule,
    port: Port,
    statement: Statement,
    expr: Expression,
    tpe: Type
) extends NoTargetAnnotation

class JsonProtocolSpec extends FlatSpec with Matchers {
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
    val inputAnnos = Seq(AnAnnotation(cir.info, cir, mod, port, stmt, expr, tpe))
    val annosString = JsonProtocol.serialize(inputAnnos)
    val outputAnnos = JsonProtocol.deserialize(annosString)
    inputAnnos should be (outputAnnos)
  }

}
