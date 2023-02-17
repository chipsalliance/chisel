// SPDX-License-Identifier: Apache-2.0

package firrtlTests.annotationTests

import firrtl._
import firrtl.annotations._
import firrtl.ir._
import firrtl.options.Dependency
import firrtl.transforms.DontTouchAnnotation
import scala.util.Failure
import _root_.logger.{LogLevel, LogLevelAnnotation, Logger}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should._
import org.scalatest.Inside._

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

  "Trying to serialize annotations that cannot be serialized" should "tell you why" in {
    case class MyAnno(x: Int) extends NoTargetAnnotation
    inside(JsonProtocol.serializeTry(MyAnno(3) :: Nil)) {
      case Failure(e: UnserializableAnnotationException) =>
        e.getMessage should include("MyAnno")
        // From json4s Exception
        e.getMessage should include("Classes defined in method bodies are not supported")
    }
  }
  "JsonProtocol.serializeRecover" should "emit even annotations that cannot be serialized" in {
    case class MyAnno(x: Int) extends NoTargetAnnotation
    val target = CircuitTarget("Top").module("Foo").ref("x")
    val annos = MyAnno(3) :: DontTouchAnnotation(target) :: Nil
    val res = JsonProtocol.serializeRecover(annos)
    res should include(""""class":"firrtl.annotations.UnserializeableAnnotation",""")
    res should include(""""error":"Classes defined in method bodies are not supported.",""")
    res should include(""""content":"MyAnno(3)"""")
  }
}
