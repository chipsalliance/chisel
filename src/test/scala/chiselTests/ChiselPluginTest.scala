package chiselTests

import Chisel.experimental.{treedump}
import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselMain, ChiselStage}

import scala.annotation.StaticAnnotation


object DivZeroTest {
  val five = 5
  val amount = five / 0
  def main(args: Array[String]): Unit = {
    println(amount)
  }
}

class wrapThisMethod extends StaticAnnotation {}
@treedump
class TestClass extends MultiIOModule{
  {
    val xyxy = Some( { _: Data => Wire(Bool()) } )
    val xxxx = xyxy.map(_(3.U)).getOrElse(true.B)
    val yyyy = { xyxy.map(_(3.U)).getOrElse(true.B) }
    println(xxxx.getName)
    println(yyyy.getName)
  }
}

object NamerTest {
  def main(args: Array[String]): Unit = {
    new ChiselStage().run(Seq(ChiselGeneratorAnnotation(() => new TestClass)))

  }
}
