package chiselTests
package experimental

import chisel3._
import chiselTests.experimental.GCDSerializableModule
import chisel3.experimental.util.SerializableModuleElaborator
import geny.Readable
import upickle.default.read

class GCDSerializableModuleElaborator extends SerializableModuleElaborator {
  val configPath = os.pwd / "config.json"
  val firPath = os.pwd / "GCD.fir"
  val annosPath = os.pwd / "GCD.anno.json"

  def config(parameter: GCDSerializableModuleParameter) =
    os.write.over(configPath, configImpl(parameter))

  def design() = {
    val (firrtl, annos) =
      designImpl[GCDSerializableModule, GCDSerializableModuleParameter](os.read.stream(configPath))
    os.write.over(firPath, firrtl)
    os.write.over(annosPath, annos)
  }
}

class SerializableModuleElaboratorSpec extends ChiselFlatSpec {
  val elaborator = new GCDSerializableModuleElaborator
  elaborator.config(GCDSerializableModuleParameter(16))
  elaborator.design()

  val firFile = os.read(elaborator.firPath)
  val annosFile = os.read(elaborator.annosPath)

  "SerializableModuleElaborator" should "elaborate firrtl" in {
    firFile should include("module GCD")
  }

  "SerializableModuleElaborator" should "filter unserializable annotations" in {
    (annosFile should not).include("UnserializeableAnnotation")
    (annosFile should not).include("DesignAnnotation")
    (annosFile should not).include("ChiselCircuitAnnotation")
  }
}
