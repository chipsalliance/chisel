package chiselTests
package experimental

import chisel3._
import chiselTests.experimental.GCDSerializableModule
import chisel3.experimental.util.SerializableModuleElaborator

class GCDSerializableModuleElaborator extends SerializableModuleElaborator {
  def config(parameter: GCDSerializableModuleParameter): String = configImpl(parameter)

  def design(parameter: GCDSerializableModuleParameter) =
    designImpl[GCDSerializableModule, GCDSerializableModuleParameter](config(parameter))
}

class SerializableModuleElaboratorSpec extends ChiselFlatSpec {
  val elaborator = new GCDSerializableModuleElaborator
  val design = elaborator.design(GCDSerializableModuleParameter(16))

  "SerializableModuleElaborator" should "elaborate firrtl" in {
    design.firFile should include("module GCD")
  }

  "SerializableModuleElaborator" should "filter unserializable annotations" in {
    (design.annosFile should not).include("UnserializeableAnnotation")
    (design.annosFile should not).include("DesignAnnotation")
    (design.annosFile should not).include("ChiselCircuitAnnotation")
  }
}
