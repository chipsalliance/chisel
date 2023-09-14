package chiselTests
package experimental

import chisel3._
import chisel3.experimental.HasAutoTypename
import chisel3.util.{Decoupled, Queue}
import circt.stage.ChiselStage

class Top(gen: Bundle) extends Module {
  val in = IO(Input(gen))
  val out = IO(Output(gen))
  override def desiredName = s"Top_${out.typeName}"
}

class SimpleBundle(width: Int) extends Bundle with HasAutoTypename {
  val foo = UInt(width.W)
}

class StringlyBundle(width: Int, desc: String) extends Bundle with HasAutoTypename {
  val foo = UInt(width.W)
}

class DataBundle(data: Data) extends Bundle with HasAutoTypename {
  val foo = data
}

class BundleBundle(gen: HasAutoTypename) extends Bundle with HasAutoTypename {
  val foo = gen
}

class MultiParamBundle(x: Int)(y: String)(implicit z: Int) extends Bundle with HasAutoTypename {
  val foo = UInt(x.W)
  val bar = UInt(z.W)
}

class AutoTypenameSpec extends ChiselFlatSpec {
  private def runTest(gen: Bundle, expectedTypename: String): Unit = {
    val chirrtl = ChiselStage.emitCHIRRTL(new Top(gen))
    // Check that autoTypename even works; the default implementation should not be used
    chirrtl shouldNot include(s"module Top_${gen.getClass.getSimpleName} :")
    chirrtl should include(s"module Top_$expectedTypename :")
  }

  class SeparateTop(gen: Bundle)

  // Tests non `typeName` parameters (`AnyVal`s)
  "Bundles with simple integer parameters" should "have predictable, stable type names" in {
    runTest(new SimpleBundle(1), "SimpleBundle_1")
    runTest(new SimpleBundle(2), "SimpleBundle_2")
    runTest(new SimpleBundle(8), "SimpleBundle_8")
  }

  // Tests recursive `typeName` calls, but not recursive `autoTypename`
  "Bundles with Data element parameters" should "have predictable, stable type names" in {
    runTest(new DataBundle(UInt(8.W)), "DataBundle_UInt8")
    runTest(new DataBundle(Bool()), "DataBundle_Bool")
    runTest(
      new DataBundle(new Bundle {
        val bar = UInt(3.W)
        override def typeName = "AnonBundle"
      }),
      "DataBundle_AnonBundle"
    )
  }

  // Tests recursive `autoTypename`
  "Bundles with auto typenamed Bundle parameters" should "have predictable, stable type names" in {
    runTest(new BundleBundle(new SimpleBundle(1)), "BundleBundle_SimpleBundle_1")
    runTest(new BundleBundle(new DataBundle(UInt(8.W))), "BundleBundle_DataBundle_UInt8")
  }

  "Bundles with string parameters" should "have sanitized type names" in {
    runTest(new StringlyBundle(8, "Some_kind_of_bundle"), "StringlyBundle_8_Some_kind_of_bundle")
    runTest(
      new StringlyBundle(2, "~@9#@$`= The quick brown fox jumped over the lazy dog"),
      "StringlyBundle_2_9Thequickbrownfoxjumpedoverthelazydog"
    )
  }

  "Bundles with multiple parameter lists" should "have generated type names of a single flattened parameter list" in {
    implicit val z: Int = 20
    runTest(new MultiParamBundle(8)("Some_kind_of_bundle"), "MultiParamBundle_8_Some_kind_of_bundle_20")
  }
}
