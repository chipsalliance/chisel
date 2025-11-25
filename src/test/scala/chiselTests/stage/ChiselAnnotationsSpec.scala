// SPDX-License-Identifier: Apache-2.0

package chiselTests.stage

import chisel3._
import chisel3.layer.{Layer, LayerConfig}
import chisel3.stage.{ChiselCircuitAnnotation, ChiselGeneratorAnnotation, DesignAnnotation, RemapLayer}
import firrtl.options.OptionsException
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import firrtl.{annoSeqToSeq, seqToAnnoSeq, AnnotationSeq}

class ChiselAnnotationsSpecFoo extends RawModule {
  val in = IO(Input(Bool()))
  val out = IO(Output(Bool()))
  out := ~in
}

class ChiselAnnotationsSpecBaz(name: String) extends ChiselAnnotationsSpecFoo {
  override val desiredName = name
}

class ChiselAnnotationsSpecQux extends ChiselAnnotationsSpecFoo {
  /* This printf requires an implicit clock and reset, but RawModule has none. This will thereby fail elaboration. */
  printf("hello")
}

class ChiselAnnotation

object ChiselAnnotationsSpec {
  object A extends Layer(LayerConfig.Extract())
}

class ChiselAnnotationsSpec extends AnyFlatSpec with Matchers {

  behavior.of("ChiselGeneratorAnnotation elaboration")

  it should "elaborate to a ChiselCircuitAnnotation" in {
    val annotation = ChiselGeneratorAnnotation(() => new ChiselAnnotationsSpecFoo)
    val res = annotation.elaborate
    res(0) shouldBe a[ChiselCircuitAnnotation]
    res(1) shouldBe a[DesignAnnotation[_]]
  }

  it should "throw an exception if elaboration fails" in {
    val annotation = ChiselGeneratorAnnotation(() => new ChiselAnnotationsSpecQux)
    intercept[ChiselException] { annotation.elaborate }
  }

  behavior.of("ChiselGeneratorAnnotation when stringly constructing from Module names")

  it should "elaborate a module without parameters" in {
    val annotation = ChiselGeneratorAnnotation("chiselTests.stage.ChiselAnnotationsSpecFoo()")
    val res = annotation.elaborate
    res(0) shouldBe a[ChiselCircuitAnnotation]
    res(1) shouldBe a[DesignAnnotation[_]]
  }

  it should "elaborate a module with parameters" in {
    val annotation = ChiselGeneratorAnnotation("""chiselTests.stage.ChiselAnnotationsSpecBaz("hello")""")
    val res = annotation.elaborate
    res(0) shouldBe a[ChiselCircuitAnnotation]
    res(1) shouldBe a[DesignAnnotation[_]]
  }

  it should "throw an exception if elaboration from a String refers to nonexistant class" in {
    val bar = "chiselTests.stage.ChiselAnnotationsSpecBar()"
    val annotation = ChiselGeneratorAnnotation(bar)
    intercept[OptionsException] { annotation.elaborate }.getMessage should startWith(
      s"Unable to run module generator '$bar' because it or one of its arguments could not be found"
    )
  }

  behavior.of("RemapLayer")

  it should "construct an existing in-tree layer" in {
    RemapLayer(ChiselAnnotationsSpec.A.getClass.getName, ChiselAnnotationsSpec.A.getClass.getName)
  }

  it should "throw an exception if the layer does not exist" in {
    intercept[OptionsException] {
      RemapLayer("foo.bar.Verification$", ChiselAnnotationsSpec.A.getClass.getName)
    }.getMessage should startWith("Unable to reflectively find layer 'foo.bar.Verification$'")
  }

  it should "throw an exception if the object is not a layer" in {
    intercept[OptionsException] {
      RemapLayer("circt.stage.ChiselStage$", ChiselAnnotationsSpec.A.getClass.getName)
    }.getMessage should startWith("Object 'circt.stage.ChiselStage$' must be a `Layer`")
  }

}
