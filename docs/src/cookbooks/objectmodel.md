---
sidebar_position: 5
---

# ObjectModel Cookbook

import TOCInline from '@theme/TOCInline';

<TOCInline toc={toc} />

## Example of accessing data from OM class

```scala
import chisel3._
import chisel3.properties._
import chisel3.panamaom._

class IntPropTest extends RawModule {
  val intProp = IO(Output(Property[Int]()))
  intProp := Property(123)
}

val converter = Seq(
  new chisel3.stage.phases.Elaborate,
  chisel3.panamaconverter.stage.Convert
).foldLeft(
  firrtl.AnnotationSeq(Seq(chisel3.stage.ChiselGeneratorAnnotation(() => new IntPropTest)))
) { case (annos, phase) => phase.transform(annos) }
  .collectFirst {
    case chisel3.panamaconverter.stage.PanamaCIRCTConverterAnnotation(converter) => converter
  }
  .get

val pm = converter.passManager()
assert(pm.populatePreprocessTransforms())
assert(pm.populateCHIRRTLToLowFIRRTL())
assert(pm.populateLowFIRRTLToHW())
assert(pm.populateFinalizeIR())
assert(pm.run())

val om = converter.om()
val evaluator = om.evaluator()
val obj = evaluator.instantiate("PropertyTest_Class", Seq(om.newBasePathEmpty)).get

val value = obj.field("intProp").asInstanceOf[PanamaCIRCTOMEvaluatorValuePrimitiveInteger].integer
assert(value === 123)
```
