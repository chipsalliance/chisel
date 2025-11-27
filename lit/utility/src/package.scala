import chisel3._

package object lit {
  object utility {
    //object panamaconverter {
    //  def getConverter(
    //    module:         => RawModule,
    //    firtoolOptions: FirtoolOptions = FirtoolOptions(Set.empty)
    //  ): PanamaCIRCTConverter = Seq(
    //    new chisel3.stage.phases.Elaborate,
    //    chisel3.panamaconverter.stage.Convert
    //  ).foldLeft(
    //    firrtl.AnnotationSeq(Seq(chisel3.stage.ChiselGeneratorAnnotation(() => module)))
    //  ) { case (annos, phase) => phase.transform(annos) }
    //    .collectFirst { case chisel3.panamaconverter.stage.PanamaCIRCTConverterAnnotation(converter) =>
    //      converter
    //    }
    //    .get

    //  def runAllPass(converter: PanamaCIRCTConverter) = {
    //    val pm = converter.passManager()
    //    assert(pm.populatePreprocessTransforms())
    //    assert(pm.populateCHIRRTLToLowFIRRTL())
    //    assert(pm.populateLowFIRRTLToHW())
    //    assert(pm.populateFinalizeIR())
    //    assert(pm.run())
    //  }

    //  def streamString(
    //    module:         => RawModule,
    //    firtoolOptions: FirtoolOptions = FirtoolOptions(Set.empty),
    //    stream:         PanamaCIRCTConverter => geny.Writable
    //  ): String = {
    //    val converter = getConverter(module)
    //    val string = new java.io.ByteArrayOutputStream
    //    stream(converter).writeBytesTo(string)
    //    new String(string.toByteArray)
    //  }

    //  def mlirString(module: => RawModule, firtoolOptions: FirtoolOptions = FirtoolOptions(Set.empty)): String =
    //    streamString(module, firtoolOptions, _.mlirStream)
    //  def firrtlString(module: => RawModule, firtoolOptions: FirtoolOptions = FirtoolOptions(Set.empty)): String =
    //    streamString(module, firtoolOptions, _.firrtlStream)
    //  def verilogString(module: => RawModule, firtoolOptions: FirtoolOptions = FirtoolOptions(Set.empty)): String =
    //    streamString(module, firtoolOptions, _.verilogStream)
    //}
  }
}
