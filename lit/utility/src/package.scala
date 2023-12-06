import chisel3._

package object utility {
  object binding {
    def streamString(module: => RawModule, stream: chisel3.internal.CIRCTConverter => geny.Writable): String = Seq(
      new chisel3.stage.phases.Elaborate,
      chisel3.internal.panama.Convert
    ).foldLeft(
        firrtl.AnnotationSeq(Seq(chisel3.stage.ChiselGeneratorAnnotation(() => module)))
      ) { case (annos, phase) => phase.transform(annos) }
      .collectFirst {
        case chisel3.internal.panama.circt.PanamaCIRCTConverterAnnotation(converter) =>
          val string = new java.io.ByteArrayOutputStream
          stream(converter).writeBytesTo(string)
          new String(string.toByteArray)
      }
      .get

    def firrtlString(module: => RawModule): String = streamString(module, _.firrtlStream)

    def verilogString(module: => RawModule): String = streamString(module, _.verilogStream)
  }
}