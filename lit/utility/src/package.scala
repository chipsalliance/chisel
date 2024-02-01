import chisel3._
import chisel3.panamaconverter.PanamaCIRCTConverter

package object lit {
  object utility {
    object panamaconverter {
      def getConverter(module: => RawModule): PanamaCIRCTConverter = Seq(
        new chisel3.stage.phases.Elaborate,
        chisel3.panamaconverter.stage.Convert
      ).foldLeft(
        firrtl.AnnotationSeq(Seq(chisel3.stage.ChiselGeneratorAnnotation(() => module)))
      ) { case (annos, phase) => phase.transform(annos) }.collectFirst {
        case chisel3.panamaconverter.stage.PanamaCIRCTConverterAnnotation(converter) =>
          converter
      }.get

      def streamString(module: => RawModule, stream: PanamaCIRCTConverter => geny.Writable): String = {
        val converter = getConverter(module)
        val string = new java.io.ByteArrayOutputStream
        stream(converter).writeBytesTo(string)
        new String(string.toByteArray)
      }

      def firrtlString(module: => RawModule): String = streamString(module, _.firrtlStream)

      def verilogString(module: => RawModule): String = streamString(module, _.verilogStream)
    }
  }
}
