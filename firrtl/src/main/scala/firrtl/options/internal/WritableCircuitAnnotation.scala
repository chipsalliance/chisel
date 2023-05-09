// SPDX-License-Identifier: Apache-2.0

package firrtl.options.internal

import firrtl.options.CustomFileEmission
import firrtl.annotations.Annotation

import java.io.File

// Hack for enabling special emission of the ChiselCircuitAnnotation in WriteOutputAnnotations
@deprecated("This trait is for internal use only. Do not use it.", "Chisel 5.0")
trait WriteableCircuitAnnotation extends Annotation with CustomFileEmission {

  protected def writeToFileImpl(file: File, annos: Seq[Annotation]): Unit

  private[firrtl] final def writeToFile(file: File, annos: Seq[Annotation]): Unit = writeToFileImpl(file, annos)
}
