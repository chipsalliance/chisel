// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.options.OptionsView
import logger.LazyLogging

/** The [[stage]] package provides Stage/Phase infrastructure for FIRRTL compilers:
  *   - A number of support [[options.Phase Phase]]s
  *   - [[FirrtlOptions]], a class representing options common to FIRRTL compilers
  *   - [[FirrtlOptionsView]], a utility that constructs an [[options.OptionsView OptionsView]] of [[FirrtlOptions]]
  *     from an [[AnnotationSeq]]
  */
package object stage {
  implicit object FirrtlOptionsView extends OptionsView[FirrtlOptions] with LazyLogging {

    /**
      * @todo custom transforms are appended as discovered, can this be prepended safely?
      */
    def view(options: AnnotationSeq): FirrtlOptions = options.collect { case a: FirrtlOption => a }
      .foldLeft(new FirrtlOptions()) { (c, x) =>
        x match {
          case OutputFileAnnotation(f)      => c.copy(outputFileName = Some(f))
          case InfoModeAnnotation(i)        => c.copy(infoModeName = i)
          case FirrtlCircuitAnnotation(cir) => c.copy(firrtlCircuit = Some(cir))
          case AllowUnrecognizedAnnotations => c
        }
      }
  }
}
