// SPDX-License-Identifier: Apache-2.0

import firrtl.AnnotationSeq
import firrtl.options.OptionsView

package object logger {

  implicit object LoggerOptionsView extends OptionsView[LoggerOptions] {
    def view(options: AnnotationSeq): LoggerOptions = options
      .foldLeft(new LoggerOptions()) { (c, x) =>
        x match {
          case LogLevelAnnotation(logLevel)         => c.copy(globalLogLevel = logLevel)
          case ClassLogLevelAnnotation(name, level) => c.copy(classLogLevels = c.classLogLevels + (name -> level))
          case LogFileAnnotation(f)                 => c.copy(logFileName = f)
          case LogClassNamesAnnotation              => c.copy(logClassNames = true)
          case _                                    => c
        }
      }
  }

}
