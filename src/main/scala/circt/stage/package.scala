// SPDX-License-Identifier: Apache-2.0

package circt

import circt.stage.{CIRCTOption, CIRCTTargetAnnotation, PreserveAggregate}

import firrtl.AnnotationSeq
import firrtl.options.OptionsView
import firrtl.stage.{FirrtlOption, OutputFileAnnotation}

import java.io.File

package object stage {

  implicit object CIRCTOptionsView extends OptionsView[CIRCTOptions] {

    def view(annotations: AnnotationSeq): CIRCTOptions =
      annotations.collect {
        case a: CIRCTOption  => a
        case a: FirrtlOption => a
      }
        .foldLeft(new CIRCTOptions()) { (acc, c) =>
          c match {
            case OutputFileAnnotation(a)  => acc.copy(outputFile = Some(new File(a)))
            case CIRCTTargetAnnotation(a) => acc.copy(target = Some(a))
            case PreserveAggregate(a)     => acc.copy(preserveAggregate = Some(a))
            case FirtoolOption(a)         => acc.copy(firtoolOptions = acc.firtoolOptions :+ a)
            case SplitVerilog             => acc.copy(splitVerilog = true)
            case DumpFir                  => acc.copy(dumpFir = true)
            case _                        => acc
          }
        }

  }

}
