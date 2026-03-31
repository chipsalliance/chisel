// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator

import chisel3.RawModule
import svsim.CommonCompilationSettings.VerilogPreprocessorDefine

/** Utilities for controlling instance choice selections */
object InstanceChoiceControl {

  /** Enum representing when instance choice specialization occurs */
  sealed trait SpecializationTime
  object SpecializationTime {
    case object FirtoolCompilationTime extends SpecializationTime
    case object VerilogElaborationTime extends SpecializationTime
  }

  /** The type of all instance choice control variations */
  sealed trait Type {

    /** Return the preprocessor defines that should be set to enable instance choices.
      *
      * Instance choices use a macro-based ABI where each option case is represented
      * by a macro with the format `targets$<option>$<case>` (e.g., `targets$Platform$FPGA`).
      *
      * @param module an elaborated Chisel module
      * @return preprocessor defines to control instance choice selection
      */
    final def preprocessorDefines(
      module: ElaboratedModule[_ <: RawModule]
    ): Seq[VerilogPreprocessorDefine] = {
      getVerilogElaborationTimeChoices.map { case (option, caseValue) =>
        VerilogPreprocessorDefine(s"targets$$${option}$$${caseValue}")
      }
    }

    /** Return the (option, value) pairs for VerilogElaborationTime choices */
    protected def getVerilogElaborationTimeChoices: Seq[(String, String)]

    /** Convert instance choices to firtool command-line options */
    def toFirtoolOptions: Seq[String]
  }

  /** Instance choice control implementation
    *
    * @param choices sequence of (specializationTime, option, case) tuples
    */
  private case class Choices(choices: Seq[(SpecializationTime, String, String)]) extends Type {

    override protected def getVerilogElaborationTimeChoices: Seq[(String, String)] = {
      choices.collect { case (SpecializationTime.VerilogElaborationTime, option, value) =>
        (option, value)
      }
    }

    override def toFirtoolOptions: Seq[String] = {
      choices.collect { case (SpecializationTime.FirtoolCompilationTime, option, caseValue) =>
        Seq("--select-instance-choice", s"$option=$caseValue")
      }.flatten
    }
  }

  /** Create instance choices from a sequence */
  def apply(choices: Seq[(SpecializationTime, String, String)]): Type = Choices(choices)

}
