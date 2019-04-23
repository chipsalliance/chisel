// See LICENSE for license details.

package chisel3.stage.phases

import firrtl.AnnotationSeq
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.{OutputAnnotationFileAnnotation, Phase, PhaseManager, PreservesAll}
import firrtl.stage.phases.DriverCompatibility.TopNameAnnotation

import chisel3.stage.{ChiselOutputFileAnnotation, ChiselStage}

/** This provides components of a compatibility wrapper around Chisel's deprecated [[chisel.Driver Driver]].
  *
  * Primarily, this object includes [[firrtl.options.Phase Phase]]s that generate [[firrtl.annotation.Annotation
  * Annotation]]s derived from the deprecated [[firrtl.options.phases.DriverCompatibility.TopNameAnnotation
  * TopNameAnnotation]].
  */
object DriverCompatibility {

  /** Adds a [[ChiselOutputFileAnnotation]] derived from a [[TopNameAnnotation]] if no [[ChiselOutputFileAnnotation]]
    * already exists. If no [[TopNameAnnotation]] exists, then no [[OutputFileAnnotation]] is added. ''This is not a
    * replacement for [[chisel3.stage.phases.AddImplicitOutputFile AddImplicitOutputFile]] as this only adds an output
    * file based on a discovered top name and not on a discovered elaborated circuit.'' Consequently, this will provide
    * the correct behavior before a circuit has been elaborated.
    * @note the output suffix is unspecified and will be set by [[chisel3.stage.phases.EmitCircuit EmitCircuit]]
    */
  private [chisel3] class AddImplicitOutputFile extends Phase with PreservesAll[Phase] {

    override val dependents: Set[Class[Phase]] =
      Set( classOf[firrtl.stage.phases.DriverCompatibility.AddImplicitOutputFile] )

    def transform(annotations: AnnotationSeq): AnnotationSeq = {
      val hasOutputFile = annotations
        .collectFirst{ case a: ChiselOutputFileAnnotation => a }
        .isDefined
      lazy val top = annotations.collectFirst{ case TopNameAnnotation(a) => a }

      if (!hasOutputFile && top.isDefined) {
        ChiselOutputFileAnnotation(top.get) +: annotations
      } else {
        annotations
      }
    }
  }

  /** If a [[firrtl.options.OutputAnnotationFileAnnotation OutputAnnotationFileAnnotation]] does not exist, this adds one
    * derived from a [[TopNameAnnotation]]. ''This is not a replacement for
    * [[chisel3.stage.phases.AddImplicitOutputAnnotationFile AddImplicitOutputAnnotationFile]] as this only adds an
    * output annotation file based on a discovered top name.'' Consequently, this will provide the correct behavior
    * before a circuit has been elaborated.
    * @note the output suffix is unspecified and will be set by [[firrtl.options.phases.WriteOutputAnnotations
    * WriteOutputAnnotations]]
    */
  private [chisel3] class AddImplicitOutputAnnotationFile extends Phase with PreservesAll[Phase] {

    def transform(annotations: AnnotationSeq): AnnotationSeq = annotations
      .collectFirst{ case a: OutputAnnotationFileAnnotation => annotations }
      .getOrElse{
        val top = annotations.collectFirst{ case TopNameAnnotation(top) => top}

        if (top.isDefined) {
          OutputAnnotationFileAnnotation(top.get) +: annotations
        } else {
          annotations
        }
      }
  }

  /** A placeholder [[Phase]] that enables the scheduling of prerequisites we cannot modify. To prevent an implicit FIRRTL
    * file from being added, this defines the current state as having run that.
    * @todo a better solution than the current state hack below may be needed
    */
  private [chisel3] class Preprocessing
      extends PhaseManager( targets = Set(
                             classOf[AddImplicitOutputFile],
                             classOf[AddImplicitOutputAnnotationFile],
                             classOf[firrtl.stage.phases.DriverCompatibility.AddImplicitOutputFile],
                             classOf[firrtl.stage.phases.DriverCompatibility.AddImplicitEmitter] )
                             .map(_.asInstanceOf[Class[Phase]]),
                            currentState = Set(
                              classOf[firrtl.stage.phases.DriverCompatibility.AddImplicitFirrtlFile])
                              .map(_.asInstanceOf[Class[Phase]]) )
      with PreservesAll[Phase] {

    override val dependents: Set[Class[Phase]] = Set( classOf[ChiselStage] )

  }

}
