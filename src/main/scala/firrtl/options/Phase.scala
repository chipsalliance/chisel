// See LICENSE for license details.

package firrtl.options

import firrtl.AnnotationSeq

/** A transformation of an [[AnnotationSeq]]
  *
  * A [[Phase]] forms one block in the Chisel/FIRRTL Hardware Compiler Framework (HCF). Note that a [[Phase]] may
  * consist of multiple phases internally.
  */
abstract class Phase {

  /** A transformation of an [[AnnotationSeq]]
    * @param annotations some annotations
    * @return transformed annotations
    */
  def transform(annotations: AnnotationSeq): AnnotationSeq

}
