// See LICENSE for license details.

package firrtl.options

import firrtl.AnnotationSeq

/** Type class defining a "view" of an [[AnnotationSeq]]
  * @tparam T the type to which this viewer converts an [[AnnotationSeq]] to
  */
trait OptionsView[T] {

  /** Convert an [[AnnotationSeq]] to some other type
    * @param options some annotations
    */
  def view(options: AnnotationSeq): Option[T]
}

/** A shim to manage multiple "views" of an [[AnnotationSeq]] */
object Viewer {

  /** Convert annotations to options using an implicitly provided [[OptionsView]]
    * @param options some annotations
    * @param optionsView a converter of options to the requested type
    * @tparam T the type to which the input [[AnnotationSeq]] should be viewed as
    */
  def view[T](options: AnnotationSeq)(implicit optionsView: OptionsView[T]): Option[T] = optionsView.view(options)
}
