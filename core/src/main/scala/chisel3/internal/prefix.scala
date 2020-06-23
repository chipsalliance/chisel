// See LICENSE for license details.

package chisel3.internal

/** Use to add a prefix to any components generated in the provided scope.
  *
  * @example {{{
  *
  * val x1 = prefix("first") {
  *   // Anything generated here will be prefixed with "first"
  * }
  *
  * val x2 = prefix(mysignal) {
  *   // Anything generated here will be prefixed with the name of mysignal
  * }
  *
  * }}}
  *
  */
private[chisel3] object prefix { // scalastyle:ignore

  /** Use to add a prefix to any components generated in the provided scope
    * The prefix is the name of the provided which, which may not be known yet.
    *
    * @param name The signal/instance whose name will be the prefix
    * @param f a function for which any generated components are given the prefix
    * @tparam T The return type of the provided function
    * @return The return value of the provided function
    */
  def apply[T](name: HasId)(f: => T): T = {
    Builder.pushPrefix(name)
    val ret = f
    Builder.popPrefix()
    ret
  }

  /** Use to add a prefix to any components generated in the provided scope
    * The prefix is a string, which must be known when this function is used.
    *
    * @param name The name which will be the prefix
    * @param f a function for which any generated components are given the prefix
    * @tparam T The return type of the provided function
    * @return The return value of the provided function
    */
  def apply[T](name: String)(f: => T): T = {
    Builder.pushPrefix(name)
    val ret = f
    // Sometimes val's can occur between the Module.apply and Module constructor
    // This causes extra prefixes to be added, and subsequently cleared in the
    // Module constructor. Thus, we need to just make sure if the previous push
    // was an incorrect one, to not pop off an empty stack
    if(Builder.getPrefix().nonEmpty) Builder.popPrefix()
    ret
  }
}

/** Use to eliminate any existing prefixes within the provided scope.
  *
  * @example {{{
  *
  * val x1 = noPrefix {
  *   // Anything generated here will not be prefixed by anything outside this scope
  * }
  *
  * }}}
  *
  */
private[chisel3] object noPrefix {

  /** Use to clear existing prefixes so no signals within the scope are prefixed by signals/names
    * outside the scope
    *
    * @param f a function for which any generated components are given the prefix
    * @tparam T The return type of the provided function
    * @return The return value of the provided function
    */
  def apply[T](f: => T): T = {
    val prefix = Builder.getPrefix()
    Builder.clearPrefix()
    val ret = f
    Builder.setPrefix(prefix)
    ret
  }
}
