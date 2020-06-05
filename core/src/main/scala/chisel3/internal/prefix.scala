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
    Builder.popPrefix()
    ret
  }
}

private[chisel3] object noPrefix {
  def apply[T](f: => T): T = {
    val prefix = Builder.getPrefix()
    Builder.clearPrefix()
    val ret = f
    Builder.setPrefix(prefix)
    ret
  }
}
