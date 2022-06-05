// SPDX-License-Identifier: Apache-2.0

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
    val pushed = Builder.pushPrefix(name)
    val ret = f
    if (pushed) {
      Builder.popPrefix()
    }
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
    if (Builder.getPrefix.nonEmpty) Builder.popPrefix()
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
    val prefix = Builder.getPrefix
    Builder.clearPrefix()
    val ret = f
    Builder.setPrefix(prefix)
    ret
  }
}

/** Use to set the name ("suggested" because identical suggestions will be uniquified)
  *
  * @param name The desired name for the entity. Collisions will be uniquified. Any prefixes will be cleared/ignored.
  * @param f a function for which the return is given the suggested name.
  * @tparam T The return type of the provided function
  * @return The return value of the provided function
  */
private[chisel3] object withSuggestedName {

  private def _apply[T <: HasId](prevId: Long, name: String, nameMe: T): T = {
    require(
      nameMe._id > prevId,
      s"Cannot call withSuggestedName($name){...} on already created hardware $nameMe. Be sure to wrap and return an original IO, Reg, Wire, Module, or Instance call, or some logical operation that creates a new node."
    )
    nameMe.suggestNameInternal(name)
  }

  /** Use to set the name ("suggested" because identical suggestions will be uniquified)
    *
    * @param name The name to use
    * @param nameMe The thing to be named
    * @tparam T The type of the thing to be named
    * @return The thing, but now named
    *
    * @example
    * ```
    * val foo = withSuggestedName("useThisName"){
    *    val bar = IO(Input(Bool()))        // will be called foo_bar
    *    val unusedName = IO(Input(Bool())) // will be called useThisName
    *    val baz = bar | unusedName         // will be called foo_baz
    *    unusedName
    * }
    * ```
    * @note The thing to be named must be created within the body of the function.
    *
    * @example
    * ```
    * // This will be a runtime error
    * val foo = IO(Input(Bool())
    * val bar = IO(Input(Bool())
    * val result = foo | bar
    * val baz = withSuggestedName("useThisName"){result} // Error: result was not created inside the body.
    * ```
    */
  def apply[T <: HasId](name: String)(nameMe: => T): T = {
    // The _id of the most recently constructed HasId
    val prevId = Builder.idGen.value
    val result = nameMe
    _apply(prevId, name, result)
  }

  /** Use to set the name ("suggested" because identical suggestions will be uniquified)
    *
    * @param names A sequence of names to use. Must be same length as nameMe has items.
    * @param nameUs A block that returns a Tuple of the things to be named. Must be same length as name.
    * @tparam T The type of the thing to be named
    * @return The thing, but now named
    *
    * @example
    * ```
    * val (fooA, fooB) = withSuggestedName(Seq("useThisNameA", "useThisNameB")){
    *    val bar = IO(Input(Bool()))        // will be called foo_bar
    *    val unusedNameA = IO(Input(Bool())) // will be called useThisNameA
    *    val unusedNameB = IO(Input(Bool())) // will be called useThisNameB
    *    val baz = bar | unusedName         // will be called foo_baz
    *    (unusedNameA, unusedNameB)
    * }
    * ```
    * @note All the things to be named must be created within the body of the function.
    *
    * @example
    * ```
    * // This will be a runtime error
    * val foo = IO(Input(Bool())
    * val bar = IO(Input(Bool())
    * val resultA = foo | bar
    * val resultB = foo & bar
    * val (bazA, bazB) = withSuggestedName(Seq("useThisNameA", "useThisNameB")){
    *   (resultA, resultB)
    * } // Error: resultA was not created inside the body.
    * ```
    */
  def apply[T <: Product](names: Seq[String])(nameUs: => T): T = {
    // The _id of the most recently constructed HasId
    val prevId = Builder.idGen.value
    val result = nameUs
    require(
      result.productIterator.length == names.length,
      s"Mismatch in lengths ${result.productIterator.length} vs ${names.length}, suggestedNames were: \n ${names.mkString(",\n ")}"
    )
    for ((name, t) <- names.iterator.zip(result.productIterator)) {
      _apply(prevId, name, t.asInstanceOf[HasId]) // TODO: type-safe checking
    }
    result
  }

  //TODO: Seq[HasId] ?
}
