// See LICENSE for license details.

package chisel3.util

import chisel3.internal.HasId

/** The purpose of `TransitName` is to improve the naming of some object created in a different scope by "transiting"
  * the name from the outer scope to the inner scope.
  *
  * Consider the example below. This shows three ways of instantiating `MyModule` and returning the IO. Normally, the
  * instance will be named `MyModule`. However, it would be better if the instance was named using the name of the `val`
  * that user provides for the returned IO. `TransitName` can then be used to "transit" the name ''from'' the IO ''to''
  * the module:
  *
  * {{{
  * /* Assign the IO of a new MyModule instance to value "foo". The instance will be named "MyModule". */
  * val foo = Module(new MyModule).io
  *
  * /* Assign the IO of a new MyModule instance to value "bar". The instance will be named "bar". */
  * val bar = {
  *   val x = Module(new MyModule)
  *   TransitName(x.io, x) // TransitName returns the first argument
  * }
  *
  * /* Assign the IO of a new MyModule instance to value "baz". The instance will be named "baz_generated". */
  * val baz = {
  *   val x = Module(new MyModule)
  *   TransitName.withSuffix("_generated")(x.io, x) // TransitName returns the first argument
  * }
  * }}}
  *
  * `TransitName` helps library writers following the [[https://en.wikipedia.org/wiki/Factory_method_pattern Factory
  * Method Pattern]] where modules may be instantiated inside an enclosing scope. For an example of this, see how the
  * [[Queue$ Queue]] factory uses `TransitName` in
  * [[https://github.com/freechipsproject/chisel3/blob/master/src/main/scala/chisel3/util/Decoupled.scala
  * Decoupled.scala]] factory.
  */
object TransitName {

  /** Transit a name from one type to another
    * @param from the thing with a "good" name
    * @param to the thing that will receive the "good" name
    * @return the `from` parameter
    */
  def apply[T<:HasId](from: T, to: HasId): T = {
    from.addPostnameHook((given_name: String) => {to.suggestName(given_name)})
    from
  }


  /** Transit a name from one type to another ''and add a suffix''
    * @param suffix the suffix to append
    * @param from the thing with a "good" name
    * @param to the thing that will receive the "good" name
    * @return the `from` parameter
    */
  def withSuffix[T<:HasId](suffix: String)(from: T, to: HasId): T = {
    from.addPostnameHook((given_name: String) => {to.suggestName(given_name + suffix)})
    from
  }

}
