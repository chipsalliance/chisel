// See LICENSE for license details.

package chisel3.internal

import scala.language.experimental.macros
import chisel3.internal.prefixing.ValNameImpl

/** Use to capture the name of the val which captures a function's return value
  *
  * @example {{{
  *
  * class Test extends MultiIOModule {
  *   def builder()(implicit valName: ValName) = {
  *     prefix(valName.name) {
  *       val wire1 = Wire(UInt(3.W))
  *       val wire2 = Wire(UInt(3.W))
  *       wire2
  *     }
  *   }
  *   val x1 = builder() // Prefixes everything with "x1" except returned value, which is now "x1"
  *   val x2 = builder() // Prefixes everything with "x2" except returned value, which is now "x2"
  * }
  *
  * }}}
  *
  * @note If you want to bit extract from the return value, this will trigger a compilation time error
  *       if done incorrectly. Given the example above, standard bit extract syntax will error:
  *          {{{ val x1 = builder()(1) // ERROR: Type mismatch; found: Int(1), required: chisel3.internal.ValName }}}
  *       Instead, please either use an intermediate val:
  *          {{{ val x1 = builder(); val bits = x1(1) // OK! }}}
  *       Or explicitly use the apply method:
  *          {{{ val x1 = builder().apply(1) // OK! }}}
  *
  * @todo Maybe use Clippy to provide a better ValName Type mismatch error?
  *
  * @param name The name of the val which it got assigned to.
  */
@scala.annotation.implicitNotFound("Cannot find val name! Did you assign this function's returned value to a val?")
private[chisel3] case class ValName(name: String)

private[chisel3] object ValName
{
  // Used to trigger the macro implementation
  implicit def materialize(implicit x: ValNameImpl): ValName = ValName(x.name)
}
