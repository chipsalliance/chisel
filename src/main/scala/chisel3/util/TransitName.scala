// See LICENSE for license details.

package chisel3.util

import chisel3._
import internal.HasId

/**
 * The purpose of TransitName is to allow a library to 'move' a name
 * call to a more appropriate place.
 * For example, a library factory function may create a module and return
 * the io. The only user-exposed field is that given IO, which can't use
 * any name supplied by the user. This can add a hook so that the supplied
 * name then names the Module.
 * See Queue companion object for working example
 */
object TransitName {
  def apply[T<:HasId](from: T, to: HasId): T = {
    from.addPostnameHook((given_name: String) => {to.suggestName(given_name)})
    from
  }
  def withSuffix[T<:HasId](suffix: String)(from: T, to: HasId): T = {
    from.addPostnameHook((given_name: String) => {to.suggestName(given_name + suffix)})
    from
  }
}
