// SPDX-License-Identifier: Apache-2.0

package chisel3

/** A trait that can be mixed into a Chisel module to indicate that a module has external users.
  *
  * This will result in a public FIRRTL module being produced.
  */
trait Public { this: RawModule =>

  override private[chisel3] def _isPublic = isPublic

  /** Is this module public?
    *
    * Users can override this if they need more control over when outputs of this Module should
    * be considered public
    */
  def isPublic: Boolean = true
}
