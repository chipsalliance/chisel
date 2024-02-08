// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.Builder

/** A trait that can be mixed into a Chisel module to indicate that a module has external users.
  *
  * This will result in a public FIRRTL module being produced.
  */
trait Public { this: RawModule =>
  isPublic = true
}
