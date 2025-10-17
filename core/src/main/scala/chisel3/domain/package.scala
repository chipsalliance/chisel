// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.Builder

package object domain {

  /** Add a [[Domain]] kind to Chisel's runtime Builder so that it will be
    * unconditionally emitted during FIRRTL emission.
    *
    * @param domain the kind of domain to add
    */
  def addDomain(domain: Domain) = {
    Builder.domains += domain
  }

}
