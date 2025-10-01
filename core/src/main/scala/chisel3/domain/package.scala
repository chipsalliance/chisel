// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.Builder

package object domain {

  def addDomain(domain: Domain) = {
    Builder.domains += domain
  }

}
