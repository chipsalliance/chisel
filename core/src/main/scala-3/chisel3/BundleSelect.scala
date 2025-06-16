// SPDX-License-Identifier: Apache-2.0

package chisel3


extension (b: Bundle) {
  def selectDynamic(field: String): Any = b.elements(field)
}
