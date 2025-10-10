// SPDX-License-Identifier: Apache-2.0

package chisel3.domain

import chisel3.experimental.SourceInfo
import chisel3.util.simpleClassName

object Field {
  sealed trait Type

  object String extends Type
}

abstract class Domain()(implicit val sourceInfo: SourceInfo) { self: Singleton =>

  private[chisel3] def name: String = simpleClassName(this.getClass())

  def fields: Seq[(String, Field.Type)] = Seq.empty

}
