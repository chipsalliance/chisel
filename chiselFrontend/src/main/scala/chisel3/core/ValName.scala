// See LICENSE.SiFive for license details.

package chisel3.core

import scala.language.experimental.macros
import chisel3.internal.ValNameImpl

case class ValName(name: String)

object ValName
{
  implicit def materialize(implicit x: ValNameImpl): ValName = ValName(x.name)
}
