// See LICENSE for license details.

package firrtl
package passes
import firrtl.Mappers._
import firrtl.ir._

/** Remove ValidIf and replace IsInvalid with a connection to zero */
object RemoveValidIf extends Pass {
  // Recursive. Removes ValidIfs
  private def onExp(e: Expression): Expression = {
    e map onExp match {
      case ValidIf(_, value, _) => value
      case x => x
    }
  }
  private val UIntZero = Utils.zero
  private val SIntZero = SIntLiteral(BigInt(0), IntWidth(1))
  private val ClockZero = DoPrim(PrimOps.AsClock, Seq(UIntZero), Seq.empty, UIntZero.tpe)

  // Recursive. Replaces IsInvalid with connecting zero
  private def onStmt(s: Statement): Statement = s map onStmt map onExp match {
    case invalid @ IsInvalid(info, loc) => loc.tpe match {
      case _: UIntType => Connect(info, loc, UIntZero)
      case _: SIntType => Connect(info, loc, SIntZero)
      case _: AnalogType => invalid // Unclear what we should do, can't remove or we emit invalid Firrtl
      case ClockType => Connect(info, loc, ClockZero)
      case other => throw new Exception("Unexpected type ${other.serialize} on LowFirrtl")
    }
    case other => other
  }

  private def onModule(m: DefModule): DefModule = {
    m match {
      case m: Module => Module(m.info, m.name, m.ports, onStmt(m.body))
      case m: ExtModule => m
    }
  }

  def run(c: Circuit): Circuit = Circuit(c.info, c.modules.map(onModule), c.main)
}
