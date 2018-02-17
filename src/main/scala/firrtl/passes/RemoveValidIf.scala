// See LICENSE for license details.

package firrtl
package passes
import firrtl.Mappers._
import firrtl.ir._
import Utils.throwInternalError
import WrappedExpression.weq

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

  private def getGroundZero(tpe: Type): Expression = tpe match {
    case _: UIntType => UIntZero
    case _: SIntType => SIntZero
    case ClockType => ClockZero
    case other => throwInternalError()
  }

  // Recursive. Replaces IsInvalid with connecting zero
  private def onStmt(s: Statement): Statement = s map onStmt map onExp match {
    case invalid @ IsInvalid(info, loc) => loc.tpe match {
      case _: AnalogType => EmptyStmt
      case tpe => Connect(info, loc, getGroundZero(tpe))
    }
    // Register connected to itself (since reset has been made explicit) is a register with no reset
    // and no connections, connect it to zero (to be constant propped later)
    case Connect(info, lref: WRef, rref: WRef) if weq(lref, rref) =>
      // We can't have an Analog reg so just get a zero
      Connect(info, lref, getGroundZero(lref.tpe))
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
