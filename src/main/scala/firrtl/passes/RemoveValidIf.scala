// See LICENSE for license details.

package firrtl
package passes
import firrtl.Mappers._
import firrtl.ir._
import Utils.throwInternalError

/** Remove [[firrtl.ir.ValidIf ValidIf]] and replace [[firrtl.ir.IsInvalid IsInvalid]] with a connection to zero */
object RemoveValidIf extends Pass {

  val UIntZero = Utils.zero
  val SIntZero = SIntLiteral(BigInt(0), IntWidth(1))
  val ClockZero = DoPrim(PrimOps.AsClock, Seq(UIntZero), Seq.empty, ClockType)
  val FixedZero = FixedLiteral(BigInt(0), IntWidth(1), IntWidth(0))

  /** Returns an [[firrtl.ir.Expression Expression]] equal to zero for a given [[firrtl.ir.GroundType GroundType]]
    * @note Accepts [[firrtl.ir.Type Type]] but dyanmically expects [[firrtl.ir.GroundType GroundType]]
    */
  def getGroundZero(tpe: Type): Expression = tpe match {
    case _: UIntType => UIntZero
    case _: SIntType => SIntZero
    case ClockType => ClockZero
    case _: FixedType => FixedZero
    case other => throwInternalError(s"Unexpected type $other")
  }

  // Recursive. Removes ValidIfs
  private def onExp(e: Expression): Expression = {
    e map onExp match {
      case ValidIf(_, value, _) => value
      case x => x
    }
  }

  // Recursive. Replaces IsInvalid with connecting zero
  private def onStmt(s: Statement): Statement = s map onStmt map onExp match {
    case invalid @ IsInvalid(info, loc) => loc.tpe match {
      case _: AnalogType => EmptyStmt
      case tpe => Connect(info, loc, getGroundZero(tpe))
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
