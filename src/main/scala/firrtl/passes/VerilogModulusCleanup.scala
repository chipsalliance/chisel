// SPDX-License-Identifier: Apache-2.0

package firrtl
package passes

import firrtl.ir._
import firrtl.Mappers._
import firrtl.PrimOps.{Bits, Rem}
import firrtl.Utils._
import firrtl.options.Dependency

import scala.collection.mutable

/**
  * Verilog has the width of (a % b) = Max(W(a), W(b))
  * FIRRTL has the width of (a % b) = Min(W(a), W(b)), which makes more sense,
  * but nevertheless is a problem when emitting verilog
  *
  * This pass finds every instance of (a % b) and:
  *   1) adds a temporary node equal to (a % b) with width Max(W(a), W(b))
  *   2) replaces the reference to (a % b) with a bitslice of the temporary node
  *      to get back down to width Min(W(a), W(b))
  *
  *  This is technically incorrect firrtl, but allows the verilog emitter
  *  to emit correct verilog without needing to add temporary nodes
  */
object VerilogModulusCleanup extends Pass {

  override def prerequisites = firrtl.stage.Forms.LowFormMinimumOptimized ++
    Seq(
      Dependency[firrtl.transforms.BlackBoxSourceHelper],
      Dependency[firrtl.transforms.FixAddingNegativeLiterals],
      Dependency[firrtl.transforms.ReplaceTruncatingArithmetic],
      Dependency[firrtl.transforms.InlineBitExtractionsTransform],
      Dependency[firrtl.transforms.InlineCastsTransform],
      Dependency[firrtl.transforms.LegalizeClocksTransform],
      Dependency[firrtl.transforms.FlattenRegUpdate]
    )

  override def optionalPrerequisites = firrtl.stage.Forms.LowFormOptimized

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Transform) = false

  private def onModule(m: Module): Module = {
    val namespace = Namespace(m)
    def onStmt(s: Statement): Statement = {
      val v = mutable.ArrayBuffer[Statement]()

      def getWidth(e: Expression): Width = e.tpe match {
        case t: GroundType => t.width
        case t => UnknownWidth
      }

      def maxWidth(ws: Seq[Width]): Width = ws.reduceLeft { (x, y) =>
        (x, y) match {
          case (IntWidth(x), IntWidth(y)) => IntWidth(x.max(y))
          case (x, y)                     => UnknownWidth
        }
      }

      def verilogRemWidth(e: DoPrim)(tpe: Type): Type = {
        val newWidth = maxWidth(e.args.map(exp => getWidth(exp)))
        tpe.mapWidth(w => newWidth)
      }

      def removeRem(e: Expression): Expression = e match {
        case e: DoPrim =>
          e.op match {
            case Rem =>
              val name = namespace.newTemp
              val newType = e.mapType(verilogRemWidth(e))
              v += DefNode(get_info(s), name, e.mapType(verilogRemWidth(e)))
              val remRef = WRef(name, newType.tpe, kind(e), flow(e))
              val remWidth = bitWidth(e.tpe)
              DoPrim(Bits, Seq(remRef), Seq(remWidth - 1, BigInt(0)), e.tpe)
            case _ => e
          }
        case _ => e
      }

      s.map(removeRem) match {
        case x: Block => x.map(onStmt)
        case EmptyStmt => EmptyStmt
        case x =>
          v += x
          v.size match {
            case 1 => v.head
            case _ => Block(v.toSeq)
          }
      }
    }
    Module(m.info, m.name, m.ports, onStmt(m.body))
  }

  def run(c: Circuit): Circuit = {
    val modules = c.modules.map {
      case m: Module    => onModule(m)
      case m: ExtModule => m
    }
    Circuit(c.info, modules, c.main)
  }
}
