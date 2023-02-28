// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Mappers._
import firrtl.options.Dependency
import Utils.throwInternalError

object CInferMDir extends Pass {

  override def prerequisites = firrtl.stage.Forms.ChirrtlForm :+ Dependency(CInferTypes)

  override def invalidates(a: Transform) = false

  type MPortDirMap = collection.mutable.LinkedHashMap[String, MPortDir]

  def infer_mdir_e(mports: MPortDirMap, dir: MPortDir)(e: Expression): Expression = e match {
    case e: Reference =>
      mports.get(e.name) match {
        case None =>
        case Some(p) =>
          mports(e.name) = (p, dir) match {
            case (MInfer, MWrite)         => MWrite
            case (MInfer, MRead)          => MRead
            case (MInfer, MReadWrite)     => MReadWrite
            case (MWrite, MWrite)         => MWrite
            case (MWrite, MRead)          => MReadWrite
            case (MWrite, MReadWrite)     => MReadWrite
            case (MRead, MWrite)          => MReadWrite
            case (MRead, MRead)           => MRead
            case (MRead, MReadWrite)      => MReadWrite
            case (MReadWrite, MWrite)     => MReadWrite
            case (MReadWrite, MRead)      => MReadWrite
            case (MReadWrite, MReadWrite) => MReadWrite
            case _                        => throwInternalError(s"infer_mdir_e: shouldn't be here - $p, $dir")
          }
      }
      e
    case e: SubAccess =>
      infer_mdir_e(mports, dir)(e.expr)
      infer_mdir_e(mports, MRead)(e.index) // index can't be a write port
      e
    case e => e.map(infer_mdir_e(mports, dir))
  }

  def infer_mdir_s(mports: MPortDirMap)(s: Statement): Statement = s match {
    case sx: CDefMPort =>
      mports(sx.name) = sx.direction
      sx.map(infer_mdir_e(mports, MRead))
    case sx: Connect =>
      infer_mdir_e(mports, MRead)(sx.expr)
      infer_mdir_e(mports, MWrite)(sx.loc)
      sx
    case sx: PartialConnect =>
      infer_mdir_e(mports, MRead)(sx.expr)
      infer_mdir_e(mports, MWrite)(sx.loc)
      sx
    case sx => sx.map(infer_mdir_s(mports)).map(infer_mdir_e(mports, MRead))
  }

  def set_mdir_s(mports: MPortDirMap)(s: Statement): Statement = s match {
    case sx: CDefMPort => sx.copy(direction = mports(sx.name))
    case sx => sx.map(set_mdir_s(mports))
  }

  def infer_mdir(m: DefModule): DefModule = {
    val mports = new MPortDirMap
    m.map(infer_mdir_s(mports)).map(set_mdir_s(mports))
  }

  def run(c: Circuit): Circuit =
    c.copy(modules = c.modules.map(infer_mdir))
}
