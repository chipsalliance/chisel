package firrtl.passes

import firrtl.ir._
import firrtl.Mappers._
import firrtl.options.{PreservesAll}
import firrtl.{Transform, UnknownFlow, UnknownKind, WDefInstance, WRef, WSubAccess, WSubField, WSubIndex}

// These should be distributed into separate files
object ToWorkingIR extends Pass with PreservesAll[Transform] {

  override val prerequisites = firrtl.stage.Forms.MinimalHighForm

  def toExp(e: Expression): Expression = e map toExp match {
    case ex: Reference => WRef(ex.name, ex.tpe, UnknownKind, UnknownFlow)
    case ex: SubField => WSubField(ex.expr, ex.name, ex.tpe, UnknownFlow)
    case ex: SubIndex => WSubIndex(ex.expr, ex.value, ex.tpe, UnknownFlow)
    case ex: SubAccess => WSubAccess(ex.expr, ex.index, ex.tpe, UnknownFlow)
    case ex => ex // This might look like a case to use case _ => e, DO NOT!
  }

  def toStmt(s: Statement): Statement = s map toExp match {
    case sx: DefInstance => WDefInstance(sx.info, sx.name, sx.module, UnknownType)
    case sx => sx map toStmt
  }

  def run (c:Circuit): Circuit =
    c copy (modules = c.modules map (_ map toStmt))
}
