package firrtl.passes

import firrtl.ir._
import firrtl.Mappers._
import firrtl.options.PreservesAll
import firrtl.{Transform, WSubAccess, WSubField, WSubIndex}

object PullMuxes extends Pass with PreservesAll[Transform] {

  override val prerequisites = firrtl.stage.Forms.Deduped

  def run(c: Circuit): Circuit = {
     def pull_muxes_e(e: Expression): Expression = e map pull_muxes_e match {
       case ex: WSubField => ex.expr match {
         case exx: Mux => Mux(exx.cond,
           WSubField(exx.tval, ex.name, ex.tpe, ex.flow),
           WSubField(exx.fval, ex.name, ex.tpe, ex.flow), ex.tpe)
         case exx: ValidIf => ValidIf(exx.cond,
           WSubField(exx.value, ex.name, ex.tpe, ex.flow), ex.tpe)
         case _ => ex  // case exx => exx causes failed tests
       }
       case ex: WSubIndex => ex.expr match {
         case exx: Mux => Mux(exx.cond,
           WSubIndex(exx.tval, ex.value, ex.tpe, ex.flow),
           WSubIndex(exx.fval, ex.value, ex.tpe, ex.flow), ex.tpe)
         case exx: ValidIf => ValidIf(exx.cond,
           WSubIndex(exx.value, ex.value, ex.tpe, ex.flow), ex.tpe)
         case _ => ex  // case exx => exx causes failed tests
       }
       case ex: WSubAccess => ex.expr match {
         case exx: Mux => Mux(exx.cond,
           WSubAccess(exx.tval, ex.index, ex.tpe, ex.flow),
           WSubAccess(exx.fval, ex.index, ex.tpe, ex.flow), ex.tpe)
         case exx: ValidIf => ValidIf(exx.cond,
           WSubAccess(exx.value, ex.index, ex.tpe, ex.flow), ex.tpe)
         case _ => ex  // case exx => exx causes failed tests
       }
       case ex => ex
     }
     def pull_muxes(s: Statement): Statement = s map pull_muxes map pull_muxes_e
     val modulesx = c.modules.map {
       case (m:Module) => Module(m.info, m.name, m.ports, pull_muxes(m.body))
       case (m:ExtModule) => m
     }
     Circuit(c.info, modulesx, c.main)
   }
}
