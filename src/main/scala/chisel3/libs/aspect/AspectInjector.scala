package chisel3.libs.aspect

import firrtl.annotations.Component
import firrtl.ir.{DefModule, Expression, Statement}

abstract class AspectInjector {
  def onStmt(c: Component)(s: Statement): Statement = s
  def onExp(c: Component)(e: Expression): Expression = e
  def onModule(c: Component)(m: DefModule): DefModule = m
}
