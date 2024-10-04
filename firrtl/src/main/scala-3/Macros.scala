package firrtl.macros

import scala.quoted._

object Macros {
  inline def isSingletonImpl[T](obj: T): Boolean = ${ isModuleClassImpl('obj) }

  def isModuleClassImpl[T: Type](obj: Expr[T])(using Quotes): Expr[Boolean] = {
    import quotes.reflect._
    val objType = TypeRepr.of[T]
    Expr(objType.typeSymbol.flags.is(Flags.Module))
  }
}
