package chisel3.internal

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context

trait ChiselSubTypeOf[A, B]

object ChiselSubTypeOf {
  def genChiselSubTypeOf[A: c.WeakTypeTag, B: c.WeakTypeTag](c: Context): c.Tree = {
    import c.universe._

    def subtypeOf(a: Type, b: Type): Boolean = {
      if (a == NoType) {
        return false
      }

      val va = a.members
      val vb = b.members

      for (bval <- vb) {
        if (bval.isPublic) {
          if (bval.isTerm && bval.asTerm.isGetter) {
            val aval = a.member(TermName(bval.name.toString()))
            if (!subtypeOf(aval.info, bval.info)) {
              return false
            }
          } else if (!bval.isTerm) {
            val aval = a.member(TypeName(bval.name.toString()))
            if (aval.info != bval.info) {
              return false
            }
          }
        }
      }
      return true
    }

    val ta = implicitly[c.WeakTypeTag[A]]
    val tb = implicitly[c.WeakTypeTag[B]]
    val empty = q""

    if (!subtypeOf(ta.tpe, tb.tpe)) {
      c.error(
        empty.pos,
        s"${ta.tpe} is not a Chisel subtype of ${tb.tpe}. Did you mean .viewAs[${tb.tpe}]? " +
          "Please see https://www.chisel-lang.org/chisel3/docs/cookbooks/dataview"
      )
    }

    return empty
  }
  implicit def genChisel[A, B]: ChiselSubTypeOf[A, B] = macro ChiselSubTypeOf.genChiselSubTypeOf[A, B]
}
