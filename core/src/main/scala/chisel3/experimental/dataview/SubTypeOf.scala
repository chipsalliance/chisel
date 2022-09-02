package chisel3.experimental

import chisel3._

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context

trait ChiselSubTypeOf[A, B]

object ChiselSubTypeOf {
  def genChiselSubTypeOf[A: c.WeakTypeTag, B: c.WeakTypeTag](c: Context): c.Tree = {
    import c.universe._

    val ta = implicitly[c.WeakTypeTag[A]]
    val tb = implicitly[c.WeakTypeTag[B]]
    val tdata = implicitly[c.WeakTypeTag[Data]]
    val empty = q""

    // Returns true if 'a' and 'b' are structurally equivalent types.
    def typeEquals(a: Type, b: Type): Boolean = {
      if (a == NoType || b == NoType) {
        return false
      }
      // Two types are equal if they are both subtypes of each other.
      subtypeOf(a, b) && subtypeOf(b, a)
    }

    // Returns true if 'a' is a structural subtype of 'b' (i.e. all fields of
    // 'b' exist within 'a' with the same names and the same types).
    def subtypeOf(a: Type, b: Type): Boolean = {
      // Only look at public members that are getters and that are subtypes of Data.
      val mb = b.members.filter(m => {
        m.isPublic && m.isMethod && m.asMethod.isGetter && m.asMethod.returnType <:< tdata.tpe
      })
      // Go through every public member of b and make sure a member with the
      // same name exists in a and it has the same structural type.
      mb.forall(vb => {
        val name = if (vb.isTerm) TermName(vb.name.toString) else TypeName(vb.name.toString)
        typeEquals(a.member(name).info, vb.info)
      })
    }

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
