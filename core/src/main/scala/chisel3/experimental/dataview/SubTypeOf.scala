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

    def baseType(t: Type): Type = t.baseType(t.baseClasses(0))

    // Returns true if 'a' is a structural subtype of 'b' (i.e. all fields of
    // 'b' exist within 'a' with the same names and the same types).
    def subtypeOf(a: Type, b: Type) = {
      // Only look at public members that are getters and that are subtypes of Data.
      val mb = b.members.filter(m => {
        m.isPublic && m.isMethod && m.asMethod.isGetter && m.asMethod.returnType <:< tdata.tpe
      })
      // Go through every public member of b and make sure a member with the
      // same name exists in a and it has the same structural type.
      for (vb <- mb) {
        val name = TermName(vb.name.toString)
        val vaTyp = a.member(name).info.resultType
        val vbTyp = vb.info.resultType
        // If one baseType is a subtype the other baseType, then these two
        // types could be equal, so we allow it and leave it to elaboration to
        // figure out. Otherwise, the types must be equal or we throw an error.
        if (vaTyp == NoType || (!(baseType(vbTyp) <:< baseType(vaTyp) || baseType(vaTyp) <:< baseType(vbTyp))) && !(vaTyp =:= vbTyp)) {
          val err = if (vaTyp == NoType) s"${ta.tpe}.${name} does not exist" else s"${vaTyp} != ${vbTyp}"
          c.error(
            empty.pos,
            s"${ta.tpe} is not a Chisel subtype of ${tb.tpe}: mismatch at ${tb.tpe}.${name}: $err. Did you mean .viewAs[${tb.tpe}]? " +
              "Please see https://www.chisel-lang.org/chisel3/docs/cookbooks/dataview"
          )
        }
      }
    }
    subtypeOf(ta.tpe, tb.tpe)

    return empty
  }
  implicit def genChisel[A, B]: ChiselSubTypeOf[A, B] = macro ChiselSubTypeOf.genChiselSubTypeOf[A, B]
}
