package chisel3.experimental

import chisel3._

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context

/** Enforces that A is a Chisel subtype of B.
  *
  * A is a Chisel subtype of B if A contains all of B's fields (same names and
  * same types). Only public fields that are subtypes of chisel3.Data are
  * considered when checking for containment.
  *
  * In the following example A is a Chisel subtype of B:
  *
  *  {{{
  *    class A extends Bundle {
  *      val x = UInt(3.W)
  *      val y = UInt(3.W)
  *      val z = UInt(3.W)
  *    }
  *    class B extends Bundle {
  *      val x = UInt(3.W)
  *      val y = UInt(3.W)
  *    }
  *  }}}
  */
sealed trait ChiselSubtypeOf[A, B]

object ChiselSubtypeOf {
  def genChiselSubtypeOf[A: c.WeakTypeTag, B: c.WeakTypeTag](c: Context): c.Tree = {
    import c.universe._

    def baseType(t: Type): Type =
      if (t.baseClasses.length > 0)
        t.baseType(t.baseClasses(0))
      else
        NoType

    def couldBeEqual(a: Type, b: Type): Boolean =
      // If one baseType is a subtype the other baseType, then these two
      // types could be equal, so we allow it and leave it to elaboration to
      // figure out. Otherwise, the types must be equal or we throw an error.
      (baseType(b) <:< baseType(a) || baseType(a) <:< baseType(b)) || a =:= b

    val a = implicitly[c.WeakTypeTag[A]].tpe
    val b = implicitly[c.WeakTypeTag[B]].tpe
    val tdata = implicitly[c.WeakTypeTag[Data]].tpe

    // Only look at public members that are getters and that are subtypes of Data.
    val mb = b.members.filter(m => {
      m.isPublic && m.isMethod && m.asMethod.isGetter && m.asMethod.returnType <:< tdata
    })
    // Go through every public member of b and make sure a member with the
    // same name exists in a and it has the same structural type.
    for (vb <- mb) {
      val name = TermName(vb.name.toString)
      val vaTyp = a.member(name).info.resultType
      val vbTyp = vb.info.resultType
      if (vaTyp == NoType || vbTyp == NoType || !couldBeEqual(vaTyp, vbTyp)) {
        val err = if (vaTyp == NoType) s"${a}.${name} does not exist" else s"${vaTyp} != ${vbTyp}"
        c.error(
          c.enclosingPosition,
          s"${a} is not a Chisel subtype of ${b}: mismatch at ${b}.${name}: $err. Did you mean .viewAs[${b}]? " +
            "Please see https://www.chisel-lang.org/chisel3/docs/cookbooks/dataview"
        )
      }
    }

    q""
  }
  implicit def genChisel[A, B]: ChiselSubtypeOf[A, B] = macro ChiselSubtypeOf.genChiselSubtypeOf[A, B]
}
