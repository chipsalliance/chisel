package chisel3.experimental

import chisel3._

import scala.quoted.*

/** Enforces that A is a Chisel subtype of B.
  *
  * A is a Chisel subtype of B if A contains all of B's fields (same
  * names and same types). Only public fields that are subtypes of
  * chisel3.Data are considered when checking for containment.
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
  inline given genChisel[A, B]: ChiselSubtypeOf[A, B] = ${ genChiselImpl[A, B] }

  private def genChiselImpl[A: Type, B: Type](using q: Quotes): Expr[ChiselSubtypeOf[A, B]] = {
    import q.reflect.*

    val a = TypeRepr.of[A]
    val b = TypeRepr.of[B]
    val tdata = TypeRepr.of[Data]

    def baseType(t: TypeRepr): TypeRepr = {
      val bases = t.baseClasses
      if (bases.nonEmpty) t.baseType(bases.head) else TypeRepr.of[Nothing]
    }

    // If one baseType is a subtype the other baseType, then these two
    // types could be equal, so we allow it and leave it to elaboration to
    // figure out. Otherwise, the types must be equal or we throw an error
    def couldBeEqual(av: TypeRepr, bv: TypeRepr): Boolean =
      baseType(bv) <:< baseType(av) || baseType(av) <:< baseType(bv) || av =:= bv

    // Only look at public field accessors (synthetic getter methods for vals/vars)
    // that are subtypes of Data. This mirrors Scala 2's `isGetter` filter and
    // excludes regular def methods like `cloneType`
    val bGetters = b.typeSymbol.fieldMembers.filter { m =>
      val isPublic = !m.flags.is(Flags.Private) && !m.flags.is(Flags.Protected)
      val retType = b.memberType(m).widenByName
      isPublic && retType <:< tdata
    }

    // Go through every public member of b and make sure a member with the
    // same name exists in a and it has the same structural type
    for (vb <- bGetters) {
      val name = vb.name
      val vaSym = a.typeSymbol.fieldMember(name)
      val vbTyp = b.memberType(vb).widenByName
      val vaExists = vaSym != Symbol.noSymbol
      val vaTyp = if (vaExists) a.memberType(vaSym).widenByName else TypeRepr.of[Nothing]

      if (!vaExists || !couldBeEqual(vaTyp, vbTyp)) {
        val aStr = a.show
        val bStr = b.show
        val err =
          if (!vaExists) s"$aStr.$name does not exist"
          else s"${vaTyp.show} != ${vbTyp.show}"
        report.errorAndAbort(
          s"$aStr is not a Chisel subtype of $bStr: mismatch at $bStr.$name: $err. " +
            s"Did you mean .viewAs[$bStr]? " +
            "Please see https://www.chisel-lang.org/chisel3/docs/cookbooks/dataview"
        )
      }
    }

    '{ null.asInstanceOf[ChiselSubtypeOf[A, B]] }
  }
}
