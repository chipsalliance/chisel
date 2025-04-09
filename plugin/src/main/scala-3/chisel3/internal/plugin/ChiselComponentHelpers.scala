// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import scala.quoted.*
import scala.collection.mutable
import dotty.tools.dotc.ast.Trees.*
import dotty.tools.dotc.core.Types.*
import dotty.tools.dotc.core.Flags
import dotty.tools.dotc.core.Symbols.*
import dotty.tools.dotc.core.Constants.*
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.ast.tpd

object ChiselTypeHelpers {

  def shouldMatchGen(bases: Type*): Type => Boolean = {
    val cache = mutable.HashMap.empty[Type, Boolean]

    def terminate(t: Type): Boolean = {
      bases.exists(base => t <:< base)
    }

    def outerMatches(t: quotes.reflect.TypeRepr): Boolean = {
      val str = t.show
      str.startsWith("Option[") || str.startsWith("Iterable[")
    }

    def recShouldMatch(tpe: quotes.reflect.TypeRepr, seen: Set[quotes.reflect.TypeRepr]): Boolean = {
      if (terminate(tpe)) {
        true
      } else if (seen.contains(tpe)) {
        false
      } else if (outerMatches(tpe)) {
        tpe match {
          case quotes.reflect.AppliedType(_, args) if args.nonEmpty =>
            recShouldMatch(args.head, seen + tpe)
          case _ => false
        }
      } else if (isTupleType(tpe)) {
        tpe match {
          case quotes.reflect.AppliedType(_, args) =>
            args.exists(arg => recShouldMatch(arg, seen + tpe))
          case _ => false
        }
      } else {
        tpe.baseClasses.exists { sym =>
          val parentTpe = sym.typeRef
          recShouldMatch(parentTpe, seen)
        }
      }
    }

    def earlyExit(t: quotes.reflect.TypeRepr): Boolean = {
      !outerMatches(t) && !isTupleType(t)
    }

    (q: quotes.reflect.TypeRepr) => {
      cache.getOrElseUpdate(q,
        if (terminate(q)) {
          true
        } else if (earlyExit(q)) {
          false
        } else {
          recShouldMatch(q, Set.empty)
        }
      )
    }
  }

  def isTupleType(using Quotes)(tpe: quotes.reflect.TypeRepr): Boolean = {
    tpe.typeSymbol.fullName.startsWith("scala.Tuple")
  }

  def okVal(using Quotes)(vd: quotes.reflect.ValDef): Boolean = {
    import quotes.reflect.Flags
    val badFlags = Set(Flags.Param, Flags.Synthetic, Flags.Deferred, Flags.CaseAccessor, Flags.ParamAccessor)
    val modsOk = badFlags.forall(f => !vd.symbol.flags.is(f))
    val rhsNotNull = vd.rhs.exists {
      case lit: quotes.reflect.Literal =>
        lit.constant.value != null
      case _ => true
    }
    modsOk && rhsNotNull && vd.rhs.isDefined
  }

  def okUnapply(using Quotes)(vd: quotes.reflect.ValDef): Boolean = {
    import quotes.reflect.Flags
    val badFlags = Set(Flags.Param, Flags.Deferred, Flags.CaseAccessor, Flags.ParamAccessor)
    val goodFlags = Set(Flags.Synthetic, Flags.Artifact)
    val flagsOk = goodFlags.forall(f => vd.symbol.flags.is(f)) && badFlags.forall(f => !vd.symbol.flags.is(f))
    val rhsNotNull = vd.rhs.exists {
      case lit: quotes.reflect.Literal =>
        lit.constant.value != null
      case _ => true
    }
    val tpe = vd.tpt.tpe
    isTupleType(tpe) && flagsOk && rhsNotNull && vd.rhs.isDefined
  }

  def findUnapplyNames(using Quotes)(tree: quotes.reflect.Tree): Option[List[String]] = {
    import quotes.reflect.*
    tree match {
      case Match(_, cases) => {
        cases.headOption match {
          case Some(CaseDef(_, _, Apply(_, args))) => {
            val names = args.collect {
              case Ident(TermRef(_, name)) => name
            }
            if (names.size == args.size) Some(names.map(_.toString)) else None
          }
          case _ => None
        }
      }
      case _ => None
    }
  }

  def inBundle(using Quotes)(vd: quotes.reflect.ValDef): Boolean = {
    vd <:< chisel3.Bundle
  }

  def stringFromTermName(using Quotes)(name: quotes.reflect.TermRef): String = {
    name.name.toString.trim
  }

  inline def shouldMatchData(using Quotes): quotes.reflect.TypeRepr => Boolean =
    shouldMatchGen(quotes.reflect.TypeRepr.of[chisel3.Data])

  inline def shouldMatchNamedComp(using Quotes): quotes.reflect.TypeRepr => Boolean =
    shouldMatchGen(
      quotes.reflect.TypeRepr.of[chisel3.Data],
      quotes.reflect.TypeRepr.of[chisel3.MemBase[?]],
      quotes.reflect.TypeRepr.of[chisel3.VerificationStatement],
      quotes.reflect.TypeRepr.of[chisel3.properties.DynamicObject],
      quotes.reflect.TypeRepr.of[chisel3.Disable],
      quotes.reflect.TypeRepr.of[chisel3.experimental.AffectsChiselName]
    )

  inline def shouldMatchModule(using Quotes): quotes.reflect.TypeRepr => Boolean =
    shouldMatchGen(quotes.reflect.TypeRepr.of[chisel3.experimental.BaseModule])

  inline def shouldMatchInstance(using Quotes): quotes.reflect.TypeRepr => Boolean =
    shouldMatchGen(quotes.reflect.TypeRepr.of[chisel3.experimental.hierarchy.Instance[?]])

  inline def shouldMatchChiselPrefixed(using Quotes): quotes.reflect.TypeRepr => Boolean =
    shouldMatchGen(quotes.reflect.TypeRepr.of[chisel3.experimental.AffectsChiselPrefix])
}
