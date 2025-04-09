// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import scala.collection.mutable
import dotty.tools.dotc.ast.Trees.*
import dotty.tools.dotc.core.Types.*
import dotty.tools.dotc.core.Flags
import dotty.tools.dotc.core.Symbols.*
import dotty.tools.dotc.core.Constants.*
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.core.Names.TermName
import dotty.tools.dotc.ast.tpd

object ChiselTypeHelpers {

  def shouldMatchGen(bases: Type*)(using Context): Type => Boolean = {
    val cache = mutable.HashMap.empty[Type, Boolean]

    def terminate(t: Type): Boolean = bases.exists(base => t <:< base)

    def outerMatches(t: Type): Boolean = {
      val str = t.show
      str.startsWith("Option[") || str.startsWith("Iterable[")
    }

    def isTupleType(t: Type): Boolean = t.typeSymbol.fullName.startsWith("scala.Tuple")

    def recShouldMatch(tpe: Type, seen: Set[Type]): Boolean = {
      if (terminate(tpe)) true
      else if (seen.contains(tpe)) false
      else if (outerMatches(tpe)) {
        tpe match {
          case AppliedType(_, args) if args.nonEmpty =>
            recShouldMatch(args.head, seen + tpe)
          case _ => false
        }
      } else if (isTupleType(tpe)) {
        tpe match {
          case AppliedType(_, args) =>
            args.exists(arg => recShouldMatch(arg, seen + tpe))
          case _ => false
        }
      } else {
        tpe.baseClasses.exists(sym => recShouldMatch(sym.typeRef, seen))
      }
    }

    def earlyExit(t: Type): Boolean = {
      !outerMatches(t) && !isTupleType(t)
    }

    (q: Type) => {
      cache.getOrElseUpdate(q,
        if (terminate(q)) true
        else if (earlyExit(q)) false
        else recShouldMatch(q, Set.empty)
      )
    }
  }

  def okVal(dd: tpd.ValDef)(using Context): Boolean = {
    val badFlags = Set(Flags.Param, Flags.Synthetic, Flags.Deferred, Flags.CaseAccessor, Flags.ParamAccessor)
    val modsOk = badFlags.forall(f => !dd.symbol.flags.is(f))
    val isNull = dd.rhs match {
      case Literal(Constant(null)) => true
      case _ => false
    }
    modsOk && !isNull && !dd.rhs.isEmpty
  }

  def okUnapply(dd: tpd.ValDef)(using Context): Boolean = {
    val badFlags = Set(Flags.Param, Flags.Deferred, Flags.CaseAccessor, Flags.ParamAccessor)
    val goodFlags = Set(Flags.Synthetic, Flags.Artifact)
    val flagsOk = goodFlags.forall(f => dd.symbol.flags.is(f)) && badFlags.forall(f => !dd.symbol.flags.is(f))
    val isNull = dd.rhs match {
      case Literal(Constant(null)) => true
      case _ => false
    }
    val tpe = dd.tpt.tpe
    tpe.typeSymbol.fullName.startsWith("scala.Tuple") && flagsOk && !isNull && !dd.rhs.isEmpty
  }

  def findUnapplyNames(tree: Tree[?]): Option[List[String]] = {
    val applyArgs = tree match {
      case Match(_, List(CaseDef(_, _, Apply(_, args)))) => Some(args)
      case _ => None
    }
    applyArgs.flatMap { args =>
      var ok = true
      val result = mutable.ListBuffer[String]()
      args.foreach {
        case x: Ident[?] => result += x.name.toString
        case _ => ok = false
      }
      if (ok) Some(result.toList) else None
    }
  }

  def inBundle(dd: tpd.ValDef)(using Context): Boolean = {
    dd.symbol.owner.owner.thisType <:< requiredClassRef("chisel3.Bundle")
  }

  def stringFromTermName(name: TermName): String = name.toString.trim

  def shouldMatchData(using Context): Type => Boolean = {
    val dataTpe = requiredClassRef("chisel3.Data")
    shouldMatchGen(dataTpe)
  }

  def shouldMatchNamedComp(using Context): Type => Boolean = {
    val dataTpe = requiredClassRef("chisel3.Data")
    val memBaseTpe = requiredClassRef("chisel3.MemBase")
    val verifTpe = requiredClassRef("chisel3.VerificationStatement")
    val disableTpe = requiredClassRef("chisel3.Disable")
    val affectsTpe = requiredClassRef("chisel3.experimental.AffectsChiselName")
    val dynObjTpe  = requiredClassRef("chisel3.properties.DynamicObject")

    shouldMatchGen(dataTpe, memBaseTpe, verifTpe, disableTpe, affectsTpe, dynObjTpe)
  }

  def shouldMatchModule(using Context): Type => Boolean = {
    val baseModuleTpe = requiredClassRef("chisel3.experimental.BaseModule")
    shouldMatchGen(baseModuleTpe)
  }

  def shouldMatchInstance(using Context): Type => Boolean = {
    val instanceTpe = requiredClassRef("chisel3.experimental.hierarchy.Instance")
    shouldMatchGen(instanceTpe)
  }

  def shouldMatchChiselPrefixed(using Context): Type => Boolean = {
    val affectsTpe = requiredClassRef("chisel3.experimental.AffectsChiselName")
    shouldMatchGen(affectsTpe)
  }
}
