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
    tpe.typeSymbol.fullName.startsWith("Tuple") && flagsOk && !isNull && !dd.rhs.isEmpty
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

  def isData(t: Type)(using Context): Boolean = {
    val dataTpe = getClassIfDefined("chisel3.Data")
    t.baseClasses.contains(dataTpe)
  }

  def isNamed(t: Type)(using Context): Boolean = {
    val dataTpe =    getClassIfDefined("chisel3.Data")
    val memBaseTpe = getClassIfDefined("chisel3.MemBase")
    val verifTpe =   getClassIfDefined("chisel3.VerificationStatement")
    val disableTpe = getClassIfDefined("chisel3.Disable")
    val affectsTpe = getClassIfDefined("chisel3.experimental.AffectsChiselName")
    val dynObjTpe  = getClassIfDefined("chisel3.properties.DynamicObject")

    t.baseClasses.contains(dataTpe)
    || t.baseClasses.contains(memBaseTpe)
    || t.baseClasses.contains(verifTpe)
    || t.baseClasses.contains(disableTpe)
    || t.baseClasses.contains(affectsTpe)
    || t.baseClasses.contains(dynObjTpe)
  }

  def isModule(t: Type)(using Context): Boolean = {
    val baseModuleTpe = getClassIfDefined("chisel3.experimental.BaseModule")
    t.baseClasses.contains(baseModuleTpe)
  }

  def isInstance(t: Type)(using Context): Boolean = {
    val instanceTpe = getClassIfDefined("chisel3.experimental.hierarchy.Instance")
    t.baseClasses.contains(instanceTpe)
  }

  def isPrefixed(t: Type)(using Context): Boolean = {
    val affectsTpe = getClassIfDefined("chisel3.experimental.AffectsChiselName")
    t.baseClasses.contains(affectsTpe)
  }
}
