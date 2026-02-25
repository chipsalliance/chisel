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

  val badFlagsVal = Set(
    Flags.Param,
    Flags.Synthetic,
    Flags.Deferred,
    Flags.CaseAccessor,
    Flags.ParamAccessor
  )

  val badFlagsUnapply = Set(
    Flags.Param,
    Flags.Deferred,
    Flags.CaseAccessor,
    Flags.ParamAccessor
  )

  val goodFlagsUnapply = Set(Flags.Synthetic)

  def okVal(dd: tpd.ValDef)(using Context): Boolean = {
    val modsOk = badFlagsVal.forall(f => !dd.symbol.flags.is(f))
    val isNull = dd.rhs match {
      case Literal(Constant(null)) => true
      case _                       => false
    }
    modsOk && !isNull && !dd.rhs.isEmpty
  }

  def okUnapply(dd: tpd.ValDef)(using Context): Boolean = {
    val flagsOk =
      goodFlagsUnapply.forall(f => dd.symbol.flags.is(f))
        && badFlagsUnapply.forall(f => !dd.symbol.flags.is(f))
    val isNull = dd.rhs match {
      case Literal(Constant(null)) => true
      case _                       => false
    }

    val tpe = dd.tpt.tpe

    // In Scala 3, synthetic unapply temps have names like $1$, $2$, ...
    val name = dd.name.toString
    val isSyntheticTempName = name.startsWith("$") && name.endsWith("$") && name.length > 2

    isTupleType(tpe)
    && flagsOk
    && isSyntheticTempName
    && !isNull
    && !dd.rhs.isEmpty
  }

  def findUnapplyNames(tree: Tree[?]): Option[List[String]] = {
    val applyArgs = tree match {
      case Match(_, List(CaseDef(_, _, Apply(_, args)))) => Some(args)
      case _                                             => None
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

  // Check if a symbol is exactly the Bundle class and not a subclass
  def isExactBundle(sym: Symbol)(using Context): Boolean = {
    val bundleTpe = requiredClass("chisel3.Bundle")
    sym == bundleTpe
  }

  def enclosingMember(sym: Symbol)(using Context): Symbol = {
    if sym.isClass || sym.is(Flags.Method) then sym
    else if sym.exists then enclosingMember(sym.owner)
    else NoSymbol
  }

  def inBundle(sym: Symbol)(using Context): Boolean = {
    enclosingMember(sym).thisType <:< requiredClassRef("chisel3.Bundle")
  }

  def stringFromTermName(name: TermName): String = name.toString.trim

  def isData(t: Type)(using Context): Boolean = {
    val dataTpe = getClassIfDefined("chisel3.Data")
    t.baseClasses.contains(dataTpe)
  }

  def isIgnoreSeq(t: Type)(using Context): Boolean = {
    val ignoreSeqTpe = getClassIfDefined("chisel3.IgnoreSeqInBundle")
    t.baseClasses.contains(ignoreSeqTpe)
  }

  def isBoxedData(tpe: Type, ignoreSeq: Boolean = false)(using Context): Boolean = {
    val optionClass = getClassIfDefined("scala.Option")
    val iterableClass = getClassIfDefined("scala.collection.Iterable")

    // Check if a type is data or a tuple that contains Data
    def containsData(t: Type): Boolean = {
      isData(t) || (isTupleType(t) && tupleHasData(t))
    }

    // Check if any element of a tuple type is a Data or contains Data
    def tupleHasData(t: Type): Boolean = t match {
      case AppliedType(_, args) =>
        args.exists(arg => isData(arg) || (isTupleType(arg) && tupleHasData(arg)))
      case _ => false
    }

    // Recursion here is needed only for nested iterables. These
    // nested iterable may or may not have Data in their leaves.
    def rec(tpe: Type): Boolean = {
      tpe match {
        case AppliedType(tycon, List(arg)) =>
          tycon match {
            case tp: TypeRef =>
              val isIterable = tp.symbol.derivesFrom(iterableClass)
              val isOption = tp.symbol == optionClass

              (isOption, isIterable, containsData(arg)) match {
                case (true, false, true)  => true // Option with Data or tuple containing Data
                case (false, true, true)  => !ignoreSeq // Iterable with Data
                case (false, true, false) => rec(arg) // Possibly nested iterable
                case _                    => false
              }
            case _ =>
              // anonymous subtype of Iterable,
              // or AppliedType from higher-kinded type
              rec(arg)
          }

        case tr: TypeRef =>
          // Follow abstract type aliases like trait Seq[A] by looking at their info
          tr.info match {
            case TypeBounds(lo, hi) if lo != hi =>
              // Try upper bound if available
              rec(hi)
            case TypeAlias(alias) =>
              rec(alias)
            case _ =>
              false
          }
        case _ =>
          false
      }
    }
    rec(tpe)
  }

  def isRecord(t: Type)(using Context): Boolean = {
    val recordTpe = requiredClass("chisel3.Record")
    t.baseClasses.contains(recordTpe)
  }

  def isBundle(t: Type)(using Context): Boolean = {
    val bundleTpe = requiredClass("chisel3.Bundle")
    t.baseClasses.contains(bundleTpe)
  }

  def isTupleType(t: Type)(using Context): Boolean = {
    ctx.definitions.isTupleClass(t.typeSymbol)
  }

  // Get the arity of a tuple type, or 0 if not a tuple
  def tupleArity(t: Type)(using Context): Int = {
    t match {
      case AppliedType(_, args) if isTupleType(t) => args.length
      case _                                      => 0
    }
  }

  // Check if type is directly a NamedComponent (not recursive through tuples)
  private def isDirectlyNamed(t: Type)(using Context): Boolean = {
    val dataTpe = getClassIfDefined("chisel3.Data")
    val memBaseTpe = getClassIfDefined("chisel3.MemBase")
    val verifTpe = getClassIfDefined("chisel3.VerificationStatement")
    val disableTpe = getClassIfDefined("chisel3.Disable")
    val affectsTpe = getClassIfDefined("chisel3.experimental.AffectsChiselName")
    val dynObjTpe = getClassIfDefined("chisel3.properties.DynamicObject")

    t.baseClasses.contains(dataTpe)
    || t.baseClasses.contains(memBaseTpe)
    || t.baseClasses.contains(verifTpe)
    || t.baseClasses.contains(disableTpe)
    || t.baseClasses.contains(affectsTpe)
    || t.baseClasses.contains(dynObjTpe)
  }

  def isNamed(t: Type)(using Context): Boolean = {
    isDirectlyNamed(t) || (isTupleType(t) && tupleHasNamed(t))
  }

  // Check if any element of a tuple type is a NamedComponent
  def tupleHasNamed(t: Type)(using Context): Boolean = t match {
    case AppliedType(_, args) =>
      args.exists { arg =>
        isDirectlyNamed(arg) || (isTupleType(arg) && tupleHasNamed(arg))
      }
    case _ => false
  }

  // Get per-element naming info for tuple types (true if element should be named)
  def tupleFieldsOfInterest(t: Type)(using Context): List[Boolean] = t match {
    case AppliedType(_, args) =>
      args.map { arg =>
        isDirectlyNamed(arg) || (isTupleType(arg) && tupleHasNamed(arg))
      }
    case _ => Nil
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
