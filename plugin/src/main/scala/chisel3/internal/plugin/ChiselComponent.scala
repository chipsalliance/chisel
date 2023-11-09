// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import scala.collection.mutable
import scala.reflect.internal.Flags
import scala.tools.nsc
import scala.tools.nsc.{Global, Phase}
import scala.tools.nsc.plugins.PluginComponent
import scala.tools.nsc.transform.TypingTransformers

import chisel3.internal.sourceinfo.SourceInfoFileResolver

// The component of the chisel plugin. Not sure exactly what the difference is between
//   a Plugin and a PluginComponent.
class ChiselComponent(val global: Global, arguments: ChiselPluginArguments)
    extends PluginComponent
    with TypingTransformers
    with ChiselOuterUtils {
  import global._
  val runsAfter: List[String] = List[String]("typer")
  val phaseName: String = "chiselcomponent"
  def newPhase(_prev: Phase): ChiselComponentPhase = new ChiselComponentPhase(_prev)
  class ChiselComponentPhase(prev: Phase) extends StdPhase(prev) {
    override def name: String = phaseName
    def apply(unit: CompilationUnit): Unit = {
      if (ChiselPlugin.runComponent(global, arguments)(unit)) {
        unit.body = new MyTypingTransformer(unit).transform(unit.body)
      }
    }
  }

  class MyTypingTransformer(unit: CompilationUnit) extends TypingTransformer(unit) with ChiselInnerUtils {

    private def shouldMatchGen(bases: Tree*): Type => Boolean = {
      val cache = mutable.HashMap.empty[Type, Boolean]
      val baseTypes = bases.map(inferType)

      // If subtype of one of the base types, it's a match!
      def terminate(t: Type): Boolean = baseTypes.exists(t <:< _)

      // Recurse through subtype hierarchy finding containers
      // Seen is only updated when we recurse into type parameters, thus it is typically small
      def recShouldMatch(s: Type, seen: Set[Type]): Boolean = {
        def outerMatches(t: Type): Boolean = {
          val str = t.toString
          str.startsWith("Option[") || str.startsWith("Iterable[")
        }
        if (terminate(s)) {
          true
        } else if (seen.contains(s)) {
          false
        } else if (outerMatches(s)) {
          // These are type parameters, loops *are* possible here
          recShouldMatch(s.typeArgs.head, seen + s)
        } else if (definitions.isTupleType(s)) {
          s.typeArgs.exists(t => recShouldMatch(t, seen + s))
        } else {
          // This is the standard inheritance hierarchy, Scalac catches loops here
          s.parents.exists(p => recShouldMatch(p, seen))
        }
      }

      // If doesn't match container pattern, exit early
      def earlyExit(t: Type): Boolean = {
        !(t.matchesPattern(inferType(tq"Iterable[_]")) || t.matchesPattern(inferType(tq"Option[_]")) || definitions
          .isTupleType(t))
      }

      // Return function so that it captures the cache
      { q: Type =>
        cache.getOrElseUpdate(
          q, {
            // First check if a match, then check early exit, then recurse
            if (terminate(q)) {
              true
            } else if (earlyExit(q)) {
              false
            } else {
              recShouldMatch(q, Set.empty)
            }
          }
        )
      }
    }

    private val shouldMatchData: Type => Boolean = shouldMatchGen(tq"chisel3.Data")
    // Checking for all chisel3.internal.NamedComponents, but since it is internal, we instead have
    // to match the public subtypes
    private val shouldMatchNamedComp: Type => Boolean =
      shouldMatchGen(
        tq"chisel3.Data",
        tq"chisel3.MemBase[_]",
        tq"chisel3.VerificationStatement",
        tq"chisel3.properties.DynamicObject",
        tq"chisel3.Disable"
      )
    private val shouldMatchModule:   Type => Boolean = shouldMatchGen(tq"chisel3.experimental.BaseModule")
    private val shouldMatchInstance: Type => Boolean = shouldMatchGen(tq"chisel3.experimental.hierarchy.Instance[_]")
    private val shouldMatchChiselPrefixed: Type => Boolean =
      shouldMatchGen(
        tq"chisel3.experimental.AffectsChiselPrefix"
      )

    // Indicates whether a ValDef is properly formed to get name
    private def okVal(dd: ValDef): Boolean = {

      // These were found through trial and error
      def okFlags(mods: Modifiers): Boolean = {
        val badFlags = Set(
          Flag.PARAM,
          Flag.SYNTHETIC,
          Flag.DEFERRED,
          Flags.TRIEDCOOKING,
          Flags.CASEACCESSOR,
          Flags.PARAMACCESSOR
        )
        badFlags.forall { x => !mods.hasFlag(x) }
      }

      // Ensure expression isn't null, as you can't call `null.autoName("myname")`
      val isNull = dd.rhs match {
        case Literal(Constant(null)) => true
        case _                       => false
      }

      okFlags(dd.mods) && !isNull && dd.rhs != EmptyTree
    }
    // TODO Unify with okVal
    private def okUnapply(dd: ValDef): Boolean = {

      // These were found through trial and error
      def okFlags(mods: Modifiers): Boolean = {
        val badFlags = Set(
          Flag.PARAM,
          Flag.DEFERRED,
          Flags.TRIEDCOOKING,
          Flags.CASEACCESSOR,
          Flags.PARAMACCESSOR
        )
        val goodFlags = Set(
          Flag.SYNTHETIC,
          Flag.ARTIFACT
        )
        goodFlags.forall(f => mods.hasFlag(f)) && badFlags.forall(f => !mods.hasFlag(f))
      }

      // Ensure expression isn't null, as you can't call `null.autoName("myname")`
      val isNull = dd.rhs match {
        case Literal(Constant(null)) => true
        case _                       => false
      }
      val tpe = inferType(dd.tpt)
      definitions.isTupleType(tpe) && okFlags(dd.mods) && !isNull && dd.rhs != EmptyTree
    }

    private def findUnapplyNames(tree: Tree): Option[List[String]] = {
      val applyArgs: Option[List[Tree]] = tree match {
        case Match(_, List(CaseDef(_, _, Apply(_, args)))) => Some(args)
        case _                                             => None
      }
      applyArgs.flatMap { args =>
        var ok = true
        val result = mutable.ListBuffer[String]()
        args.foreach {
          case Ident(TermName(name)) => result += name
          // Anything unexpected and we abort
          case _ => ok = false
        }
        if (ok) Some(result.toList) else None
      }
    }

    // Whether this val is directly enclosed by a Bundle type
    private def inBundle(dd: ValDef): Boolean = {
      dd.symbol.logicallyEnclosingMember.thisType <:< inferType(tq"chisel3.Bundle")
    }

    private def stringFromTermName(name: TermName): String =
      name.toString.trim() // Remove trailing space (Scalac implementation detail)

    // Method called by the compiler to modify source tree
    override def transform(tree: Tree): Tree = tree match {
      // Check if a subtree is a candidate
      case dd @ ValDef(mods, name, tpt, rhs) if okVal(dd) =>
        val tpe = inferType(tpt)
        val isData = shouldMatchData(tpe)
        val isNamedComp = isData || shouldMatchNamedComp(tpe)
        val isPrefixed = isNamedComp || shouldMatchChiselPrefixed(tpe)

        // If a Data and in a Bundle, just get the name but not a prefix
        if (isData && inBundle(dd)) {
          val str = stringFromTermName(name)
          val newRHS = transform(rhs) // chisel3.internal.plugin.autoNameRecursively
          val named = q"chisel3.internal.plugin.autoNameRecursively($str)($newRHS)"
          treeCopy.ValDef(dd, mods, name, tpt, localTyper.typed(named))
        }
        // If a Data or a Memory, get the name and a prefix
        else if (isData || isPrefixed) {
          val str = stringFromTermName(name)
          // Starting with '_' signifies a temporary, we ignore it for prefixing because we don't
          // want double "__" in names when the user is just specifying a temporary
          val prefix = if (str.head == '_') str.tail else str
          val newRHS = transform(rhs)
          val prefixed = q"chisel3.experimental.prefix.apply[$tpt](name=$prefix)(f=$newRHS)"

          val named =
            if (isNamedComp) {
              // Only name named components (not things that are merely prefixed)
              q"chisel3.internal.plugin.autoNameRecursively($str)($prefixed)"
            } else {
              prefixed
            }

          treeCopy.ValDef(dd, mods, name, tpt, localTyper.typed(named))
        }
        // If an instance, just get a name but no prefix
        else if (shouldMatchModule(tpe)) {
          val str = stringFromTermName(name)
          val newRHS = transform(rhs)
          val named = q"chisel3.internal.plugin.autoNameRecursively($str)($newRHS)"
          treeCopy.ValDef(dd, mods, name, tpt, localTyper.typed(named))
        } else if (shouldMatchInstance(tpe)) {
          val str = stringFromTermName(name)
          val newRHS = transform(rhs)
          val named = q"chisel3.internal.plugin.autoNameRecursively($str)($newRHS)"
          treeCopy.ValDef(dd, mods, name, tpt, localTyper.typed(named))
        } else {
          // Otherwise, continue
          super.transform(tree)
        }
      case dd @ ValDef(mods, name, tpt, rhs @ Match(_, _)) if okUnapply(dd) =>
        val tpe = inferType(tpt)
        val fieldsOfInterest: List[Boolean] = tpe.typeArgs.map(shouldMatchData)
        // Only transform if at least one field is of interest
        if (fieldsOfInterest.reduce(_ || _)) {
          findUnapplyNames(rhs) match {
            case Some(names) =>
              val onames: List[Option[String]] =
                fieldsOfInterest.zip(names).map { case (ok, name) => if (ok) Some(name) else None }
              val newRHS = transform(rhs)
              val named = q"chisel3.internal.plugin.autoNameRecursivelyProduct($onames)($newRHS)"
              treeCopy.ValDef(dd, mods, name, tpt, localTyper.typed(named))
            case None => // It's not clear how this could happen but we don't want to crash
              super.transform(tree)
          }
        } else {
          super.transform(tree)
        }
      // Also look for Module class definitions for inserting source locators
      case module: ClassDef if isAModule(module.symbol) && !module.mods.hasFlag(Flag.ABSTRACT) =>
        val path = SourceInfoFileResolver.resolve(module.pos.source)
        val info = localTyper.typed(q"chisel3.experimental.SourceLine($path, ${module.pos.line}, ${module.pos.column})")

        val sourceInfoSym =
          module.symbol.newMethod(TermName("_sourceInfo"), module.symbol.pos.focus, Flag.OVERRIDE | Flag.PROTECTED)
        sourceInfoSym.resetFlag(Flags.METHOD)
        sourceInfoSym.setInfo(NullaryMethodType(sourceInfoTpe))
        val sourceInfoImpl = localTyper.typed(
          DefDef(sourceInfoSym, info)
        )

        val moduleWithInfo = deriveClassDef(module) { t =>
          deriveTemplate(t)(sourceInfoImpl :: _)
        }
        super.transform(localTyper.typed(moduleWithInfo))

      // Otherwise, continue
      case _ => super.transform(tree)
    }
  }
}
