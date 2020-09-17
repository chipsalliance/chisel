// See LICENSE for license details.

package chisel3.internal.plugin

import scala.tools.nsc
import nsc.{Global, Phase}
import nsc.plugins.Plugin
import nsc.plugins.PluginComponent
import scala.reflect.internal.Flags
import scala.tools.nsc.transform.TypingTransformers

import scala.collection.mutable

// The plugin to be run by the Scala compiler during compilation of Chisel code
class ChiselPlugin(val global: Global) extends Plugin {
  val name = "chiselplugin"
  val description = "Plugin for Chisel 3 Hardware Description Language"
  val components: List[PluginComponent] = List[PluginComponent](new ChiselComponent(global))
}

// The component of the chisel plugin. Not sure exactly what the difference is between
//   a Plugin and a PluginComponent.
class ChiselComponent(val global: Global) extends PluginComponent with TypingTransformers {
  import global._
  val runsAfter: List[String] = List[String]("typer")
  val phaseName: String = "chiselcomponent"
  def newPhase(_prev: Phase): ChiselComponentPhase = new ChiselComponentPhase(_prev)
  class ChiselComponentPhase(prev: Phase) extends StdPhase(prev) {
    override def name: String = phaseName
    def apply(unit: CompilationUnit): Unit = {
      // This plugin doesn't work on Scala 2.11. Rather than complicate the sbt build flow,
      // instead we just check the version and if its an early Scala version, the plugin does nothing
      if(scala.util.Properties.versionNumberString.split('.')(1).toInt >= 12) {
        unit.body = new MyTypingTransformer(unit).transform(unit.body)
      }
    }
  }

  class MyTypingTransformer(unit: CompilationUnit)
    extends TypingTransformer(unit) {

    private def shouldMatchGen(bases: Tree*): ValDef => Boolean = {
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
        } else {
          // This is the standard inheritance hierarchy, Scalac catches loops here
          s.parents.exists( p => recShouldMatch(p, seen) )
        }
      }

      // If doesn't match container pattern, exit early
      def earlyExit(t: Type): Boolean = {
        !(t.matchesPattern(inferType(tq"Iterable[_]")) || t.matchesPattern(inferType(tq"Option[_]")))
      }

      // Return function so that it captures the cache
      { p: ValDef =>
        val q = inferType(p.tpt)
        cache.getOrElseUpdate(q, {
          // First check if a match, then check early exit, then recurse
          if(terminate(q)){
            true
          } else if(earlyExit(q)) {
            false
          } else {
            recShouldMatch(q, Set.empty)
          }
        })
      }
    }

    private val shouldMatchData      : ValDef => Boolean = shouldMatchGen(tq"chisel3.Data")
    private val shouldMatchDataOrMem : ValDef => Boolean = shouldMatchGen(tq"chisel3.Data", tq"chisel3.MemBase[_]")
    private val shouldMatchModule    : ValDef => Boolean = shouldMatchGen(tq"chisel3.experimental.BaseModule")

    // Given a type tree, infer the type and return it
    private def inferType(t: Tree): Type = localTyper.typed(t, nsc.Mode.TYPEmode).tpe

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
        badFlags.forall{ x => !mods.hasFlag(x)}
      }

      // Ensure expression isn't null, as you can't call `null.autoName("myname")`
      val isNull = dd.rhs match {
        case Literal(Constant(null)) => true
        case _ => false
      }

      okFlags(dd.mods) && !isNull && dd.rhs != EmptyTree
    }

    // Whether this val is directly enclosed by a Bundle type
    private def inBundle(dd: ValDef): Boolean = {
      dd.symbol.logicallyEnclosingMember.thisType <:< inferType(tq"chisel3.Bundle")
    }

    // Method called by the compiler to modify source tree
    override def transform(tree: Tree): Tree = tree match {
      // Check if a subtree is a candidate
      case dd @ ValDef(mods, name, tpt, rhs) if okVal(dd) =>
      // If a Data and in a Bundle, just get the name but not a prefix
        if (shouldMatchData(dd) && inBundle(dd)) {
          val TermName(str: String) = name
          val newRHS = transform(rhs)  // chisel3.internal.plugin.autoNameRecursively
          val named = q"chisel3.internal.plugin.autoNameRecursively($str, $newRHS)"
          treeCopy.ValDef(dd, mods, name, tpt, localTyper typed named)
        }
        // If a Data or a Memory, get the name and a prefix
        else if (shouldMatchDataOrMem(dd)) {
          val TermName(str: String) = name
          val newRHS = transform(rhs)
          val prefixed = q"chisel3.experimental.prefix.apply[$tpt](name=$str)(f=$newRHS)"
          val named = q"chisel3.internal.plugin.autoNameRecursively($str, $prefixed)"
          treeCopy.ValDef(dd, mods, name, tpt, localTyper typed named)
        // If an instance, just get a name but no prefix
        } else if (shouldMatchModule(dd)) {
          val TermName(str: String) = name
          val newRHS = transform(rhs)
          val named = q"chisel3.internal.plugin.autoNameRecursively($str, $newRHS)"
          treeCopy.ValDef(dd, mods, name, tpt, localTyper typed named)
        } else {
          // Otherwise, continue
          super.transform(tree)
        }
      // Otherwise, continue
      case _ => super.transform(tree)
    }
  }
}
