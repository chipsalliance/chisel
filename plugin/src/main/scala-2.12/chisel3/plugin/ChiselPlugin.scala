// See LICENSE for license details.

package chisel3.plugin

import scala.tools.nsc
import nsc.{Global, Phase}
import nsc.plugins.Plugin
import nsc.plugins.PluginComponent
import scala.reflect.internal.Flags
import scala.tools.nsc.transform.TypingTransformers

// The plugin to be run by the Scala compiler during compilation of Chisel code
class ChiselPlugin(val global: Global) extends Plugin {
  val name = "chiselplugin"
  val description = "Plugin for Chisel 3 Hardware Description Language"
  val components = List[PluginComponent](new ChiselComponent(global))
}

// The component of the chisel plugin. Not sure exactly what the difference is between
//   a Plugin and a PluginComponent.
class ChiselComponent(val global: Global) extends PluginComponent with TypingTransformers {
  import global._
  val runsAfter = List[String]("typer")
  override val runsRightAfter: Option[String] = Some("typer")
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

    // Determines if the chisel plugin should match on this type
    def shouldMatch(q: Type, bases: Seq[Tree]): Boolean = {

      // If subtype of Data or BaseModule, its a match!
      def terminate(t: Type): Boolean = bases.exists { base => t <:< inferType(base) }

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

      // First check if a match, then check early exit, then recurse
      if(terminate(q)){
        true
      } else if(earlyExit(q)) {
        false
      } else {
        recShouldMatch(q, Set.empty)
      }
    }

    // Given a type tree, infer the type and return it
    def inferType(t: Tree): Type = localTyper.typed(t, nsc.Mode.TYPEmode).tpe

    // Indicates whether a ValDef is properly formed to get name
    def okVal(dd: ValDef, bases: Tree*): Boolean = {

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
      okFlags(dd.mods) && shouldMatch(inferType(dd.tpt), bases) && !isNull && dd.rhs != EmptyTree
    }

    // Whether this val is directly enclosed by a Bundle type
    def inBundle(dd: ValDef): Boolean = {
      dd.symbol.logicallyEnclosingMember.thisType <:< inferType(tq"chisel3.Bundle")
    }

    // Method called by the compiler to modify source tree
    override def transform(tree: Tree): Tree = tree match {
      // If a Data and in a Bundle, just get the name but not a prefix
      case dd @ ValDef(mods, name, tpt, rhs) if okVal(dd, tq"chisel3.Data") && inBundle(dd) =>
        val TermName(str: String) = name
        val newRHS = super.transform(rhs)
        val named = q"chisel3.experimental.autoNameRecursively($str, $newRHS)"
        treeCopy.ValDef(dd, mods, name, tpt, localTyper typed named)
      // If a Data or a Memory, get the name and a prefix
      case dd @ ValDef(mods, name, tpt, rhs) if okVal(dd, tq"chisel3.Data", tq"chisel3.MemBase[_]") =>
        val TermName(str: String) = name
        val newRHS = super.transform(rhs)
        val prefixed = q"chisel3.experimental.prefix.apply[$tpt](name=$str)(f=$newRHS)"
        val named = q"chisel3.experimental.autoNameRecursively($str, $prefixed)"
        treeCopy.ValDef(dd, mods, name, tpt, localTyper typed named)
      // If an instance, just get a name but no prefix
      case dd @ ValDef(mods, name, tpt, rhs) if okVal(dd, tq"chisel3.experimental.BaseModule") =>
        val TermName(str: String) = name
        val newRHS = super.transform(rhs)
        val named = q"chisel3.experimental.autoNameRecursively($str, $newRHS)"
        treeCopy.ValDef(dd, mods, name, tpt, localTyper typed named)
      // Otherwise, continue
      case _ => super.transform(tree)
    }
  }
}
