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

private object ChiselComponent {
  sealed trait TypeOfInterest
  case object DataT extends TypeOfInterest
  case object MemT extends TypeOfInterest
  case object ModuleT extends TypeOfInterest
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

    import ChiselComponent._

    // Memoize type lookup results
    private val typeOfInterestCache = mutable.HashMap.empty[Type, Option[TypeOfInterest]]
    private val dataType = inferType(tq"chisel3.Data")
    private val memType = inferType(tq"chisel3.MemBase[_]")
    private val moduleType = inferType(tq"chisel3.experimental.BaseModule")

    // Determines if a ValDef has a type of interest somewhere in its type hierarchy
    // This includes as a type parameter of Iterable[_] or Option[_]
    private def hasSuperTypeOfInterest(p: ValDef): Option[TypeOfInterest] = {
      // If subtype of one of the base types, it's a match!
      def terminate(t: Type): Option[TypeOfInterest] =
        if (t <:< dataType) Some(DataT)
        else if (t <:< memType) Some(MemT)
        else if (t <:< moduleType) Some(ModuleT)
        else None

      // Recurse through subtype hierarchy finding containers
      // Seen is only updated when we recurse into type parameters, thus it is typically small
      def recShouldMatch(s: Type, seen: Set[Type]): Option[TypeOfInterest] = {
        def outerMatches(t: Type): Boolean = {
          val str = t.toString
          str.startsWith("Option[") || str.startsWith("Iterable[")
        }
        terminate(s).orElse {
          if (seen.contains(s)) {
            None
          } else if (outerMatches(s)) {
            // These are type parameters, loops *are* possible here
            recShouldMatch(s.typeArgs.head, seen + s)
          } else {
            // This is the standard inheritance hierarchy, Scalac catches loops here
            s.parents.view.map(p => recShouldMatch(p, seen)).collectFirst { case Some(a) => a }
          }
        }
      }

      // If doesn't match container pattern, exit early
      def earlyExit(t: Type): Boolean = {
        !(t.matchesPattern(inferType(tq"Iterable[_]")) || t.matchesPattern(inferType(tq"Option[_]")))
      }

      val q = inferType(p.tpt)

      typeOfInterestCache.getOrElseUpdate(q, {
        // First check if a match, then check early exit, then recurse
        terminate(q).orElse {
          if (earlyExit(q)) None
          else recShouldMatch(q, Set.empty)
        }
      })
    }

    // Given a type tree, infer the type and return it
    def inferType(t: Tree): Type = localTyper.typed(t, nsc.Mode.TYPEmode).tpe

    private val badFlags: Long =
      Flag.PARAM |
      Flag.SYNTHETIC |
      Flag.DEFERRED |
      Flags.TRIEDCOOKING |
      Flags.CASEACCESSOR |
      Flags.PARAMACCESSOR

    // Indicates whether a ValDef is properly formed to get name
    def okVal(dd: ValDef): Boolean = {

      // These were found through trial and error
      def okFlags(mods: Modifiers): Boolean = (mods.flags & badFlags) == 0L

      // Ensure expression isn't null, as you can't call `null.autoName("myname")`
      val isNull = dd.rhs match {
        case Literal(Constant(null)) => true
        case _ => false
      }

      okFlags(dd.mods) && !isNull && dd.rhs != EmptyTree
    }

    // Whether this val is directly enclosed by a Bundle type
    def inBundle(dd: ValDef): Boolean = {
      dd.symbol.logicallyEnclosingMember.thisType <:< inferType(tq"chisel3.Bundle")
    }

    // Method called by the compiler to modify source tree
    override def transform(tree: Tree): Tree = tree match {
      // Check if a subtree is a candidate
      case dd @ ValDef(mods, name, tpt, rhs) if okVal(dd) =>
        hasSuperTypeOfInterest(dd) match {
          // If a Data and in a Bundle, just get the name but not a prefix
          case Some(DataT) if inBundle(dd) =>
            val TermName(str: String) = name
            val newRHS = transform(rhs)
            val named = q"chisel3.internal.plugin.autoNameRecursively($str, $newRHS)"
            treeCopy.ValDef(dd, mods, name, tpt, localTyper typed named)
          // If a Data or a Memory, get the name and a prefix
          case Some(DataT) | Some(MemT) =>
            val TermName(str: String) = name
            val newRHS = transform(rhs)
            val prefixed = q"chisel3.experimental.prefix.apply[$tpt](name=$str)(f=$newRHS)"
            val named = q"chisel3.internal.plugin.autoNameRecursively($str, $prefixed)"
            treeCopy.ValDef(dd, mods, name, tpt, localTyper typed named)
          // If an instance, just get a name but no prefix
          case Some(ModuleT) =>
            val TermName(str: String) = name
            val newRHS = transform(rhs)
            val named = q"chisel3.internal.plugin.autoNameRecursively($str, $newRHS)"
            treeCopy.ValDef(dd, mods, name, tpt, localTyper typed named)
          case _ =>
            // Otherwise, continue
            super.transform(tree)
        }
      // Otherwise, continue
      case _ => super.transform(tree)
    }
  }
}
