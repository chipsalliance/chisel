// See LICENSE for license details.

package plugin

import scala.tools.nsc
import nsc.{Global, Phase}
import nsc.plugins.Plugin
import nsc.plugins.PluginComponent
import scala.reflect.internal.Flags
import scala.tools.nsc.transform.TypingTransformers

// The plugin to be run by the Scala compiler during compilation of Chisel code
class ChiselPlugin(val global: Global) extends Plugin {
  val name = "chiselplugin"
  val description = "chisel's plugin"
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
      unit.body = new MyTypingTransformer(unit).transform(unit.body)
    }
  }

  class MyTypingTransformer(unit: CompilationUnit)
    extends TypingTransformer(unit) {

    // Determines if a type has a given parent trait
    def typeHasTrait(s: Type, name: String): Boolean = {
      s.toString().toString == name  || s.parents.exists { p =>
        p.toString().toString == name  || typeHasTrait(p, name)
      }
    }

    def iterableTypeHasTrait(s: Type, name: String): Boolean = {
      def check(t: Type): Boolean = {
        val str = t.toString
        val isIter = (str.startsWith("Option[") || str.startsWith("Iterable[")) &&
          iterableTypeHasTrait(t.typeArgs.head, name)
        isIter || typeHasTrait(t, name)
      }
      check(s) || s.parents.exists { p =>
        //if(p.toString.toString.contains("Option")) {
        //  println(p.toString().toString)
        //  println(p.typeArgs.map(_.toString.toString))
        //}
        check(p) || iterableTypeHasTrait(p, name)
      }
    }

    // Utility function to help debug compiler plugin
    def serializeError(original: ValDef, modified: ValDef): Unit = {
      global.reporter.error(modified.pos, show(modified))
      writeAST("originalRaw", showRaw(original))
      write("original", show(original))
      writeAST("modifiedRaw", showRaw(modified))
      write("modified", show(modified))
    }

    // Indicates whether a ValDef is properly formed to get name
    def okVal(dd: ValDef, tpe: String): Boolean = {

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

      // Ensure expression isn't null, as you can't call `null.pluginName("myname")`
      val isNull = dd.rhs match {
        case Literal(Constant(null)) => true
        case _ => false
      }
      okFlags(dd.mods) && iterableTypeHasTrait(dd.tpt.tpe, tpe) && !isNull && dd.rhs != EmptyTree
    }

    // Method called by the compiler to modify source tree
    override def transform(tree: Tree): Tree = tree match {
      // If a Data, get name and prefix
      //case dd @ ValDef(mods, name, tpt, rhs) if name.toString.contains("CHECKME") =>
      //  write("iterable", showRaw(dd, true))
      //  write("res", iterableTypeHasTrait(tpt.tpe, "chisel3.Data").toString)
      //  dd
      case dd @ ValDef(mods, name, tpt, rhs) if okVal(dd, "chisel3.Data") =>
        val TermName(str: String) = name
        val newRHS = super.transform(rhs)
        val prefixed = q"chisel3.experimental.prefix.apply[$tpt](name=$str)(f=$newRHS)"
        val named = q"chisel3.experimental.pluginNameRecursively($str, $prefixed)"
        treeCopy.ValDef(dd, mods, name, tpt, localTyper typed named)
      // If a HasId (includes modules/instances) just get name
      case dd @ ValDef(mods, name, tpt, rhs) if okVal(dd, "chisel3.internal.HasId") =>
        val TermName(str: String) = name
        val newRHS = super.transform(rhs)
        val named = q"chisel3.experimental.pluginNameRecursively($str, $newRHS)"
        treeCopy.ValDef(dd, mods, name, tpt, localTyper typed named)
      case _ => super.transform(tree)
    }
  }
}
