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
      s.toString == name  || s.parents.exists { p =>
        p.toString == name  || typeHasTrait(p, name)
      }
    }

    def inferType(t: Tree): Type = localTyper.typed(t, nsc.Mode.TYPEmode).tpe

    def containerTpe(givenTpe: Type, internalTpe: Type): Boolean = {
      lazy val isType =
        givenTpe <:< internalTpe
      lazy val isIType =
        givenTpe.matchesPattern(inferType(tq"Iterable[$internalTpe]"))
      lazy val isOType =
        givenTpe.matchesPattern(inferType(tq"Option[$internalTpe]"))
      lazy val isIIType =
        givenTpe.matchesPattern(inferType(tq"Iterable[Iterable[_]]"))
      lazy val isIOType =
        givenTpe.matchesPattern(inferType(tq"Iterable[Option[_]]"))
      lazy val isOIType =
        givenTpe.matchesPattern(inferType(tq"Option[Iterable[_]]"))
      lazy val isOOType =
        givenTpe.matchesPattern(inferType(tq"Option[Option[_]]"))
      isType || isIType || isOType || isIIType || isIOType || isOIType || isOOType
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
    def okVal(dd: ValDef, tpe: Tree): Boolean = {

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
      okFlags(dd.mods) && containerTpe(inferType(dd.tpt), inferType(tpe)) && !isNull && dd.rhs != EmptyTree
    }

    // Method called by the compiler to modify source tree
    override def transform(tree: Tree): Tree = tree match {
      // If a Data, get name and prefix
      case dd @ ValDef(mods, name, tpt, rhs) if okVal(dd, tq"chisel3.Data") =>
        val TermName(str: String) = name
        val newRHS = super.transform(rhs)
        val prefixed = q"chisel3.experimental.prefix.apply[$tpt](name=$str)(f=$newRHS)"
        val named = q"chisel3.experimental.pluginNameRecursively($str, $prefixed)"
        treeCopy.ValDef(dd, mods, name, tpt, localTyper typed named)
      // If a HasId (includes modules/instances) just get name
      // TODO: Add test for doubled-nested modules getting prefixed
      // TODO: Why do modules elide the prefixing? Add test
      // TODO: Why does pluginName on Data specialize ports in current module?
      case dd @ ValDef(mods, name, tpt, rhs) if okVal(dd, tq"chisel3.experimental.BaseModule") =>
        val TermName(str: String) = name
        val newRHS = super.transform(rhs)
        val named = q"chisel3.experimental.pluginNameRecursively($str, $newRHS)"
        treeCopy.ValDef(dd, mods, name, tpt, localTyper typed named)
      case _ => super.transform(tree)
    }
  }
}
