// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import scala.collection.mutable
import scala.tools.nsc
import scala.tools.nsc.{Global, Phase}
import scala.tools.nsc.plugins.PluginComponent
import scala.tools.nsc.symtab.Flags
import scala.tools.nsc.transform.TypingTransformers

// TODO This component could also implement val elements in Bundles
private[plugin] class BundleComponent(val global: Global, arguments: ChiselPluginArguments)
    extends PluginComponent
    with TypingTransformers {
  import global._

  val phaseName: String = "chiselbundlephase"
  val runsAfter: List[String] = "typer" :: Nil
  def newPhase(prev: Phase): Phase = new BundlePhase(prev)

  private class BundlePhase(prev: Phase) extends StdPhase(prev) {
    override def name: String = phaseName
    def apply(unit: CompilationUnit): Unit = {
      // This plugin doesn't work on Scala 2.11 nor Scala 3. Rather than complicate the sbt build flow,
      // instead we just check the version and if its an early Scala version, the plugin does nothing
      val scalaVersion = scala.util.Properties.versionNumberString.split('.')
      val scalaVersionOk = scalaVersion(0).toInt == 2 && scalaVersion(1).toInt >= 12
      if (scalaVersionOk && arguments.useBundlePlugin) {
        unit.body = new MyTypingTransformer(unit).transform(unit.body)
      } else {
        val reason = if (!scalaVersionOk) {
          s"invalid Scala version '${scala.util.Properties.versionNumberString}'"
        } else {
          s"not enabled via '${arguments.useBundlePluginFullOpt}'"
        }
        // Enable this with scalacOption '-Ylog:chiselbundlephase'
        global.log(s"Skipping BundleComponent on account of $reason.")
      }
    }
  }

  private class MyTypingTransformer(unit: CompilationUnit) extends TypingTransformer(unit) {

    def inferType(t: Tree): Type = localTyper.typed(t, nsc.Mode.TYPEmode).tpe

    val bundleTpe = inferType(tq"chisel3.Bundle")
    val dataTpe = inferType(tq"chisel3.Data")

    // Not cached because it should only be run once per class (thus once per Type)
    def isBundle(sym: Symbol): Boolean = sym.tpe <:< bundleTpe

    val isDataCache = new mutable.HashMap[Type, Boolean]
    // Cached because this is run on every argument to every Bundle
    def isData(sym: Symbol): Boolean = isDataCache.getOrElseUpdate(sym.tpe, sym.tpe <:< dataTpe)

    def cloneTypeFull(tree: Tree): Tree =
      localTyper.typed(q"chisel3.experimental.DataMirror.internal.chiselTypeClone[${tree.tpe}]($tree)")

    def isNullaryMethodNamed(name: String, defdef: DefDef): Boolean =
      defdef.name.decodedName.toString == name && defdef.tparams.isEmpty && defdef.vparamss.isEmpty

    def getConstructorAndParams(body: List[Tree]): (Option[DefDef], Seq[Symbol]) = {
      val paramAccessors = mutable.ListBuffer[Symbol]()
      var primaryConstructor: Option[DefDef] = None
      body.foreach {
        case acc: ValDef if acc.symbol.isParamAccessor =>
          paramAccessors += acc.symbol
        case con: DefDef if con.symbol.isPrimaryConstructor =>
          primaryConstructor = Some(con)
        case d: DefDef if isNullaryMethodNamed("_cloneTypeImpl", d) =>
          val msg = "Users cannot override _cloneTypeImpl. Let the compiler plugin generate it. If you must, override cloneType instead."
          global.globalError(d.pos, msg)
        case d: DefDef if isNullaryMethodNamed("_usingPlugin", d) =>
          val msg = "Users cannot override _usingPlugin, it is for the compiler plugin's use only."
          global.globalError(d.pos, msg)
        case _ =>
      }
      (primaryConstructor, paramAccessors.toList)
    }


    override def transform(tree: Tree): Tree = tree match {

      case bundle: ClassDef if isBundle(bundle.symbol) && !bundle.mods.hasFlag(Flag.ABSTRACT) =>

        // ==================== Generate _cloneTypeImpl ====================
        val (con, params) = getConstructorAndParams(bundle.impl.body)
        if (con.isEmpty) {
          global.reporter.warning(bundle.pos, "Unable to determine primary constructor!")
          return super.transform(tree)
        }
        val constructor = con.get

        val thiz = gen.mkAttributedThis(bundle.symbol)

        // The params have spaces after them (Scalac implementation detail)
        val paramLookup: String => Symbol = params.map(sym => sym.name.toString.trim -> sym).toMap

        // Create a this.<ref> for each field matching order of constructor arguments
        // List of Lists because we can have multiple parameter lists
        val conArgs: List[List[Tree]] =
          constructor.vparamss.map(_.map { vp =>
            val p = paramLookup(vp.name.toString)
            // Make this.<ref>
            val select = gen.mkAttributedSelect(thiz, p)
            // Clone any Data parameters to avoid field aliasing, need full clone to include direction
            if (isData(vp.symbol)) cloneTypeFull(select) else select
          })

        val ttpe = Ident(bundle.symbol)
        val neww = localTyper.typed(New(ttpe, conArgs))

        // Create the symbol for the method and have it be associated with the Bundle class
        val cloneTypeSym =  bundle.symbol.newMethod(TermName("_cloneTypeImpl"), bundle.symbol.pos.focus, Flag.OVERRIDE | Flag.PROTECTED)
        // Handwritten cloneTypes don't have the Method flag set, unclear if it matters
        cloneTypeSym.resetFlag(Flags.METHOD)
        // Need to set the type to chisel3.Bundle for the override to work
        cloneTypeSym.setInfo(NullaryMethodType(bundleTpe))

        val cloneTypeImpl = localTyper.typed(DefDef(cloneTypeSym, neww))

        // ==================== Generate _usingPlugin ====================
        // Unclear why quasiquotes work here but didn't for cloneTypeSym, maybe they could.
        val usingPlugin = localTyper.typed(q"override protected def _usingPlugin: Boolean = true")

        val withMethods = deriveClassDef(bundle) { t =>
          deriveTemplate(t)(_ :+ cloneTypeImpl :+ usingPlugin)
        }

        super.transform(localTyper.typed(withMethods))

      case _ => super.transform(tree)
    }
  }
}
