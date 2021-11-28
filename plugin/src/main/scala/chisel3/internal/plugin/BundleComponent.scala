// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import scala.collection.mutable
import scala.tools.nsc
import scala.tools.nsc.{Global, Phase}
import scala.tools.nsc.plugins.PluginComponent
import scala.tools.nsc.symtab.Flags
import scala.tools.nsc.transform.TypingTransformers

/** Performs three operations
  * 1) Records that this plugin ran on a bundle by adding a method
  *    `override protected def _usingPlugin: Boolean = true`
  * 2) Constructs a cloneType method
  * 3) Builds a `def elements` that is computed once in this plugin
  *    Eliminates needing reflection to discover the hardware fields of a `Bundle`
  *
  * @param global
  * @param arguments
  */
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
    val seqMapTpe = inferType(tq"scala.collection.immutable.SeqMap[String,$dataTpe]")

    // Not cached because it should only be run once per class (thus once per Type)
    def isBundle(sym: Symbol): Boolean = sym.tpe <:< bundleTpe

    val isDataCache = new mutable.HashMap[Type, Boolean]
    // Cached because this is run on every argument to every Bundle
    def isData(sym: Symbol): Boolean = isDataCache.getOrElseUpdate(sym.tpe, sym.tpe <:< dataTpe)

    def isEmptyTree(tree: Tree): Boolean = tree == EmptyTree

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
          val msg = "Users cannot override _cloneTypeImpl. Let the compiler plugin generate it."
          global.globalError(d.pos, msg)
        case d: DefDef if isNullaryMethodNamed("_usingPlugin", d) =>
          val msg = "Users cannot override _usingPlugin, it is for the compiler plugin's use only."
          global.globalError(d.pos, msg)
        case d: DefDef if isNullaryMethodNamed("cloneType", d) =>
          val msg = "Users cannot override cloneType. Let the compiler plugin generate it."
          global.globalError(d.pos, msg)
        case _ =>
      }
      (primaryConstructor, paramAccessors.toList)
    }

    def getBundleElements(body: List[Tree]): List[(Symbol, Tree)] = {
      val elements = mutable.ListBuffer[(Symbol, Tree)]()
      body.foreach {
        case acc: ValDef if isBundle(acc.symbol) =>
          elements += acc.symbol -> acc.rhs
        case acc: ValDef if isData(acc.symbol) && ! isEmptyTree(acc.rhs) =>
          // empty tree test seems necessary to rule out generator methods passed into bundles
          // but there must be a better way here
          elements += acc.symbol -> acc.rhs
        case _ =>
      }
      elements.toList
    }

    override def transform(tree: Tree): Tree = tree match {

      case bundle: ClassDef if isBundle(bundle.symbol) && !bundle.mods.hasFlag(Flag.ABSTRACT) =>
        def show(string: String): Unit = {
          if (bundle.symbol.name.toString == "DemoBundle" || bundle.symbol.name.toString == "AnimalBundle") {
            println(("=" * 100 + "\n") * 1)
            println(string)
          }
        }

        show(s"Bundle: ${show(bundle.toString)}")
        show(s"BundleType: ${show(bundle.tpe.toString)}")

        show("Demo")
        show(s"BundleName: '${bundle.symbol.name}'")

        // ==================== Generate _cloneTypeImpl ====================
        val (con, params) = getConstructorAndParams(bundle.impl.body)
        if (con.isEmpty) {
          global.reporter.warning(bundle.pos, "Unable to determine primary constructor!")
          return super.transform(tree)
        }

        val constructor = con.get
        show(s"Constructor: ${constructor.toString()}")
        val thiz = gen.mkAttributedThis(bundle.symbol)

//          show(s"thiz: ${thiz.toString()}")
        show(s"thiz: ${thiz.tpe.toString()}")
//          show(s"thiz: ${thiz.tpe.members.mkString("\n")}")

        // The params have spaces after them (Scalac implementation detail)
        val paramLookup: String => Symbol = params.map(sym => sym.name.toString.trim -> sym).toMap

        show(paramLookup.toString())

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

        val tparamList = bundle.tparams.map { t => Ident(t.symbol) }
        val ttpe = if (tparamList.nonEmpty) AppliedTypeTree(Ident(bundle.symbol), tparamList) else Ident(bundle.symbol)
        val newUntyped = New(ttpe, conArgs)
        val neww = localTyper.typed(newUntyped)

        // Create the symbol for the method and have it be associated with the Bundle class
        val cloneTypeSym =  bundle.symbol.newMethod(TermName("_cloneTypeImpl"), bundle.symbol.pos.focus, Flag.OVERRIDE | Flag.PROTECTED)
        // Handwritten cloneTypes don't have the Method flag set, unclear if it matters
        cloneTypeSym.resetFlag(Flags.METHOD)
        // Need to set the type to chisel3.Bundle for the override to work
        cloneTypeSym.setInfo(NullaryMethodType(bundleTpe))

        val cloneTypeImpl = localTyper.typed(DefDef(cloneTypeSym, neww))

        // ==================== Generate val elements ====================
        // Create the symbol for the method and have it be associated with the Bundle class

        val elements = getBundleElements(bundle.impl.body).reverse
        val elementArgs: List[(String, Tree)] = elements.map { case (symbol, chiselType) =>
           // Make this.<ref>
          val select = gen.mkAttributedSelect(thiz, symbol)
          (symbol.name.toString, select)
        }

        val elementsImplSym = bundle.symbol.newMethod(TermName("_elementsImpl"), bundle.symbol.pos.focus, Flag.OVERRIDE | Flag.PROTECTED)
        // Handwritten cloneTypes don't have the Method flag set, unclear if it matters
        elementsImplSym.resetFlag(Flags.METHOD)
        // Need to set the type to chisel3.Bundle for the override to work
        elementsImplSym.setInfo(NullaryMethodType(seqMapTpe))

        val elementsImpl = localTyper.typed(DefDef(elementsImplSym, q"scala.collection.immutable.SeqMap.apply[String, chisel3.Data](..$elementArgs)"))

        show("ELEMENTS:\n" + elements.map { case (symbol, tree) => s"(${symbol}, ${tree})" }.mkString("\n"))
        show("ElementsImpl: " + showRaw(elementsImpl) + "\n\n\n")

        // ==================== Generate _usingPlugin ====================
        // Unclear why quasiquotes work here but didn't for cloneTypeSym, maybe they could.
        val usingPlugin = localTyper.typed(q"override protected def _usingPlugin: Boolean = true")

        val withMethods = deriveClassDef(bundle) { t =>
          deriveTemplate(t)(_ :+ cloneTypeImpl :+ usingPlugin :+ elementsImpl)
//            deriveTemplate(t)(_ :+ cloneTypeImpl :+ usingPlugin)
        }

        super.transform(localTyper.typed(withMethods))

      case _ => super.transform(tree)
    }
  }
}
