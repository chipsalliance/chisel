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
  * @param global     the environment
  * @param arguments  run time parameters to code
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
      if (ChiselPlugin.runComponent(global, arguments)(unit)) {
        unit.body = new MyTypingTransformer(unit).transform(unit.body)
      }
    }
  }

  private class MyTypingTransformer(unit: CompilationUnit) extends TypingTransformer(unit) {

    def inferType(t: Tree): Type = localTyper.typed(t, nsc.Mode.TYPEmode).tpe

    val bundleTpe:      Type = inferType(tq"chisel3.Bundle")
    val dataTpe:        Type = inferType(tq"chisel3.Data")
    val ignoreSeqTpe:   Type = inferType(tq"chisel3.IgnoreSeqInBundle")
    val seqOfDataTpe:   Type = inferType(tq"scala.collection.Seq[chisel3.Data]")
    val someOfDataTpe:  Type = inferType(tq"scala.Option[chisel3.Data]")
    val itStringAnyTpe: Type = inferType(tq"scala.collection.Iterable[(String,Any)]")

    // Not cached because it should only be run once per class (thus once per Type)
    def isBundle(sym: Symbol): Boolean = { sym.tpe <:< bundleTpe }

    def isIgnoreSeqInBundle(sym: Symbol): Boolean = { sym.tpe <:< ignoreSeqTpe }

    def isSeqOfData(sym: Symbol): Boolean = {
      val tpe = sym.tpe
      tpe match {
        case NullaryMethodType(resultType) =>
          resultType <:< seqOfDataTpe
        case _ =>
          false
      }
    }

    def isOptionOfData(symbol: Symbol): Boolean = {
      val tpe = symbol.tpe
      tpe match {
        case NullaryMethodType(resultType) =>
          resultType <:< someOfDataTpe
        case _ =>
          false
      }
    }
    def isExactBundle(sym: Symbol): Boolean = { sym.tpe =:= bundleTpe }

    // Cached because this is run on every argument to every Bundle
    val isDataCache = new mutable.HashMap[Type, Boolean]
    def isData(sym: Symbol): Boolean = isDataCache.getOrElseUpdate(sym.tpe, sym.tpe <:< dataTpe)

    def cloneTypeFull(tree: Tree): Tree =
      localTyper.typed(q"chisel3.experimental.DataMirror.internal.chiselTypeClone[${tree.tpe}]($tree)")

    def isNullaryMethodNamed(name: String, defdef: DefDef): Boolean =
      defdef.name.decodedName.toString == name && defdef.tparams.isEmpty && defdef.vparamss.isEmpty

    def isVarArgs(sym: Symbol): Boolean = definitions.isRepeatedParamType(sym.tpe)

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
        case d: DefDef if isNullaryMethodNamed("_elementsImpl", d) =>
          val msg = "Users cannot override _elementsImpl. Let the compiler plugin generate it."
          global.globalError(d.pos, msg)
        case d: DefDef if isNullaryMethodNamed("_usingPlugin", d) =>
          val msg = "Users cannot override _usingPlugin, it is for the compiler plugin's use only."
          global.globalError(d.pos, msg)
        case d: DefDef if isNullaryMethodNamed("cloneType", d) =>
          val msg = "Users cannot override cloneType.  Let the compiler plugin generate it."
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

        val cloneTypeImplOpt = if (!bundle.mods.hasFlag(Flag.ABSTRACT)) {
          // Create a this.<ref> for each field matching order of constructor arguments
          // List of Lists because we can have multiple parameter lists
          val conArgs: List[List[Tree]] =
            constructor.vparamss.map(_.map { vp =>
              val p = paramLookup(vp.name.toString)
              // Make this.<ref>
              val select = gen.mkAttributedSelect(thiz.asInstanceOf[Tree], p)
              // Clone any Data parameters to avoid field aliasing, need full clone to include direction
              val cloned = if (isData(vp.symbol)) cloneTypeFull(select.asInstanceOf[Tree]) else select
              // Need to splat varargs
              if (isVarArgs(vp.symbol)) q"$cloned: _*" else cloned
            })

          val tparamList = bundle.tparams.map { t => Ident(t.symbol) }
          val ttpe =
            if (tparamList.nonEmpty) AppliedTypeTree(Ident(bundle.symbol), tparamList) else Ident(bundle.symbol)
          val newUntyped = New(ttpe, conArgs)
          val neww = localTyper.typed(newUntyped)

          // Create the symbol for the method and have it be associated with the Bundle class
          val cloneTypeSym =
            bundle.symbol.newMethod(TermName("_cloneTypeImpl"), bundle.symbol.pos.focus, Flag.OVERRIDE | Flag.PROTECTED)
          // Handwritten cloneTypes don't have the Method flag set, unclear if it matters
          cloneTypeSym.resetFlag(Flags.METHOD)
          // Need to set the type to chisel3.Bundle for the override to work
          cloneTypeSym.setInfo(NullaryMethodType(bundleTpe))

          Some(localTyper.typed(DefDef(cloneTypeSym, neww)))
        } else {
          // Don't create if this Bundle is abstract
          None
        }

        // ==================== Generate val elements ====================

        /* Test to see if the bundle found is amenable to having it's elements
         * converted to an immediate form that will not require reflection
         */
        def isSupportedBundleType: Boolean = !bundle.mods.hasFlag(Flag.ABSTRACT)

        val elementsImplOpt = if (isSupportedBundleType) {
          /* extract the true fields from the super classes a given bundle
           * depth argument can be helpful for debugging
           */
          def getAllBundleFields(bundleSymbol: Symbol, depth: Int = 0): List[(String, Tree)] = {

            def isBundleField(member: Symbol): Boolean = {
              if (!member.isAccessor) {
                false
              } else if (isData(member.tpe.typeSymbol)) {
                true
              } else if (isOptionOfData(member)) {
                true
              } else if (isSeqOfData(member)) {
                // This field is passed along, even though it is illegal
                // An error for this will be generated in `Bundle.elements`
                // It would be possible here to check for Seq[Data] and make a compiler error, but
                // that would be a API error difference. See reference in docs/chisel-plugin.md
                // If Bundle is subclass of IgnoreSeqInBundle then don't pass this field along

                !isIgnoreSeqInBundle(bundleSymbol)
              } else {
                // none of the above
                false
              }
            }

            val currentFields = bundleSymbol.info.members.flatMap {

              case member if member.isPublic =>
                if (isBundleField(member)) {
                  // The params have spaces after them (Scalac implementation detail)
                  Some(member.name.toString.trim -> gen.mkAttributedSelect(thiz.asInstanceOf[Tree], member))
                } else {
                  None
                }

              case _ => None
            }.toList

            val allParentFields = bundleSymbol.parentSymbols.flatMap { parentSymbol =>
              val fieldsFromParent = if (depth < 1 && !isExactBundle(bundleSymbol)) {
                val foundFields = getAllBundleFields(parentSymbol, depth + 1)
                foundFields
              } else {
                List()
              }
              fieldsFromParent
            }
            allParentFields ++ currentFields
          }

          val elementArgs = getAllBundleFields(bundle.symbol)

          val elementsImplSym =
            bundle.symbol.newMethod(TermName("_elementsImpl"), bundle.symbol.pos.focus, Flag.OVERRIDE | Flag.PROTECTED)
          elementsImplSym.resetFlag(Flags.METHOD)
          elementsImplSym.setInfo(NullaryMethodType(itStringAnyTpe))

          val elementsImpl = localTyper.typed(
            DefDef(elementsImplSym, q"scala.collection.immutable.Vector.apply[(String, Any)](..$elementArgs)")
          )

          Some(elementsImpl)
        } else {
          // No code generated for elements accessor
          None
        }

        // ==================== Generate _usingPlugin ====================
        // Unclear why quasiquotes work here but didn't for cloneTypeSym, maybe they could.
        val usingPluginOpt = Some(localTyper.typed(q"override protected def _usingPlugin: Boolean = true"))

        val withMethods = deriveClassDef(bundle) { t =>
          deriveTemplate(t)(_ ++ cloneTypeImplOpt ++ usingPluginOpt ++ elementsImplOpt)
        }

        super.transform(localTyper.typed(withMethods))

      case _ => super.transform(tree)
    }
  }
}
