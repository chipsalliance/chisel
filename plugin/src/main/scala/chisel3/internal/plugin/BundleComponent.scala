// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import scala.collection.mutable
import scala.tools.nsc
import scala.tools.nsc.{Global, Phase}
import scala.tools.nsc.plugins.PluginComponent
import scala.tools.nsc.symtab.Flags
import scala.tools.nsc.transform.TypingTransformers

/** Performs four operations
  * 1) Records that this plugin ran on a bundle by adding a method
  *    `override protected def _usingPlugin: Boolean = true`
  * 2) Constructs a cloneType method
  * 3) Builds a `def elements` that is computed once in this plugin
  *    Eliminates needing reflection to discover the hardware fields of a `Bundle`
  * 4) For Bundles which mix-in `HasAutoTypename`, builds a `def _typeNameConParams`
  *    statement to override the constructor parameter list, used to generate a typeName
  *
  * @param global     the environment
  * @param arguments  run time parameters to code
  */
private[plugin] class BundleComponent(val global: Global, arguments: ChiselPluginArguments)
    extends PluginComponent
    with TypingTransformers
    with ChiselOuterUtils {
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

  private class MyTypingTransformer(unit: CompilationUnit) extends TypingTransformer(unit) with ChiselInnerUtils {

    def cloneTypeFull(tree: Tree): Tree =
      localTyper.typed(q"chisel3.reflect.DataMirror.internal.chiselTypeClone[${tree.tpe}]($tree)")

    def isVarArgs(sym: Symbol): Boolean = definitions.isRepeatedParamType(sym.tpe)

    def getConstructorAndParams(body: List[Tree], isBundle: Boolean): (Option[DefDef], Seq[Symbol]) = {
      val paramAccessors = mutable.ListBuffer[Symbol]()
      var primaryConstructor: Option[DefDef] = None
      body.foreach {
        case acc: ValDef if acc.symbol.isParamAccessor =>
          paramAccessors += acc.symbol
        case con: DefDef if con.symbol.isPrimaryConstructor =>
          if (con.symbol.isPrivate) {
            val msg = "Private bundle constructors cannot automatically be cloned, try making it package private"
            global.reporter.error(con.pos, msg)
          } else {
            primaryConstructor = Some(con)
          }

        case d: DefDef if isNullaryMethodNamed("_cloneTypeImpl", d) =>
          val msg = "Users cannot override _cloneTypeImpl. Let the compiler plugin generate it."
          global.reporter.error(d.pos, msg)
        case d: DefDef if isNullaryMethodNamed("_elementsImpl", d) && isBundle =>
          val msg = "Users cannot override _elementsImpl. Let the compiler plugin generate it."
          global.reporter.error(d.pos, msg)
        case d: DefDef if isNullaryMethodNamed("_usingPlugin", d) && isBundle =>
          val msg = "Users cannot override _usingPlugin, it is for the compiler plugin's use only."
          global.reporter.error(d.pos, msg)
        case d: DefDef if isNullaryMethodNamed("cloneType", d) =>
          val prefix = if (isBundle) "Bundles" else "Records"
          val msg = s"$prefix cannot override cloneType. Let the compiler plugin generate it."
          global.reporter.error(d.pos, msg)
        case d: DefDef if isNullaryMethodNamed("_typeNameConParams", d) =>
          val msg = "Users cannot override _typeNameConParams. Let the compiler plugin generate it."
          global.reporter.error(d.pos, msg)
        case _ =>
      }
      (primaryConstructor, paramAccessors.toList)
    }

    def generateAutoCloneType(
      record:     ClassDef,
      thiz:       global.This,
      conArgsOpt: Option[List[List[Tree]]],
      isBundle:   Boolean
    ): Option[Tree] = {
      conArgsOpt.map { conArgs =>
        val tparamList = record.tparams.map { t => Ident(t.symbol) }
        val ttpe =
          if (tparamList.nonEmpty) AppliedTypeTree(Ident(record.symbol), tparamList) else Ident(record.symbol)
        val newUntyped = New(ttpe, conArgs)

        // TODO For private default constructors this crashes with a
        // TypeError. Figure out how to make this local to the object so
        // that private default constructors work.
        val neww = localTyper.typed(newUntyped)

        // Create the symbol for the method and have it be associated with the Record class
        val cloneTypeSym =
          record.symbol.newMethod(TermName("_cloneTypeImpl"), record.symbol.pos.focus, Flag.OVERRIDE | Flag.PROTECTED)
        // Handwritten cloneTypes don't have the Method flag set, unclear if it matters
        cloneTypeSym.resetFlag(Flags.METHOD)

        cloneTypeSym.setInfo(NullaryMethodType(recordTpe))

        localTyper.typed(DefDef(cloneTypeSym, neww))
      }
    }

    def generateElements(bundle: ClassDef, thiz: global.This): Tree = {
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

      elementsImpl
    }

    def generateAutoTypename(record: ClassDef, thiz: global.This, conArgsOpt: Option[List[Tree]]): Option[Tree] = {
      conArgsOpt.flatMap { conArgs =>
        val recordIsAnon = record.symbol.isAnonOrRefinementClass

        // Anonymous `Records` in any scope are forbidden
        if (record.symbol.isAnonOrRefinementClass) {
          global.reporter.error(
            record.pos,
            "Users cannot mix 'HasAutoTypename' into an anonymous Record. Create a named class."
          )
          None
        } else {
          // Create an iterable out of all constructor argument accessors
          val typeNameConParamsSym =
            record.symbol.newMethod(
              TermName("_typeNameConParams"),
              record.symbol.pos.focus,
              Flag.OVERRIDE | Flag.PROTECTED
            )
          typeNameConParamsSym.resetFlag(Flags.METHOD)
          typeNameConParamsSym.setInfo(NullaryMethodType(itAnyTpe))

          Some(
            localTyper.typed(
              DefDef(typeNameConParamsSym, q"scala.collection.immutable.Vector.apply[Any](..${conArgs})")
            )
          )
        }
      }
    }

    // Creates a list of constructor parameter accessors from the argument value. Returns a Some if a constructor
    // is found, or None otherwise.
    private def extractConArgs(record: ClassDef, thiz: global.This, isBundle: Boolean): Option[List[List[Tree]]] = {
      val (con, params) = getConstructorAndParams(record.impl.body, isBundle)
      if (con.isEmpty) {
        global.reporter.warning(record.pos, "Unable to determine primary constructor!")
        return None
      }

      val constructor = con.get

      // The params have spaces after them (Scalac implementation detail)
      val paramLookup: String => Symbol = params.map(sym => sym.name.toString.trim -> sym).toMap

      // Create a this.<ref> for each field matching order of constructor arguments
      // List of Lists because we can have multiple parameter lists
      Some(constructor.vparamss.map(_.map { vp =>
        val p = paramLookup(vp.name.toString)
        // Make this.<ref>
        val select = gen.mkAttributedSelect(thiz.asInstanceOf[Tree], p)
        // Clone any Data parameters to avoid field aliasing, need full clone to include direction
        val cloned = if (isData(vp.symbol)) cloneTypeFull(select.asInstanceOf[Tree]) else select
        // Need to splat varargs
        if (isVarArgs(vp.symbol)) q"$cloned: _*" else cloned
      }))
    }

    override def transform(tree: Tree): Tree = tree match {

      case record: ClassDef
          if isARecord(record.symbol) && !record.mods.hasFlag(Flag.ABSTRACT) => // check that its not abstract
        val isBundle: Boolean = isABundle(record.symbol)
        val thiz:     global.This = gen.mkAttributedThis(record.symbol)
        val conArgs:  Option[List[List[Tree]]] = extractConArgs(record, thiz, isBundle)

        // ==================== Generate _cloneTypeImpl ====================
        val cloneTypeImplOpt = generateAutoCloneType(record, thiz, conArgs, isBundle)

        // ==================== Generate val elements (Bundles only) ====================
        val elementsImplOpt = if (isBundle) Some(generateElements(record, thiz)) else None

        // ==================== Generate _usingPlugin ====================
        val usingPluginOpt = if (isBundle) {
          // Unclear why quasiquotes work here but didn't for cloneTypeSym, maybe they could.
          Some(localTyper.typed(q"override protected def _usingPlugin: Boolean = true"))
        } else {
          None
        }

        val autoTypenameOpt =
          if (isAutoTypenamed(record.symbol)) generateAutoTypename(record, thiz, conArgs.map(_.flatten))
          else None

        val withMethods = deriveClassDef(record) { t =>
          deriveTemplate(t)(_ ++ cloneTypeImplOpt ++ usingPluginOpt ++ elementsImplOpt ++ autoTypenameOpt)
        }

        super.transform(localTyper.typed(withMethods))

      case _ => super.transform(tree)
    }
  }
}
