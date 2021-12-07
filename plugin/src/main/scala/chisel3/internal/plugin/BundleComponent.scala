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

    val bundleTpe:         Type = inferType(tq"chisel3.Bundle")
    val dataTpe:           Type = inferType(tq"chisel3.Data")
    val seqOfDataTpe:      Type = inferType(tq"scala.collection.Seq[chisel3.Data]")
    val seqMapTpe:         Type = inferType(tq"scala.collection.immutable.SeqMap[String,$dataTpe]")
    val ignoreSeqInBundle: Type = inferType(tq"chisel3.IgnoreSeqInBundle")

    // Not cached because it should only be run once per class (thus once per Type)
    def isBundle(sym: Symbol): Boolean = {
      sym.tpe <:< bundleTpe
    }
    def isSeqOfData(sym: Symbol): Boolean = {
      sym.tpe <:< seqOfDataTpe
    }
    def isExactBundle(sym: Symbol): Boolean = {
      sym.tpe =:= bundleTpe
    }
    def isIgnoreSeqInBundle(sym: Symbol): Boolean = {
      sym.tpe <:< ignoreSeqInBundle
    }

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

    override def transform(tree: Tree): Tree = tree match {

      case bundle: ClassDef if isBundle(bundle.symbol) =>
        // We need to learn how to match on abstact bundles, line below fails to instantiate abstact
//      case bundle: ClassDef if isBundle(bundle.symbol) =>
        def show(string: => String): Unit = {
          if (bundle.symbol.name.toString.startsWith("Bpip")) {
            println(string)
          }
        }

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

        var additionalMethods: Seq[Tree] = Seq()

        if (!bundle.mods.hasFlag(Flag.ABSTRACT)) {
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

          additionalMethods ++= Seq(localTyper.typed(DefDef(cloneTypeSym, neww)))
        }

        // ==================== Generate val elements ====================
        // Create the symbol for the method and have it be associated with the Bundle class
        // Iff we are sure this is a Bundle construction we can handle.
        // Currently Bundles extending Bundles and traits are not supported
        // Just simple Bundles

        /* Test to see if the bundle found is amenable to having it's elements
         * converted to an immediate form that will not require reflection
         * First pass at this, we will only process Bundles with one parent
         */
        def isSupportedBundleType: Boolean = {
          val result =
            arguments.buildElementsAccessor && bundle.impl.parents.length < 100 &&
              !bundle.mods.hasFlag(Flag.ABSTRACT)
          show(
            s"buildElementsAccessor=${arguments.buildElementsAccessor} && " +
              s"parents=${bundle.impl.parents.length} result=$result"
          )
          result
        }

        def showInfo(info: Type): String = {
          info.members.mkString("\n")
        }

        if (isSupportedBundleType) {
          show(("#" * 80 + "\n") * 2)
          show(s"Processing: Bundle named: ${bundle.name.toString}")
          show(" ")
          show(s"BundleType: ${bundle.symbol.typeSignature.toString}")

          show("Bundle parents: \n" + bundle.impl.parents.map { parent =>
            s"parent: ${parent.symbol.name} [${isBundle(parent.symbol)}] " +
              s"IsBundle=" + isBundle(parent.symbol) + parent.symbol + "\n" +
//              "parent.symbol.info.decl: " + parent.symbol.info.decl(parent.symbol.info.decls.head.name). +
              s"\nDecls::\n  " + showInfo(parent.symbol.info)
          }.mkString("\n"))

          //TODO: Without the trims over the next dozen or so lines the variable names seem to have
          //      an extra space on the end.

          /* extracts the true fields of the Bundle
           */
          def getBundleElements(body: List[Tree]): List[(String, Tree)] = {
            val elements = mutable.ListBuffer[(String, Tree)]()
            body.foreach {
              case acc: ValDef if isBundle(acc.symbol) =>
                show(s"getBundleElements:Bundle rhs ${acc.rhs} ${showRaw(acc.rhs)}")
                elements += acc.symbol.name.toString.trim -> gen.mkAttributedSelect(thiz, acc.symbol)
              case acc: ValDef if isData(acc.symbol) && !isEmptyTree(acc.rhs) =>
                // empty tree test seems necessary to rule out generator methods passed into bundles
                // but there must be a better way here
                show(s"getBundleElements:ValDefNotBundle rhs ${acc.rhs} ${showRaw(acc.rhs)}")
                elements += acc.symbol.name.toString.trim -> gen.mkAttributedSelect(thiz, acc.symbol)
              case acc: ValDef =>
                def ik(t: Symbol): Unit = {
                  show(s"--->  gBE: rhs ${isSeqOfData(t)} : ${showRaw(t).take(60)}")
                }
//                show(s"gBE: acc ${showRaw(acc).take(60)}")
                //TODO: Remove the following ik lines, only the first one showed the Seq[Data]
//                ik(acc.symbol)
//                ik(acc.rhs.symbol)
//                ik(acc.rhs.tpe.typeSymbol)
//                ik(acc.tpe.typeSymbol)
//                ik(acc.tpe.termSymbol)
                if (isSeqOfData(acc.symbol) && !isIgnoreSeqInBundle(bundle.symbol)) {
                  show(s"--> gBE: ${bundle.symbol.parentSymbols.mkString(",")}")
                  show(s"--> gBE: isIgnore ${isIgnoreSeqInBundle(bundle.symbol)}")

                  global.reporter.error(acc.pos, s"Bundle: ${acc.symbol.name} has field which is a Seq[Data]")
                }
              case _ =>
            }
            elements.toList.reverse
          }

          /* extract the true fields from the super classes a given bundle
           */
          def getSuperClassBundleFields(bundleSymbol: Symbol, depth: Int = 0): List[(String, Tree)] = {
            val parents = bundleSymbol.parentSymbols

            show(s"getSuperClassBundle(${bundleSymbol.name.toString}): parents: " + parents.map(_.name.toString))

            def showType(tpe: Type): String = {
              val tf = BooleanFlag(Some(true))
              tpe match {
                case NullaryMethodType(TypeRef(ThisType(t1), t2, list)) =>
                  s"matched: ${t1}:$t2}" +
                    s"IsData: ${isData(TypeRef(ThisType(t1), t2, list).typeSymbol)}" +
                    showRaw(tpe, printTypes = tf, printKinds = tf, printPositions = tf)
                case _ =>
                  "Not matched" + showRaw(tpe, printTypes = tf, printKinds = tf, printPositions = tf)
              }
            }

            def isMethodReturningData(tpe: Type): Boolean = {
              tpe match {
                case NullaryMethodType(TypeRef(ThisType(t1), t2, list)) =>
                  isData(TypeRef(ThisType(t1), t2, list).typeSymbol)
                case _ =>
                  false
              }
            }

            def isBundleField(member: Symbol): Boolean = {
              member.isAccessor && isMethodReturningData(member.tpe)
            }

            parents.flatMap {
              case parent =>
                show(s"parent.info.decls:\n" + showInfo(parent.info))
                val currentFields = parent.info.members.flatMap {

                  case member if member.isPublic =>
                    show(
                      s"    Processing member ${member} isMethod=${member.isMethod} isAccessor==${member.isAccessor} tpe=${member.tpe}:${showType(member.tpe)} : kind: ${member.kindString} " +
                        s"${showRaw(member.typeSignature, printKinds = BooleanFlag(Some(true)), printIds = BooleanFlag(Some(true)))}"
                    )

                    if (isBundleField(member)) {
                      show(
                        s"     MATCHED: Trait Member ${member}  isData=${isData(member)} ${showRaw(member)} : ${showRaw(member.typeSignature)}"
                      )
                      Some(member.name.toString.trim -> gen.mkAttributedSelect(thiz, member))
                    } else if (isData(member)) {
                      show(
                        s"     Matched Bundle Member ${member} isData=${isData(member)} ${showRaw(member)} : ${showRaw(member.typeSignature)}"
                      )
                      show(s"     Found a field in $parent.${member}")
                      Some(member.name.toString.trim -> gen.mkAttributedSelect(thiz, member))
                      Some(member.name.toString.trim -> gen.mkAttributedSelect(thiz, member))
                    } else {
                      show(s"    method: ${member.name} was not a field")
                      None
                    }

                  case _ => None
                }
                val superFields = if (depth < 4 && !isExactBundle(parent)) {
                  getSuperClassBundleFields(parent, depth + 1)
                } else { List() }
                show(s"getSuper processing ${bundleSymbol.name.toString}.${parent}: \n" + showRaw(parent))
                superFields ++ currentFields
            }.reverse
          }

          val superFields = getSuperClassBundleFields(bundle.symbol)
          show(s"SuperFields:\n" + superFields.map(x => s"${x._1}: ${x._2}").mkString("\n"))

          val elementArgs: List[(String, Tree)] = getBundleElements(bundle.impl.body) ++ superFields

          val elementsImplSym =
            bundle.symbol.newMethod(TermName("_elementsImpl"), bundle.symbol.pos.focus, Flag.OVERRIDE | Flag.PROTECTED)
          // Handwritten cloneTypes don't have the Method flag set, unclear if it matters
          elementsImplSym.resetFlag(Flags.METHOD)
          // Need to set the type to chisel3.Bundle for the override to work
          elementsImplSym.setInfo(NullaryMethodType(seqMapTpe))

          val elementsImpl = localTyper.typed(
            DefDef(elementsImplSym, q"scala.collection.immutable.SeqMap.apply[String, chisel3.Data](..$elementArgs)")
          )

          show("ELEMENTS: \n" + elementArgs.map { case (symbol, tree) => s"(${symbol}, ${tree})" }.mkString("\n"))
          show("ElementsImpl: " + showRaw(elementsImpl) + "\n\n\n")
          show(s"Made: buildElementAccessor was built for ${bundle.symbol.name.toString}")
          additionalMethods ++= Seq(elementsImpl)

          show(("#" * 80 + "\n") * 2)
        }

        // ==================== Generate _usingPlugin ====================
        // Unclear why quasiquotes work here but didn't for cloneTypeSym, maybe they could.
        val usingPlugin = localTyper.typed(q"override protected def _usingPlugin: Boolean = true")
        additionalMethods ++= Seq(usingPlugin)

        val withMethods = deriveClassDef(bundle) { t =>
          deriveTemplate(t)(_ ++ additionalMethods)
        }

        super.transform(localTyper.typed(withMethods))

      case _ => super.transform(tree)
    }
  }
}
