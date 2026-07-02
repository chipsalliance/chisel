package unroll

import scala.tools.nsc.symtab.Flags
import scala.tools.nsc.transform.{Transform, TypingTransformers}
import scala.tools.nsc.{Global, Phase}
import tools.nsc.plugins.PluginComponent

class UnrollPhaseScala2(val global: Global) extends PluginComponent with TypingTransformers with Transform {

  import global._

  val runsAfter = List("pickler")

  override val runsBefore = List("refchecks")

  val phaseName = "unroll"

  override def newTransformer(unit: global.CompilationUnit): global.Transformer = {
    new UnrollTransformer(unit)
  }

  private def isValidUnrolledMethod(method: Symbol, origin: Position) = {
    val isCtor = method.isConstructor

    def explanation = {
      def what = if (isCtor) s"a class constructor" else s"method ${method.name}"
      val prefix = s"Cannot unroll parameters of $what"
      if (isLocalMethod(method))
        s"$prefix because it is a local method"
      else if (!method.isEffectivelyFinal && !method.owner.isEffectivelyFinal)
        s"$prefix because it can be overridden"
      else if (method.owner.companionClass.isCaseClass)
        s"$prefix of a case class companion object: please annotate the class constructor instead"
      else
        s"$prefix of a case class: please annotate the class constructor instead"
    }

    if (isCtor) true
    else if (isLocalMethod(method)
      || !method.isEffectivelyFinal && !method.owner.isEffectivelyFinal
      || method.owner.companionClass.isCaseClass && method.name == nme.apply
      || method.owner.isCaseClass && method.name == nme.copy) {
        globalError(origin, explanation)
        false
      } else true
  }

  def findUnrollAnnotations(params: Seq[Symbol]): Seq[Int] = {
    params.toList.zipWithIndex.collect {
      case (v, i) if hasUnrollAnnotation(v) && isValidUnrolledMethod(v.owner, v.pos) => i
    }
  }

  private def hasUnrollAnnotation(symbol: Symbol) =
    symbol.annotations.exists(_.tpe =:= typeOf[com.lihaoyi.unroll])

  private def methodHasUnroll(method: Symbol) =
    method.paramss.exists(_.exists(hasUnrollAnnotation))

  def copyValDef(vd: ValDef) = {
    val newMods = vd.mods.copy(flags = vd.mods.flags ^ Flags.DEFAULTPARAM)
    newStrictTreeCopier.ValDef(vd, newMods, vd.name, vd.tpt, EmptyTree)
      .setSymbol(vd.symbol)
  }

  def copySymbol(owner: Symbol, s: Symbol) = {
    val newSymbol = owner.newValueParameter(s.name.toTermName)
    newSymbol.setInfo(s.tpe)
    newSymbol
  }

  implicit class Setter[T <: Tree](t: T){
    def set(s: Symbol) = t.setType(s.tpe).setSymbol(s)
  }
  def generateSingleForwarder(implDef: ImplDef,
                              defdef: DefDef,
                              paramIndex: Int,
                              nextParamIndex: Int,
                              nextSymbol: Symbol,
                              annotatedParamListIndex: Int,
                              paramLists: List[List[ValDef]]) = {

    val forwarderDefSymbol = defdef.symbol.owner.newMethod(defdef.name)
    val symbolReplacements = defdef
      .vparamss
      .flatten
      .map { p => p.symbol -> copySymbol(forwarderDefSymbol, p.symbol) }
      .toMap ++ {
      defdef.symbol.tpe match{
        case MethodType(originalParams, result) => Nil
        case PolyType(tparams, MethodType(originalParams, result)) =>
          // Not sure why this is necessary, but the `originalParams` here
          // is different from `defdef.vparamss`, and we need both
          originalParams.map(p => (p, copySymbol(forwarderDefSymbol, p)))
      }
    }

    def forwarderMethodType0(t: Type, n: Int): Type = t match{
      case MethodType(originalParams, result) =>
        val forwarderParams = originalParams.map(symbolReplacements)
        if (n == annotatedParamListIndex) MethodType(forwarderParams.take(paramIndex), result)
        else MethodType(forwarderParams, forwarderMethodType0(result, n + 1))

      case PolyType(tparams, res) => PolyType(tparams, forwarderMethodType0(res, n))
    }

    val forwarderMethodType = forwarderMethodType0(defdef.symbol.tpe, 0)

    forwarderDefSymbol.setInfo(forwarderMethodType)

    val newParamLists = paramLists
      .zipWithIndex
      .map{ case (paramList, i) =>
        if (i != annotatedParamListIndex) paramList
        else paramList.take(paramIndex)
      }
      .map(_.map(copyValDef))


    val defaultOffset = paramLists
      .iterator
      .take(annotatedParamListIndex)
      .map(_.size)
      .sum

    val forwardedValueParams = newParamLists(annotatedParamListIndex).map(p => Ident(p.name).set(p.symbol))

    val nestedForwarderMethodTypes = Seq
      .iterate(nextSymbol.tpe, defdef.vparamss.length + 1){
        case MethodType(args, res) => res
        case PolyType(tparams, MethodType(args, res)) => res
      }
      .drop(1)

    val defaultCalls = Range(paramIndex, nextParamIndex).map{n =>
      val mangledName = defdef.name.toString + "$default$" + (defaultOffset + n + 1)

      val defaultOwner =
        if (defdef.symbol.isConstructor) implDef.symbol.companionModule
        else implDef.symbol

      val defaultMember = defaultOwner.tpe.member(TermName(scala.reflect.NameTransformer.encode(mangledName)))
      newParamLists.take(annotatedParamListIndex).map(_.map( p => Ident(p.name).set(p.symbol)))
        .zip(nestedForwarderMethodTypes)
        .foldLeft(Ident(mangledName).setSymbol(defaultMember).set(defaultMember).set(defaultMember): Tree) {
          case (lhs, (ps, methodType)) => Apply(fun = lhs, args = ps).setType(methodType)
        }

    }

    val forwarderCallArgs = newParamLists.zipWithIndex.map{case (v, i) =>
      if (i == annotatedParamListIndex) forwardedValueParams.take(nextParamIndex) ++ defaultCalls
      else v.map( p => Ident(p.name).set(p.symbol))
    }

    val forwarderThis = This(defdef.symbol.owner).set(defdef.symbol.owner)

    val forwarderInner =
      if (defdef.symbol.isConstructor) Super(forwarderThis, typeNames.EMPTY).set(defdef.symbol.owner)
      else forwarderThis

    val forwarderCall0 = forwarderCallArgs
      .zip(nestedForwarderMethodTypes)
      .foldLeft(Select(forwarderInner, defdef.name).set(nextSymbol): Tree){
        case (lhs, (ps, methodType)) => Apply(fun = lhs, args = ps).setType(methodType)
      }

    val forwarderCall =
      if (!defdef.symbol.isConstructor) forwarderCall0
      else Block(List(forwarderCall0), Literal(Constant(())).setType(typeOf[Unit]))

    val forwarderDef = treeCopy.DefDef(
      defdef,
      mods = defdef.mods,
      name = defdef.name,
      tparams = defdef.tparams,
      vparamss = newParamLists,
      tpt = defdef.tpt,
      rhs = if (nextParamIndex == -1) EmptyTree else forwarderCall
    ).set(forwarderDefSymbol)

    // No idea why this `if` guard is necessary, but without it we end up missing
    // some generated forwarders on trait methods inherited by static objects
    if (defdef.symbol.owner.isTrait && !defdef.symbol.isAbstract) {
      defdef.symbol.owner.info.asInstanceOf[ClassInfoType].decls.enter(forwarderDefSymbol)
    }

    val (fromSyms, toSyms) = symbolReplacements.toList.unzip
    forwarderDef.substituteSymbols(fromSyms, toSyms).asInstanceOf[DefDef]
  }

  private object LintInnerMethods extends Traverser {
    override def traverse(tree: Tree): Unit = {
      tree match {
        case method: DefDef =>
          if (methodHasUnroll(method.symbol)) isValidUnrolledMethod(method.symbol, method.pos)
          traverseChildren(method.rhs)
        case _ => ()
      }
    }

    final def traverseChildren(tree: Tree): Unit = super.traverse(tree)
  }

  private def isLocalMethod(sym: Symbol): Boolean = {
    sym.ownerChain.tail.exists(_.isMethod) || sym.isLocalToBlock
  }

  class UnrollTransformer(unit: global.CompilationUnit) extends TypingTransformer(unit) {
    def generateDefForwarders(implDef: ImplDef): List[(Option[Symbol], Seq[DefDef])] = {
      implDef.impl.body.collect{ case defdef: DefDef =>
        LintInnerMethods.traverseChildren(defdef.rhs)

        val annotatedOpt =
          if (defdef.symbol.isCaseCopy && defdef.symbol.name.toString == "copy") {
            Some(defdef.symbol.owner.primaryConstructor)
          } else if (defdef.symbol.isCaseApplyOrUnapply && defdef.symbol.name.toString == "apply"){
            val classConstructor = defdef.symbol.owner.companionClass.primaryConstructor
            if (classConstructor == NoSymbol) None
            else Some(classConstructor)
          } else {
            Some(defdef.symbol)
          }

        // Somehow the "apply" methods of case class companions defined within local scopes
        // do not have companion class primary constructor symbols, so we just skip them here
        try {
          if (annotatedOpt.isEmpty) (None, Nil)
          else {
            annotatedOpt.get.tpe.paramss
              .zipWithIndex
              .flatMap{case (annotatedParamList, paramListIndex) =>
                val annotationIndices = findUnrollAnnotations(annotatedParamList)
                if (annotationIndices.isEmpty) None
                else Some((annotatedParamList, annotationIndices, paramListIndex))
              } match{
              case Nil => (None, Nil)
              case Seq((annotatedParamList, annotationIndices, paramListIndex)) =>
                if (defdef.symbol.isAbstract) {
                  (Some(defdef.symbol),
                  (Seq(-1) ++ annotationIndices ++ Seq(annotatedParamList.length)).sliding(2).toList.foldLeft((Seq.empty[DefDef], defdef.symbol)) {
                    case ((defdefs, nextSymbol), Seq(paramIndex, nextParamIndex)) =>
                      val forwarderDef = generateSingleForwarder(
                        implDef,
                        defdef,
                        nextParamIndex,
                        paramIndex,
                        nextSymbol,
                        paramListIndex,
                        defdef.vparamss
                      )
                      (forwarderDef +: defdefs, forwarderDef.symbol)
                  }._1)
                }
                else {
                  (None,
                  (annotationIndices :+ annotatedParamList.length).sliding(2).toList.reverse.foldLeft((Seq.empty[DefDef], defdef.symbol)){
                    case ((defdefs, nextSymbol), Seq(paramIndex, nextParamIndex)) =>
                      val forwarderDef =  generateSingleForwarder(
                        implDef,
                        defdef,
                        paramIndex,
                        nextParamIndex,
                        nextSymbol,
                        paramListIndex,
                        defdef.vparamss
                      )
                      (forwarderDef +: defdefs, forwarderDef.symbol)
                  }._1)
                }

              case multiple => sys.error("Multiple")
            }
          }
        }catch{case e: Throwable =>
          throw new Exception(
            s"Failed to generate unrolled defs for $defdef in ${implDef.symbol} in ${implDef.symbol.pos}",
            e
          )

        }
      }
    }

    override def transform(tree: global.Tree): global.Tree = {
      tree match{
        case md: ModuleDef =>
          val (removed0, allNewMethodsLists) = generateDefForwarders(md).unzip
          val removed = removed0.flatten.toSet
          val allNewMethods = allNewMethodsLists.flatten
          val classInfoType = md.symbol.moduleClass.info.asInstanceOf[ClassInfoType]
          val newClassInfoType = classInfoType.copy(decls = newScopeWith(allNewMethods.map(_.symbol) ++ classInfoType.decls:_*))

          md.symbol.moduleClass.setInfo(newClassInfoType)
          super.transform(
            treeCopy.ModuleDef(
              md,
              mods = md.mods,
              name = md.name,
              impl = treeCopy.Template(
                md.impl,
                parents = md.impl.parents,
                self = md.impl.self,
                body = md.impl.body.filter(t => !removed.contains(t.symbol)) ++ allNewMethods
              )
            )
          )
        case cd: ClassDef =>
          val (removed0, allNewMethodsLists) = generateDefForwarders(cd).unzip
          val removed = removed0.flatten.toSet
          val allNewMethods = allNewMethodsLists.flatten
          super.transform(
            treeCopy.ClassDef(
              cd,
              mods = cd.mods,
              name = cd.name,
              tparams = cd.tparams,
              impl = treeCopy.Template(
                cd.impl,
                parents = cd.impl.parents,
                self = cd.impl.self,
                body = cd.impl.body.filter(t => !removed.contains(t.symbol)) ++ allNewMethods
              )
            )
          )
        case _ => super.transform(tree)
      }
    }
  }
}
