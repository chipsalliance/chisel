// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import scala.quoted.*
import chisel3.experimental.hierarchy.core.{Hierarchy, Instance, Lookupable}
import chisel3.experimental.hierarchy.{instantiable, public}

private[hierarchy] inline def hierarchyLookupAux[A, B, C0](
  inst:        Hierarchy[A],
  inline that: A => B
)(
  using lookup: Lookupable.Aux[B, C0]
): C0 = {
  given chisel3.internal.MacroGenerated = new chisel3.internal.MacroGenerated {}
  inst._lookup(that)(using lookup, summon[chisel3.internal.MacroGenerated]).asInstanceOf[C0]
}

private[hierarchy] object HierarchyLookupMacro {
  // Search a symbol for a no-arg val/def member with the given name.
  // Used by resolvePublicField as a single-symbol probe.
  private def memberLookup(using q: Quotes)(sym: q.reflect.Symbol, name: String): q.reflect.Symbol = {
    import q.reflect.*
    sym.fieldMember(name) match {
      case s if s != Symbol.noSymbol => s
      case _ =>
        sym.methodMember(name).find(_.paramSymss.flatten.isEmpty).getOrElse(Symbol.noSymbol)
    }
  }

  // Collect the class symbols that contribute members to `tpe`.
  // For intersection types `A & B`, the syntactic typeSymbol picks one
  // arm only, so we descend into AndType to gather both.
  private def classParts(using q: Quotes)(tpe: q.reflect.TypeRepr): List[q.reflect.Symbol] = {
    import q.reflect.*
    def go(t: TypeRepr): List[Symbol] = t.dealias match {
      case AndType(l, r) => go(l) ++ go(r)
      case other         => List(other.typeSymbol).filter(_ != Symbol.noSymbol)
    }
    go(tpe).distinct
  }

  // Resolve @public member of A from its string name and return its
  // symbol and widened type
  private def resolvePublicField[A: Type](
    name: String
  )(using q: Quotes): (q.reflect.Symbol, q.reflect.TypeRepr) = {
    import q.reflect.*
    val tpe = TypeRepr.of[A]
    // Build the search order: each component of A (handling AndType)
    // first, then their inherited base classes. A direct lookup uses
    // `fieldMember`; falling back to a no-arg `methodMember` covers
    // inherited vals which Dotty represents as accessor methods.
    val seedSyms = classParts(tpe)
    val searchSyms =
      seedSyms ++ seedSyms.flatMap(_.typeRef.baseClasses).distinct
    val fieldSym = searchSyms.iterator
      .map(s => memberLookup(s, name))
      .find(_ != Symbol.noSymbol)
      .getOrElse(Symbol.noSymbol)
    if (fieldSym == Symbol.noSymbol) {
      report.errorAndAbort(
        s"value `$name` is not a member of ${tpe.show}"
      )
    }

    val publicSym = TypeRepr.of[public].typeSymbol
    if (!fieldSym.annotations.exists(_.tpe.typeSymbol == publicSym)) {
      report.errorAndAbort(
        s"value `$name` in ${tpe.show} is not marked @public"
      )
    }
    (fieldSym, tpe.memberType(fieldSym).widen)
  }

  // Generate a selector lambda for _lookup from the given field symbol
  private def genSelector[A: Type, T: Type](using q: Quotes)(
    fieldSym: q.reflect.Symbol
  ): Expr[A => T] = {
    import q.reflect.*
    Lambda(
      owner = Symbol.spliceOwner,
      tpe = MethodType(List("proto"))(_ => List(TypeRepr.of[A]), _ => TypeRepr.of[T]),
      rhsFn = (lamSym, args) => Select(args.head.asInstanceOf[Term], fieldSym).changeOwner(lamSym)
    ).asExprOf[A => T]
  }

  private def summonOrAbort(using q: Quotes)(
    tpe:    q.reflect.TypeRepr,
    errMsg: String => String
  ): q.reflect.Term = {
    import q.reflect.*
    Implicits.search(tpe) match {
      case s: ImplicitSearchSuccess => s.tree
      case f: ImplicitSearchFailure => report.errorAndAbort(errMsg(f.explanation))
    }
  }

  // Extract the refined type from Lookupable. For example:
  //   Lookupable[Seq[UInt]] { type C = Seq[UInt] }
  // Dotty will widen the path-dependent type lookup.C to Any. With
  // the extracted type, we can cast it to the correct refined type
  private def extractCType(using q: Quotes)(
    termTpe: q.reflect.TypeRepr
  ): q.reflect.TypeRepr = {
    import q.reflect.*
    def fromBounds(tp: TypeRepr): TypeRepr = tp match {
      case TypeBounds(_, hi) => hi
      case other             => other
    }
    def walk(tp: TypeRepr): Option[TypeRepr] = tp match {
      case Refinement(_, "C", info) => Some(fromBounds(info))
      case Refinement(parent, _, _) => walk(parent)
      case AndType(l, r)            => walk(l).orElse(walk(r))
      case _                        => None
    }
    walk(termTpe).getOrElse {
      val cType = TypeRepr.of[Lookupable[Any]].typeSymbol.typeMember("C")
      fromBounds(termTpe.memberType(cType))
    }
  }

  // Returns true if `tpe` (or any of its base classes) carries the
  // `@instantiable` annotation. Used to enable a synthesized fallback
  // Lookupable for user types that the Scala 2 macro would have given
  // an IsInstantiable parent.
  private def hasInstantiableAnnotation(using q: Quotes)(
    tpe: q.reflect.TypeRepr
  ): Boolean = {
    import q.reflect.*
    val instantiableSym = TypeRepr.of[instantiable].typeSymbol
    val sym = tpe.typeSymbol
    sym.annotations.exists(_.tpe.typeSymbol == instantiableSym) ||
    sym.typeRef.baseClasses.exists { bc =>
      bc.annotations.exists(_.tpe.typeSymbol == instantiableSym)
    }
  }

  def selectDynamicImpl[A: Type](
    inst:     Expr[Hierarchy[A]],
    nameExpr: Expr[String]
  )(using q: Quotes): Expr[Any] = {
    import q.reflect.*
    val name = nameExpr.valueOrAbort
    val (fieldSym, fieldTpe) = resolvePublicField[A](name)

    fieldTpe.asType match {
      case '[t] =>
        val selector = genSelector[A, t](fieldSym)
        val refinedType =
          if (fieldTpe <:< TypeRepr.of[chisel3.Data]) TypeRepr.of[Lookupable.Aux[t, t]]
          else TypeRepr.of[Lookupable[t]]

        val lookupableTermOpt = Implicits.search(refinedType) match {
          case s: ImplicitSearchSuccess => Some(s.tree)
          case _: ImplicitSearchFailure => None
        }

        lookupableTermOpt match {
          case Some(lookupableTerm) =>
            val cTpe = extractCType(lookupableTerm.tpe.widen.dealias)
            cTpe.asType match {
              case '[c] =>
                val aux = lookupableTerm.asExprOf[Lookupable.Aux[t, c]]
                '{ hierarchyLookupAux[A, t, c](${ inst }, $selector)(using $aux) }
            }
          case None if hasInstantiableAnnotation(fieldTpe) =>
            // Synthesize a Lookupable for @instantiable user types so
            // Scala 3 cross-compiles do not need IsInstantiable in the
            // type hierarchy. Mirrors Lookupable.lookupIsInstantiable.
            '{
              hierarchyLookupAux[A, t, Instance[t]](
                ${ inst },
                $selector
              )(using InstantiableLookupable.synthesize[t])
            }
          case None =>
            report.errorAndAbort(
              s"value `$name` in ${TypeRepr.of[A].show} has type ${fieldTpe.show} which has no Lookupable instance"
            )
        }
    }
  }

  def applyDynamicImpl[A: Type](
    inst:     Expr[Hierarchy[A]],
    nameExpr: Expr[String],
    argsExpr: Expr[Seq[Any]]
  )(using q: Quotes): Expr[Any] = {
    import q.reflect.*
    // Restrict method calls on Instance to explicit arguments. Splice
    // syntax like `inst.someMethod(args*)` should report an error
    val argTerms = argsExpr match {
      case Varargs(args) => args.toList.map(_.asTerm)
      case _             => report.errorAndAbort("applyDynamic on an Instance requires explicit varargs")
    }
    val selectExpr = selectDynamicImpl[A](inst, nameExpr)
    val cTpe = selectExpr.asTerm.tpe.widen
    cTpe.asType match {
      case '[c] =>
        // Find an apply on a container like List or Vec of the shape
        // `def apply(idx: Int)(sourceInfo: ...)`
        val applySym = cTpe.typeSymbol
          .methodMember("apply")
          .find { sym =>
            sym.paramSymss.headOption.exists { firstList =>
              firstList.length == argTerms.length &&
              firstList.lazyZip(argTerms).forall((p, a) => a.tpe <:< cTpe.memberType(p).widen)
            }
          }
          .getOrElse {
            report.errorAndAbort(
              s"no `apply` method on ${cTpe.show} matches arguments " +
                argTerms.map(_.tpe.show).mkString("(", ", ", ")")
            )
          }
        val qual = Typed(selectExpr.asTerm, TypeTree.of[c])
        val initial: Term = qual.select(applySym)
        val finalTerm = applySym.paramSymss.zipWithIndex.foldLeft(initial) { case (acc, (_, idx)) =>
          val args =
            if (idx == 0) argTerms
            else {
              val paramTpes = acc.tpe.widen match {
                case mt: MethodType => mt.paramTypes
                case _ => Nil
              }
              paramTpes.map { pt =>
                summonOrAbort(
                  pt,
                  explanation => s"could not summon implicit ${pt.show} when calling ${cTpe.show}.apply: $explanation"
                )
              }
            }
          Apply(acc, args)
        }
        finalTerm.asExpr
    }
  }
}
