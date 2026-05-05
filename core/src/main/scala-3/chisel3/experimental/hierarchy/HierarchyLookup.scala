// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import scala.quoted.*
import chisel3.experimental.hierarchy.core.{Hierarchy, Lookupable}
import chisel3.experimental.hierarchy.public

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
  // Resolve @public member of A from its string name and return its
  // symbol and widened type
  private def resolvePublicField[A: Type](
    name: String
  )(using q: Quotes): (q.reflect.Symbol, q.reflect.TypeRepr) = {
    import q.reflect.*
    val tpe = TypeRepr.of[A]
    val typeSym = tpe.typeSymbol
    // Handle vals inherited from parent of an instance: fieldMember
    // is declared in the Instance, while methodMember is the Scala 3
    // representation of vals inherited from the parent
    val fieldSym = typeSym.fieldMember(name) match {
      case s if s != Symbol.noSymbol => s
      case _ =>
        typeSym.methodMember(name).find(_.paramSymss.flatten.isEmpty).getOrElse(Symbol.noSymbol)
    }
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

        val lookupableTerm = summonOrAbort(
          refinedType,
          explanation =>
            s"value `$name` in ${TypeRepr.of[A].show} has type ${fieldTpe.show} which has no Lookupable instance"
        )

        val cTpe = extractCType(lookupableTerm.tpe.widen.dealias)
        cTpe.asType match {
          case '[c] =>
            val aux = lookupableTerm.asExprOf[Lookupable.Aux[t, c]]
            '{ hierarchyLookupAux[A, t, c](${ inst }, $selector)(using $aux) }
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
