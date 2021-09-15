// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import scala.collection.mutable
import scala.reflect.internal.Flags
import scala.tools.nsc
import scala.tools.nsc.{Global, Phase}
import scala.tools.nsc.plugins.PluginComponent
import scala.tools.nsc.transform.TypingTransformers

// The component of the chisel plugin. Not sure exactly what the difference is between
//   a Plugin and a PluginComponent.
class InstanceComponent(val global: Global) extends PluginComponent with TypingTransformers {
  import global._
  val runsAfter: List[String] = List[String]("typer")
  val phaseName: String = "instancecomponent"
  def newPhase(_prev: Phase): InstanceComponentPhase = new InstanceComponentPhase(_prev)
  class InstanceComponentPhase(prev: Phase) extends StdPhase(prev) {
    override def name: String = phaseName
    def apply(unit: CompilationUnit): Unit = {
      // This plugin doesn't work on Scala 2.11 nor Scala 3. Rather than complicate the sbt build flow,
      // instead we just check the version and if its an early Scala version, the plugin does nothing
      val scalaVersion = scala.util.Properties.versionNumberString.split('.')
      if (scalaVersion(0).toInt == 2 && scalaVersion(1).toInt >= 12) {
        unit.body = new MyInstanceTransformer(unit).transform(unit.body)
      }
    }
  }

  class MyInstanceTransformer(unit: CompilationUnit)
    extends TypingTransformer(unit) {

    private def shouldMatchGen(bases: Tree*): Type => Boolean = {
      val cache = mutable.HashMap.empty[Type, Boolean]
      val baseTypes = bases.map(inferType)

      // If subtype of one of the base types, it's a match!
      def terminate(t: Type): Boolean = baseTypes.exists(t <:< _)

      // Recurse through subtype hierarchy finding containers
      // Seen is only updated when we recurse into type parameters, thus it is typically small
      def recShouldMatch(s: Type, seen: Set[Type]): Boolean = {
        def outerMatches(t: Type): Boolean = {
          val str = t.toString
          str.startsWith("Option[") || str.startsWith("Iterable[")
        }
        if (terminate(s)) {
          true
        } else if (seen.contains(s)) {
          false
        } else if (outerMatches(s)) {
          // These are type parameters, loops *are* possible here
          recShouldMatch(s.typeArgs.head, seen + s)
        } else {
          // This is the standard inheritance hierarchy, Scalac catches loops here
          s.parents.exists( p => recShouldMatch(p, seen) )
        }
      }

      // If doesn't match container pattern, exit early
      def earlyExit(t: Type): Boolean = {
        !(t.matchesPattern(inferType(tq"Iterable[_]")) || t.matchesPattern(inferType(tq"Option[_]")))
      }

      // Return function so that it captures the cache
      { q: Type =>
        cache.getOrElseUpdate(q, {
          // First check if a match, then check early exit, then recurse
          if(terminate(q)){
            true
          } else if(earlyExit(q)) {
            false
          } else {
            recShouldMatch(q, Set.empty)
          }
        })
      }
    }


    private val shouldMatchData      : Type => Boolean = shouldMatchGen(tq"chisel3.Data")
    private val shouldMatchDataOrMem : Type => Boolean = shouldMatchGen(tq"chisel3.Data", tq"chisel3.MemBase[_]")
    private val shouldMatchModule    : Type => Boolean = shouldMatchGen(tq"chisel3.experimental.BaseModule")
    private val shouldMatchInstance  : Type => Boolean = shouldMatchGen(tq"chisel3.experimental.hierarchy.Instance[_]")

    // Given a type tree, infer the type and return it
    private def inferType(t: Tree): Type = localTyper.typed(t, nsc.Mode.TYPEmode).tpe
    private def inferTreeType(t: Tree): Tree = localTyper.typed(t, nsc.Mode.TYPEmode)

    // Indicates whether a ValDef is properly formed to get name
    // TODO Unify with okVal
    private def okUnapply(dd: ValDef): Boolean = {

      // These were found through trial and error
      def okFlags(mods: Modifiers): Boolean = {
        val badFlags = Set(
          Flag.PARAM,
          Flag.DEFERRED,
          Flags.TRIEDCOOKING,
          Flags.CASEACCESSOR,
          Flags.PARAMACCESSOR
        )
        val goodFlags = Set(
          Flag.SYNTHETIC,
          Flag.ARTIFACT
        )
        goodFlags.forall(f => mods.hasFlag(f)) && badFlags.forall(f => !mods.hasFlag(f))
      }

      // Ensure expression isn't null, as you can't call `null.autoName("myname")`
      val isNull = dd.rhs match {
        case Literal(Constant(null)) => true
        case _ => false
      }
      val tpe = inferType(dd.tpt)
      definitions.isTupleType(tpe) && okFlags(dd.mods) && !isNull && dd.rhs != EmptyTree
    }

    private def findUnapplyNames(tree: Tree): Option[List[String]] = {
      val applyArgs: Option[List[Tree]] = tree match {
        case Match(_, List(CaseDef(_, _, Apply(_, args)))) => Some(args)
        case _ => None
      }
      applyArgs.flatMap { args =>
        var ok = true
        val result = mutable.ListBuffer[String]()
        args.foreach {
          case Ident(TermName(name)) => result += name
          // Anything unexpected and we abort
          case _                     => ok = false
        }
        if (ok) Some(result.toList) else None
      }
    }

    // Whether this val is directly enclosed by a Bundle type
    private def inBundle(dd: ValDef): Boolean = {
      dd.symbol.logicallyEnclosingMember.thisType <:< inferType(tq"chisel3.Bundle")
    }

    private def stringFromTermName(name: TermName): String =
      name.toString.trim() // Remove trailing space (Scalac implementation detail)

    // Method called by the compiler to modify source tree
    override def transform(tree: Tree): Tree = tree match {
      // Check if a subtree is a candidate
      case dd @ CaseDef(pat: Tree, guard: Tree, body: Tree) if(show(pat).contains("Boxy")) =>
        //val (newPat, name) = pat match {
        //    case bind@Bind(name, body) => 
        //      //globalError(s"Bind ${show(name)}\t${showRaw(name)}")
        //      val newBody = body match {
        //          case tped@Typed(expr, tpe) =>
        //            //globalError(s"Typed ${show(expr)}\t${showRaw(expr)}")
        //            val newTpe = tpe match {
        //                case tt: TypeTree =>
        //                  val newAppliedTypeTree = tt.original match {
        //                      case att@AppliedTypeTree(tpt, args) => 
        //                        //globalError(s"AppliedTypeTree ${show(tpt)}\t${showRaw(tpt)}")
        //                        //globalError(s"AppliedTypeTree ${show(args)}\t${showRaw(args)}")
        //                        val newArgs = args match {
        //                            case lst@List(t: TypeTree) =>
        //                                List(Annotated(Apply(Select(New(Ident(TypeName("unchecked"))), termNames.CONSTRUCTOR), List()), t))
        //                                //List(t)
        //                            case other => ???
        //                        }
        //                        AppliedTypeTree(tpt, newArgs)
        //                        //att
        //                      case other =>
        //                        //globalError(s"OTHER ${show(tpe)}\t${showRaw(tpe)}")
        //                        other
        //                  }
        //                  treeCopy.TypeTree(tt).setOriginal(newAppliedTypeTree)
        //                  //tt
        //                case other =>
        //                  globalError(s"OTHER2 ${show(tpe)}\t${showRaw(tpe)}")
        //                  other

        //            }
        //            treeCopy.Typed(tped, expr, newTpe)
        //            //tped
        //      }
        //      (treeCopy.Bind(bind, name, newBody), name)
        //    case other =>
        //      //globalError(s"OTHER ${show(other)}\n${showRaw(other)}")
        //      other
        //}
        //globalError(showRaw(guard))
        //globalError(show(guard))
        //treeCopy.CaseDef(dd, newPat, guard, body)
        super.transform(tree)
      // Otherwise, continue
      case _ => super.transform(tree)
    }
  }
}
