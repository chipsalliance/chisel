// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context
import scala.reflect.runtime.{universe => ru}

import scala.collection.mutable

import chisel3._
import chisel3.experimental.{BaseModule, SourceInfo, UnlocatableSourceInfo}
import chisel3.reflect.DataMirror
import chisel3.reflect.DataMirror.internal.isSynthesizable
import chisel3.internal.Builder

/** Create an [[Instance]] of a [[Module]]
  *
  * Acts as a nicer API wrapping [[Definition]] and [[Instance]].
  * Used in a similar way to traditional module instantiation using `Module(...)`.
  *
  * {{{
  * val inst0: Instance[MyModule] = Instantiate(new MyModule(arg1, arg2))
  *
  * // Note that you cannot pass arguments by name (this will not compile)
  * val inst1 = Instantiate(new OtherModule(n = 3))
  *
  * // Instead, only pass arguments positionally
  * val inst2 = Instantiate(new OtherModule(3))
  * }}}
  *
  * ==Limitations==
  *   - The caching does not work for Modules that are inner classes. This is due to the fact that
  *     the WeakTypeTags for instances will be different and thus will not hit in the cache.
  *   - Passing parameters by name to module constructors is not supported.
  *   - User-defined types that wrap up Data will use the default Data equality and hashCode
  *     implementations which use referential equality, thus will not hit in the cache.
  */
object Instantiate extends InstantiateImpl {

  /** Create an `Instance` of a `Module`
    *
    * This is similar to `Module(...)` except that it returns an `Instance[_]` object.
    *
    * @param con module construction, must be actual call to constructor (`new MyModule(...)`)
    * @return constructed module `Instance`
    */
  def apply[A <: BaseModule](con: => A): Instance[A] = macro internal.instance[A]

  def definition[A <: BaseModule](con: => A): Definition[A] = macro internal.definition[A]

  def _instance[K, A <: BaseModule: ru.WeakTypeTag](
    args: K,
    f:    K => A
  )(
    implicit sourceInfo: SourceInfo
  ): Instance[A] = _instanceImpl(args, f)

  /** This is not part of the public API, do not call directly! */
  def _defination[K, A <: BaseModule: ru.WeakTypeTag](
    args: K,
    f:    K => A
  ): Definition[A] = _definitionImpl(args, f)

  private object internal {

    def instance[A <: BaseModule: c.WeakTypeTag](c: Context)(con: c.Tree): c.Tree = {
      import c.universe._

      def matchStructure(proto: List[List[Tree]], args: List[Tree]): List[List[Tree]] = {
        val it = args.iterator
        proto.map(_.map(_ => it.next()))
      }

      def untupleFuncArgsToConArgs(argss: List[List[Tree]], args: List[Tree], funcArg: Ident): List[List[Tree]] = {
        var n = 0
        argss.map { inner =>
          inner.map { _ =>
            n += 1
            val term = TermName(s"_$n")
            q"$funcArg.$term"
          }
        }
      }

      con match {
        case q"new $tpname[..$tparams](...$argss)" =>
          // We first flatten the [potentially] multiple parameter lists into a single tuple (size 0 and 1 are special)
          val args = argss.flatten
          val nargs = args.size

          val funcArg = Ident(TermName("arg"))

          // 0 and 1 arguments to the constructor are special (ie. there isn't a tuple)
          val conArgs: List[List[Tree]] = nargs match {
            case 0 => Nil
            // Must match structure for case of only 1 implicit argument (ie. ()(arg))
            case 1 => matchStructure(argss, List(funcArg))
            case _ => untupleFuncArgsToConArgs(argss, args, funcArg)
          }

          // We can't quasi-quote this too early, needs to be splatted in the later context
          // widen turns singleton type into nearest non-singleton type, eg. Int(3) => Int
          val funcArgTypes = args.map(_.asInstanceOf[Tree].tpe.widen)
          val constructor = q"(($funcArg: (..$funcArgTypes)) => new $tpname[..$tparams](...$conArgs))"
          val tup = q"(..$args)"
          q"chisel3.experimental.hierarchy.Instantiate._instance[(..$funcArgTypes), $tpname]($tup, $constructor)"

        case _ =>
          val msg =
            s"Argument to Instantiate(...) must be of form 'new <T <: chisel3.Module>(<arguments...>)'.\n" +
              "Note that named arguments are currently not supported.\n" +
              s"Got: '$con'"
          c.error(con.pos, msg)
          con
      }
    }
    // definition cannot be private, but it can be inside of a private object which hides it from the public
    // API and ScalaDoc
    def definition[A <: BaseModule: c.WeakTypeTag](c: Context)(con: c.Tree): c.Tree = {
      import c.universe._

      def matchStructure(proto: List[List[Tree]], args: List[Tree]): List[List[Tree]] = {
        val it = args.iterator
        proto.map(_.map(_ => it.next()))
      }

      // The arguments to the constructor have been flattened to a single tuple
      // When there is more than 1 argument, we need to turn the flat tuple back into the multiple parameter lists,
      //   eg. (arg: (A, B, C, D)) becomes (arg._1)(arg._2, arg._3)(arg._4)
      def untupleFuncArgsToConArgs(argss: List[List[Tree]], args: List[Tree], funcArg: Ident): List[List[Tree]] = {
        var n = 0
        argss.map { inner =>
          inner.map { _ =>
            n += 1
            val term = TermName(s"_$n")
            q"$funcArg.$term"
          }
        }
      }

      con match {
        case q"new $tpname[..$tparams](...$argss)" =>
          // We first flatten the [potentially] multiple parameter lists into a single tuple (size 0 and 1 are special)
          val args = argss.flatten
          val nargs = args.size

          val funcArg = Ident(TermName("arg"))

          // 0 and 1 arguments to the constructor are special (ie. there isn't a tuple)
          val conArgs: List[List[Tree]] = nargs match {
            case 0 => Nil
            // Must match structure for case of only 1 implicit argument (ie. ()(arg))
            case 1 => matchStructure(argss, List(funcArg))
            case _ => untupleFuncArgsToConArgs(argss, args, funcArg)
          }

          // We can't quasi-quote this too early, needs to be splatted in the later context
          // widen turns singleton type into nearest non-singleton type, eg. Int(3) => Int
          val funcArgTypes = args.map(_.asInstanceOf[Tree].tpe.widen)
          val constructor = q"(($funcArg: (..$funcArgTypes)) => new $tpname[..$tparams](...$conArgs))"
          val tup = q"(..$args)"
          q"chisel3.experimental.hierarchy.Instantiate._defination[(..$funcArgTypes), $tpname]($tup, $constructor)"

        case _ =>
          val msg =
            s"Argument to Instantiate.defination(...) must be of form 'new <T <: chisel3.Module>(<arguments...>)'.\n" +
              "Note that named arguments are currently not supported.\n" +
              s"Got: '$con'"
          c.error(con.pos, msg)
          con
      }
    }
  }
}
