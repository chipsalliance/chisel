// SPDX-License-Identifier: Apache-2.0

// This file transform macro definitions to explicitly add implicit source info to Chisel method
// calls.

package chisel3.internal.sourceinfo

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context
import scala.reflect.macros.whitebox

/** Transforms a function call so that it can both provide implicit-style source information and
  * have a chained apply call. Without macros, only one is possible, since having a implicit
  * argument in the definition will cause the compiler to interpret a chained apply as an
  * explicit implicit argument and give type errors.
  *
  * Instead of an implicit argument, the public-facing function no longer takes a SourceInfo at all.
  * The macro transforms the public-facing function into a call to an internal function that takes
  * an explicit SourceInfo by inserting an implicitly[SourceInfo] as the explicit argument.
  */
trait SourceInfoTransformMacro {
  val c: Context
  import c.universe._
  def thisObj: Tree = c.prefix.tree
  def implicitSourceInfo = q"implicitly[_root_.chisel3.internal.sourceinfo.SourceInfo]"
  def implicitCompileOptions = q"implicitly[_root_.chisel3.CompileOptions]"
}

// Workaround for https://github.com/sbt/sbt/issues/3966
object UIntTransform
class UIntTransform(val c: Context) extends SourceInfoTransformMacro {
  import c.universe._
  def bitset(off: c.Tree, dat: c.Tree): c.Tree = {
    q"$thisObj.do_bitSet($off, $dat)($implicitSourceInfo, $implicitCompileOptions)"
  }
}

// Workaround for https://github.com/sbt/sbt/issues/3966
object InstTransform
// Module instantiation transform
class InstTransform(val c: Context) extends SourceInfoTransformMacro {
  import c.universe._
  def apply[T: c.WeakTypeTag](bc: c.Tree): c.Tree = {
    q"$thisObj.do_apply($bc)($implicitSourceInfo, $implicitCompileOptions)"
  }
}

// Workaround for https://github.com/sbt/sbt/issues/3966
object DefinitionTransform
// Module instantiation transform
class DefinitionTransform(val c: Context) extends SourceInfoTransformMacro {
  import c.universe._
  def apply[T: c.WeakTypeTag](proto: c.Tree): c.Tree = {
    q"$thisObj.do_apply($proto)($implicitSourceInfo, $implicitCompileOptions)"
  }
}

object DefinitionWrapTransform
// Module instantiation transform
class DefinitionWrapTransform(val c: Context) extends SourceInfoTransformMacro {
  import c.universe._
  def wrap[T: c.WeakTypeTag](proto: c.Tree): c.Tree = {
    q"$thisObj.do_wrap($proto)($implicitSourceInfo)"
  }
}

// Workaround for https://github.com/sbt/sbt/issues/3966
object InstanceTransform
// Module instantiation transform
class InstanceTransform(val c: Context) extends SourceInfoTransformMacro {
  import c.universe._
  def apply[T: c.WeakTypeTag](definition: c.Tree): c.Tree = {
    q"$thisObj.do_apply($definition)($implicitSourceInfo, $implicitCompileOptions)"
  }
}

// Workaround for https://github.com/sbt/sbt/issues/3966
object MemTransform
class MemTransform(val c: Context) extends SourceInfoTransformMacro {
  import c.universe._
  def apply[T: c.WeakTypeTag](size: c.Tree, t: c.Tree): c.Tree = {
    q"$thisObj.do_apply($size, $t)($implicitSourceInfo, $implicitCompileOptions)"
  }
  def apply_ruw[T: c.WeakTypeTag](size: c.Tree, t: c.Tree, ruw: c.Tree): c.Tree = {
    q"$thisObj.do_apply($size, $t, $ruw)($implicitSourceInfo, $implicitCompileOptions)"
  }
}

// Workaround for https://github.com/sbt/sbt/issues/3966
object MuxTransform
class MuxTransform(val c: Context) extends SourceInfoTransformMacro {
  import c.universe._
  def apply[T: c.WeakTypeTag](cond: c.Tree, con: c.Tree, alt: c.Tree): c.Tree = {
    val tpe = weakTypeOf[T]
    q"$thisObj.do_apply[$tpe]($cond, $con, $alt)($implicitSourceInfo, $implicitCompileOptions)"
  }
}

// Workaround for https://github.com/sbt/sbt/issues/3966
object VecTransform
class VecTransform(val c: Context) extends SourceInfoTransformMacro {
  import c.universe._
  def apply_elts(elts: c.Tree): c.Tree = {
    q"$thisObj.do_apply($elts)($implicitSourceInfo, $implicitCompileOptions)"
  }
  def apply_elt0(elt0: c.Tree, elts: c.Tree*): c.Tree = {
    q"$thisObj.do_apply($elt0, ..$elts)($implicitSourceInfo, $implicitCompileOptions)"
  }
  def tabulate(n: c.Tree)(gen: c.Tree): c.Tree = {
    q"$thisObj.do_tabulate($n)($gen)($implicitSourceInfo, $implicitCompileOptions)"
  }
  def tabulate2D(n: c.Tree, m: c.Tree)(gen: c.Tree): c.Tree = {
    q"$thisObj.do_tabulate($n,$m)($gen)($implicitSourceInfo, $implicitCompileOptions)"
  }
  def tabulate3D(n: c.Tree, m: c.Tree, p: c.Tree)(gen: c.Tree): c.Tree = {
    q"$thisObj.do_tabulate($n,$m,$p)($gen)($implicitSourceInfo, $implicitCompileOptions)"
  }
  def fill(n: c.Tree)(gen: c.Tree): c.Tree = {
    q"$thisObj.do_fill($n)($gen)($implicitSourceInfo, $implicitCompileOptions)"
  }
  def fill2D(n: c.Tree, m: c.Tree)(gen: c.Tree): c.Tree = {
    q"$thisObj.do_fill($n,$m)($gen)($implicitSourceInfo, $implicitCompileOptions)"
  }
  def fill3D(n: c.Tree, m: c.Tree, p: c.Tree)(gen: c.Tree): c.Tree = {
    q"$thisObj.do_fill($n,$m,$p)($gen)($implicitSourceInfo, $implicitCompileOptions)"
  }
  def fill4D(n: c.Tree, m: c.Tree, p: c.Tree, q: c.Tree)(gen: c.Tree): c.Tree = {
    q"$thisObj.do_fill($n,$m,$p,$q)($gen)($implicitSourceInfo, $implicitCompileOptions)"
  }
  def iterate(start: c.Tree, len: c.Tree)(f: c.Tree): c.Tree = {
    q"$thisObj.do_iterate($start,$len)($f)($implicitSourceInfo, $implicitCompileOptions)"
  }
  def contains(x: c.Tree)(ev: c.Tree): c.Tree = {
    q"$thisObj.do_contains($x)($implicitSourceInfo, $ev, $implicitCompileOptions)"
  }
  def reduceTree(redOp: c.Tree, layerOp: c.Tree): c.Tree = {
    q"$thisObj.do_reduceTree($redOp,$layerOp)($implicitSourceInfo, $implicitCompileOptions)"
  }
  def reduceTreeDefault(redOp: c.Tree): c.Tree = {
    q"$thisObj.do_reduceTree($redOp)($implicitSourceInfo, $implicitCompileOptions)"
  }
}

/** "Automatic" source information transform / insertion macros, which generate the function name
  * based on the macro invocation (instead of explicitly writing out every transform).
  */
abstract class AutoSourceTransform extends SourceInfoTransformMacro {
  import c.universe._

  /** Returns the TermName of the transformed function, which is the applied function name with do_
    * prepended.
    */
  def doFuncTerm: TermName = {
    val funcName = c.macroApplication match {
      case q"$_.$funcName[..$_](...$_)" => funcName
      case _ =>
        throw new Exception(
          s"Chisel Internal Error: Could not resolve function name from macro application: ${showCode(c.macroApplication)}"
        )
    }
    TermName("do_" + funcName)
  }
}

// Workaround for https://github.com/sbt/sbt/issues/3966
object SourceInfoTransform
class SourceInfoTransform(val c: Context) extends AutoSourceTransform {
  import c.universe._

  def noArg: c.Tree = {
    q"$thisObj.$doFuncTerm($implicitSourceInfo, $implicitCompileOptions)"
  }

  /** Necessary for dummy methods to auto-apply their arguments to this macro */
  def noArgDummy(dummy: c.Tree*): c.Tree = {
    q"$thisObj.$doFuncTerm($implicitSourceInfo, $implicitCompileOptions)"
  }

  def thatArg(that: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($that)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def nArg(n: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($n)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def pArg(p: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($p)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def inArg(in: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($in)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def xArg(x: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($x)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def xyArg(x: c.Tree, y: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($x, $y)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def nxArg(n: c.Tree, x: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($n, $x)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def idxEnClockArg(idx: c.Tree, en: c.Tree, clock: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($idx, $en, $clock)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def xEnArg(x: c.Tree, en: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($x, $en)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def arArg(a: c.Tree, r: c.Tree*): c.Tree = {
    q"$thisObj.$doFuncTerm($a, ..$r)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def rArg(r: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($r)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def nInArg(n: c.Tree, in: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($n, $in)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def nextEnableArg(next: c.Tree, enable: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($next, $enable)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def nextInitEnableArg(next: c.Tree, init: c.Tree, enable: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($next, $init, $enable)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def inNArg(in: c.Tree, n: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($in, $n)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def inNEnArg(in: c.Tree, n: c.Tree, en: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($in, $n, $en)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def inNResetEnArg(in: c.Tree, n: c.Tree, reset: c.Tree, en: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($in, $n, $reset, $en)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def inNResetDataArg(in: c.Tree, n: c.Tree, resetData: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($in, $n, $resetData)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def inNResetDataEnArg(in: c.Tree, n: c.Tree, resetData: c.Tree, en: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($in, $n, $resetData, $en)($implicitSourceInfo, $implicitCompileOptions)"
  }
}

// Workaround for https://github.com/sbt/sbt/issues/3966
object CompileOptionsTransform
class CompileOptionsTransform(val c: Context) extends AutoSourceTransform {
  import c.universe._

  def thatArg(that: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($that)($implicitCompileOptions)"
  }

  def inArg(in: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($in)($implicitCompileOptions)"
  }

  def pArg(p: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($p)($implicitCompileOptions)"
  }
}

// Workaround for https://github.com/sbt/sbt/issues/3966
object SourceInfoWhiteboxTransform

/** Special whitebox version of the blackbox SourceInfoTransform, used when fun things need to
  * happen to satisfy the type system while preventing the use of macro overrides.
  */
class SourceInfoWhiteboxTransform(val c: whitebox.Context) extends AutoSourceTransform {
  import c.universe._

  def noArg: c.Tree = {
    q"$thisObj.$doFuncTerm($implicitSourceInfo, $implicitCompileOptions)"
  }

  /** Necessary for dummy methods to auto-apply their arguments to this macro */
  def noArgDummy(dummy: c.Tree*): c.Tree = {
    q"$thisObj.$doFuncTerm($implicitSourceInfo, $implicitCompileOptions)"
  }

  def thatArg(that: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($that)($implicitSourceInfo, $implicitCompileOptions)"
  }
}

// Workaround for https://github.com/sbt/sbt/issues/3966
object IntLiteralApplyTransform

class IntLiteralApplyTransform(val c: Context) extends AutoSourceTransform {
  import c.universe._

  def safeApply(x: c.Tree): c.Tree = {
    c.macroApplication match {
      case q"$_.$clazz($lit).$func.apply($arg)" =>
        if (
          Set("U", "S").contains(func.toString) &&
          Set("fromStringToLiteral", "fromIntToLiteral", "fromLongToIteral", "fromBigIntToLiteral").contains(
            clazz.toString
          )
        ) {
          val msg =
            s"""Passing an Int to .$func is usually a mistake: It does *not* set the width but does a bit extract.
               |Did you mean .$func($arg.W)?
               |If you do want bit extraction, use .$func.extract($arg) instead.
               |""".stripMargin
          c.warning(c.enclosingPosition, msg)
        }
      case _ => // do nothing
    }
    q"$thisObj.$doFuncTerm($x)($implicitSourceInfo, $implicitCompileOptions)"
  }
}
