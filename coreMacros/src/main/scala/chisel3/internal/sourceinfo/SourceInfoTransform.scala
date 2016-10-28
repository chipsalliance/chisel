// See LICENSE for license details.

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
  def thisObj = c.prefix.tree
  def implicitSourceInfo = q"implicitly[_root_.chisel3.internal.sourceinfo.SourceInfo]"
  def implicitCompileOptions = q"implicitly[_root_.chisel3.core.CompileOptions]"
}

class WireTransform(val c: Context) extends SourceInfoTransformMacro {
  import c.universe._
  def apply[T: c.WeakTypeTag](t: c.Tree): c.Tree = {
    val tpe = weakTypeOf[T]
    q"$thisObj.do_apply($t, null.asInstanceOf[$tpe])($implicitSourceInfo, $implicitCompileOptions)"
  }
}

class UIntTransform(val c: Context) extends SourceInfoTransformMacro {
  import c.universe._
  def bitset(off: c.Tree, dat: c.Tree): c.Tree = {
    q"$thisObj.do_bitSet($off, $dat)($implicitSourceInfo)"
  }
}

// Module instantiation transform
class InstTransform(val c: Context) extends SourceInfoTransformMacro {
  import c.universe._
  def apply[T: c.WeakTypeTag](bc: c.Tree): c.Tree = {
    q"$thisObj.do_apply($bc)($implicitSourceInfo)"
  }
}

class MemTransform(val c: Context) extends SourceInfoTransformMacro {
  import c.universe._
  def apply[T: c.WeakTypeTag](size: c.Tree, t: c.Tree): c.Tree = {
    q"$thisObj.do_apply($size, $t)($implicitSourceInfo)"
  }
}

class RegTransform(val c: Context) extends SourceInfoTransformMacro {
  import c.universe._
  def apply[T: c.WeakTypeTag](t: c.Tree): c.Tree = {
    val tpe = weakTypeOf[T]
    q"$thisObj.do_apply($t, null.asInstanceOf[$tpe], null.asInstanceOf[$tpe])($implicitSourceInfo)"
  }
}

class MuxTransform(val c: Context) extends SourceInfoTransformMacro {
  import c.universe._
  def apply[T: c.WeakTypeTag](cond: c.Tree, con: c.Tree, alt: c.Tree): c.Tree = {
    val tpe = weakTypeOf[T]
    q"$thisObj.do_apply[$tpe]($cond, $con, $alt)($implicitSourceInfo)"
  }
}

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
  def fill(n: c.Tree)(gen: c.Tree): c.Tree = {
    q"$thisObj.do_fill($n)($gen)($implicitSourceInfo, $implicitCompileOptions)"
  }
  def contains(x: c.Tree)(ev: c.Tree): c.Tree = {
    q"$thisObj.do_contains($x)($implicitSourceInfo, $ev)"
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
  def doFuncTerm = {
    val funcName = c.macroApplication match {
      case q"$_.$funcName[..$_](...$_)" => funcName
      case _ => throw new Exception(s"Chisel Internal Error: Could not resolve function name from macro application: ${showCode(c.macroApplication)}")
    }
    TermName("do_" + funcName)
  }
}

class SourceInfoTransform(val c: Context) extends AutoSourceTransform {
  import c.universe._

  def noArg(): c.Tree = {
    q"$thisObj.$doFuncTerm($implicitSourceInfo)"
  }

  def thatArg(that: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($that)($implicitSourceInfo)"
  }

  def nArg(n: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($n)($implicitSourceInfo)"
  }

  def pArg(p: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($p)($implicitSourceInfo)"
  }

  def inArg(in: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($in)($implicitSourceInfo)"
  }

  def xArg(x: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($x)($implicitSourceInfo)"
  }

  def xyArg(x: c.Tree, y: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($x, $y)($implicitSourceInfo)"
  }
}

class CompileOptionsTransform(val c: Context) extends AutoSourceTransform {
  import c.universe._

  def thatArg(that: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($that)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def inArg(in: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($in)($implicitSourceInfo, $implicitCompileOptions)"
  }

  def pArg(p: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($p)($implicitSourceInfo, $implicitCompileOptions)"
  }
}

/** Special whitebox version of the blackbox SourceInfoTransform, used when fun things need to happen to satisfy the
  * type system while preventing the use of macro overrides.
  */
class SourceInfoWhiteboxTransform(val c: whitebox.Context) extends AutoSourceTransform {
  import c.universe._

  def noArg(): c.Tree = {
    q"$thisObj.$doFuncTerm($implicitSourceInfo)"
  }

  def thatArg(that: c.Tree): c.Tree = {
    q"$thisObj.$doFuncTerm($that)($implicitSourceInfo)"
  }
}
