package plugin

import java.io.FileWriter

import scala.tools.nsc
import nsc.Global
import nsc.Phase
import nsc.plugins.Plugin
import nsc.plugins.PluginComponent
import scala.reflect.internal.Flags
import scala.tools.nsc.transform.TypingTransformers

class ChiselPlugin(val global: Global) extends Plugin {
  val name = "chiselplugin"
  val description = "chisel's plugin"
  val components = List[PluginComponent](new ChiselComponent(global)/*, new Namer(global)*/)
}

class ChiselComponent(val global: Global) extends PluginComponent with TypingTransformers {
  import global._
  val runsAfter = List[String]("uncurry")
  override val runsRightAfter: Option[String] = Some("uncurry")
  val phaseName = "divbyzero"
  def newPhase(_prev: Phase) = new ChiselComponentPhase(_prev)
  class ChiselComponentPhase(prev: Phase) extends StdPhase(prev) {
    override def name = phaseName
    def apply(unit: CompilationUnit): Unit = {
      unit.body = new MyTypingTransformer(unit).transform(unit.body)
    }
  }

  class MyTypingTransformer(unit: CompilationUnit)
    extends TypingTransformer(unit) {

    def typeHasTrait(s: Type, name: String): Boolean = {
      s.parents.exists { p =>
        p.toString().toString == name  || typeHasTrait(p, name)
      }
    }
    var counter = 0
    val badFlags = Set(Flag.PARAM, Flag.SYNTHETIC, Flag.DEFERRED, Flags.TRIEDCOOKING, Flags.CASEACCESSOR, Flags.PARAMACCESSOR)
    val goodFlags = Set(Flag.PRIVATE, Flag.PROTECTED)
    def okFlags(mods: Modifiers): Boolean = {
      badFlags.forall{ x => !mods.hasFlag(x)}
    }

    def write(filename: String, body: String): Unit = {
      val fw = new FileWriter(s"$filename.txt")
      fw.write(body)
      fw.close()
    }
    def writeAST(filename: String, body: String): Unit = {
      write(filename, stringifyAST(body))
    }
    val original = "wantsLock"
    val goal = "WANTSLOCK"

    def okVal(dd: ValDef): Boolean = okFlags(dd.mods) && typeHasTrait(dd.tpt.tpe, "chisel3.Data")
    def originalVal(dd: ValDef): Boolean = dd.name == TermName(original) && okVal(dd)
    def goalVal(dd: ValDef): Boolean = dd.name == TermName(goal) && okVal(dd)

    def stringifyAST(ast: String): String = {
      var ntabs = 0
      val buf = new StringBuilder
      ast.zipWithIndex.foreach { case (c, idx) =>
        c match {
          case ' ' =>
          case '(' =>
            ntabs += 1
            buf ++= "(\n" + "| " * ntabs
          case ')' =>
            ntabs -= 1
            buf ++= "\n" + "| " * ntabs + ")"
          case ','=> buf ++= ",\n" + "| " * ntabs
          case  c if idx > 0 && ast(idx-1)==')' =>
            buf ++= "\n" + "| " * ntabs + c
          case c => buf += c
        }
      }
      buf.toString
    }

    override def transform(tree: Tree) = tree match {
        //[error] ValDef(Modifiers(), TermName("yyyy"), TypeTree(), Apply(Select(Apply(TypeApply(Select(Apply(TypeApply(Select(Ident(TermName("xyxy")), TermName("map")), List(TypeTree())), List(Block(List(DefDef(Modifiers(FINAL | METHOD | ARTIFACT), TermName("$anonfun$new"), List(), List(List(ValDef(Modifiers(PARAM | SYNTHETIC), TermName("x$2"), TypeTree(), EmptyTree))), TypeTree(), Typed(Apply(Select(Ident(TermName("x$2")), TermName("do_apply")), List(Literal(Constant(3)), Apply(TypeApply(Select(Select(Ident(scala), scala.Predef), TermName("implicitly")), List(TypeTree().setOriginal(Select(Select(Select(Ident(chisel3), chisel3.internal), chisel3.internal.sourceinfo), chisel3.internal.sourceinfo.SourceInfo)))), List(Typed(Apply(Select(New(TypeTree()), termNames.CONSTRUCTOR), List(Literal(Constant("DivZeroTest.scala")), Literal(Constant(24)), Literal(Constant(28)))), TypeTree()))), Apply(TypeApply(Select(Select(Ident(scala), scala.Predef), TermName("implicitly")), List(TypeTree().setOriginal(Select(Ident(chisel3), chisel3.CompileOptions)))), List(Typed(Apply(Select(Select(Ident(chisel3), chisel3.ExplicitCompileOptions), TermName("Strict")), List()), TypeTree()))))), TypeTree()))), Function(List(ValDef(Modifiers(PARAM | SYNTHETIC), TermName("x$2"), TypeTree(), EmptyTree)), Apply(Ident(TermName("$anonfun$new")), List(Ident(TermName("x$2")))))))), TermName("getOrElse")), List(TypeTree())), List(Block(List(DefDef(Modifiers(FINAL | METHOD | ARTIFACT), TermName("$anonfun$new"), List(), List(List()), TypeTree(), Apply(Select(Apply(Select(Select(Ident(chisel3), chisel3.package), TermName("fromBooleanToLiteral")), List(Literal(Constant(true)))), TermName("B")), List()))), Function(List(), Apply(Ident(TermName("$anonfun$new")), List()))))), TermName("macroName")), List(Literal(Constant("yyy")))))
        //[error] ValDef(Modifiers(), TermName("xxxx"), TypeTree(), Apply(Select(Apply(TypeApply(Select(Apply(TypeApply(Select(Ident(TermName("xyxy")), TermName("map")), List(TypeTree())), List(Block(List(DefDef(Modifiers(FINAL | METHOD | ARTIFACT), TermName("$anonfun$new"), List(), List(List(ValDef(Modifiers(PARAM | SYNTHETIC), TermName("x$1"), TypeTree(), EmptyTree))), TypeTree(), Typed(Apply(Select(Ident(TermName("x$1")), TermName("do_apply")), List(Literal(Constant(3)), Apply(TypeApply(Select(Select(Ident(scala), scala.Predef), TermName("implicitly")), List(TypeTree().setOriginal(Select(Select(Select(Ident(chisel3), chisel3.internal), chisel3.internal.sourceinfo), chisel3.internal.sourceinfo.SourceInfo)))), List(Typed(Apply(Select(New(TypeTree()), termNames.CONSTRUCTOR), List(Literal(Constant("DivZeroTest.scala")), Literal(Constant(23)), Literal(Constant(26)))), TypeTree()))), Apply(TypeApply(Select(Select(Ident(scala), scala.Predef), TermName("implicitly")), List(TypeTree().setOriginal(Select(Ident(chisel3), chisel3.CompileOptions)))), List(Typed(Apply(Select(Select(Ident(chisel3), chisel3.ExplicitCompileOptions), TermName("Strict")), List()), TypeTree()))))), TypeTree()))), Function(List(ValDef(Modifiers(PARAM | SYNTHETIC), TermName("x$1"), TypeTree(), EmptyTree)), Apply(Ident(TermName("$anonfun$new")), List(Ident(TermName("x$1")))))))), TermName("getOrElse")), List(TypeTree())), List(Block(List(DefDef(Modifiers(FINAL | METHOD | ARTIFACT), TermName("$anonfun$new"), List(), List(List()), TypeTree(), Apply(Select(Apply(Select(Select(Ident(chisel3), chisel3.package), TermName("fromBooleanToLiteral")), List(Literal(Constant(true)))), TermName("B")), List()))), Function(List(), Apply(Ident(TermName("$anonfun$new")), List()))))), TermName("macroName")), List(Literal(Constant("xxx")))))
      //case dd: ValDef if goalVal(dd) =>
      //  writeAST("goalRaw", showRaw(dd))
      //  write("goal", show(dd))
      //  write("goalNotes", dd.rhs.symbol.asInstanceOf[MethodSymbol].debugFlagString)
      //  dd
      case dd @ ValDef(mods, name, tpt, rhs) if okVal(dd) =>
        val TermName(str: String) = name
        //global.reporter.error(dd.pos, show(dd))
        val sel = localTyper.typed1(Select(rhs, TermName("macroName")), nsc.EXPRmode, MethodType(List(definitions.StringTpe.typeSymbol),tpt.tpe))
        val appl = localTyper.doTypedApply(rhs, sel, List(Literal(Constant(str))), nsc.EXPRmode, tpt.tpe)
        val ret = treeCopy.ValDef(dd, mods, name, tpt, appl)
        //global.reporter.error(ret.pos, show(ret))
        //writeAST("originalRaw", showRaw(dd))
        //write("original", show(dd))
        //writeAST("modifiedRaw", showRaw(ret))
        //write("modified", show(ret))
        //write("modifiedNotes", appl.symbol.toString)
        ret
      case _ =>
        super.transform(tree)
    }
  }

  val globalNamingStack = q"_root_.chisel3.internal.DynamicNamingStack"

  /** Base transformer that provides the val name transform.
    * Should not be instantiated, since by default this will recurse everywhere and break the
    * naming context variable bounds.
    */
  trait ValNameTransformer extends Transformer {
    val contextVar: TermName

    override def transform(tree: global.Tree): global.Tree = tree match {
      // Intentionally not prefixed with $mods, since modifiers usually mean the val definition
      // is in a non-transformable location, like as a parameter list.
      // TODO: is this exhaustive / correct in all cases?
      case q"val $tname: $tpt = $expr" => {
        val TermName(tnameStr: String) = tname
        val transformedExpr = super.transform(expr)
        q"val $tname: $tpt = $contextVar.name($transformedExpr, $tnameStr)"
      }
      case other => super.transform(other)
    }
  }

  /** Module-specific val name transform, containing logic to prevent from recursing into inner
    * classes and applies the naming transform on inner functions.
    */
  class ClassBodyTransformer(val contextVar: TermName) extends ValNameTransformer {
    override def transform(tree: Tree): Tree = tree match {
      case q"$mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..$stats }" => // scalastyle:ignore line.size.limit
        tree  // don't recurse into inner classes
      case q"$mods trait $tpname[..$tparams] extends { ..$earlydefns } with ..$parents { $self => ..$stats }" =>
        tree  // don't recurse into inner classes
      case q"$mods def $tname[..$tparams](...$paramss): $tpt = $expr" => {
        val Modifiers(_, _, annotations) = mods
        // don't apply naming transform twice
        val containsChiselName = annotations.map({q"new chiselName()" equalsStructure _}).fold(false)({_||_})
        // transforming overloaded initializers causes errors, and the transform isn't helpful
        val isInitializer = tname == TermName("<init>")
        if (containsChiselName || isInitializer) {
          tree
        } else {
          // apply chiselName transform by default
          val transformedExpr = transformHierarchicalMethod(expr)
          q"$mods def $tname[..$tparams](...$paramss): $tpt = $transformedExpr"
        }
      }
      case other => super.transform(other)
    }
  }

  /** Method-specific val name transform, handling the return case.
    */
  class MethodTransformer(val contextVar: TermName) extends ValNameTransformer {
    override def transform(tree: Tree): Tree = tree match {
      // TODO: better error messages when returning nothing
      case q"return $expr" => q"return $globalNamingStack.popReturnContext($expr, $contextVar)"
      // Do not recurse into methods
      case q"$mods def $tname[..$tparams](...$paramss): $tpt = $expr" => tree
      case other => super.transform(other)
    }
  }

  /** Applies the val name transform to a class body.
    * Closes context on top level or return local context to englobing context.
    * Closing context only makes sense when top level a Module.
    * A Module is always the naming top level.
    * Transformed classes can be either Module or standard class.
    */
  def transformClassBody(stats: List[global.Tree]): global.Tree = {
    val contextVar = global.freshTermName("namingContext")(global.currentFreshNameCreator)
    val transformedBody = (new ClassBodyTransformer(contextVar)).transformTrees(stats)
    // Note: passing "this" to popReturnContext is mandatory for propagation through non-module classes
    q"""
    val $contextVar = $globalNamingStack.pushContext()
    ..$transformedBody
    if($globalNamingStack.length == 1){
      $contextVar.namePrefix("")
    }
    $globalNamingStack.popReturnContext(this, $contextVar)
    """
  }

  /** Applies the val name transform to a method body, doing additional bookkeeping with the
    * context to allow names to propagate and prefix through the function call stack.
    */
  def transformHierarchicalMethod(expr: global.Tree): global.Tree = {
    val contextVar = global.freshTermName("namingContext")(global.currentFreshNameCreator)
    val transformedBody = (new MethodTransformer(contextVar)).transform(expr)

    q"""{
      val $contextVar = $globalNamingStack.pushContext()
      $globalNamingStack.popReturnContext($transformedBody, $contextVar)
    }
    """
  }

  /** Applies naming transforms to vals in the annotated module or method.
    *
    * For methods, a hierarchical naming transform is used, where it will try to give objects names
    * based on the call stack, assuming all functions on the stack are annotated as such and return
    * a non-AnyVal object. Does not recurse into inner functions.
    *
    * For modules, this serves as the root of the call stack hierarchy for naming purposes. Methods
    * will have chiselName annotations (non-recursively), but this does NOT affect inner classes.
    *
    * Basically rewrites all instances of:
    * val name = expr
    * to:
    * val name = context.name(expr, name)
    */
  def chiselName(annottees: global.Tree*): global.Tree = {
    var namedElts: Int = 0

    val transformed = annottees.map(annottee => annottee match {
      // scalastyle:off line.size.limit
      case q"$mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..$stats }" => {
        val transformedStats = transformClassBody(stats)
        namedElts += 1
        q"$mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..$transformedStats }"
      }
      case q"$mods object $tname extends { ..$earlydefns } with ..$parents { $self => ..$body }" => {
        val transformedBody = body.map {
          case q"$mods def $tname[..$tparams](...$paramss): $tpt = $expr" => {
            val transformedExpr = transformHierarchicalMethod(expr)
            namedElts += 1
            q"$mods def $tname[..$tparams](...$paramss): $tpt = $transformedExpr"
          }
          case other => other
        }
        q"$mods object $tname extends { ..$earlydefns } with ..$parents { $self => ..$transformedBody }"
      }
      // Currently disallow on traits, this won't work well with inheritance.
      case q"$mods def $tname[..$tparams](...$paramss): $tpt = $expr" => {
        val transformedExpr = transformHierarchicalMethod(expr)
        namedElts += 1
        q"$mods def $tname[..$tparams](...$paramss): $tpt = $transformedExpr"
      }
      case q"$vmods val $vname = { $mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns} with ..$parents { $self => ..$stats }; new $blah() }" => {
        val transformedStats = transformClassBody(stats)
        namedElts += 1
        q"$vmods val $vname = { $mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..$transformedStats }; new $blah() }"
      }
      case other =>
        //global.reporter.error(other.pos, s"@chiselName annotation may only be used on classes and methods, got ${showCode(other)}")
        other
    })

    // scalastyle:on line.size.limit

    q"..$transformed"
  }
}
