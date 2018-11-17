// See LICENSE for license details.

package firrtl.transforms

import firrtl._

import firrtl.analyses.InstanceGraph
import firrtl.annotations.{Named, CircuitName, ModuleName, ComponentName}
import firrtl.ir
import firrtl.passes.{Uniquify, PassException}
import firrtl.Utils.v_keywords
import firrtl.Mappers._
import scala.collection.mutable

/** Transform that removes collisions with reserved keywords
  * @param keywords a set of reserved words
  * @define implicitRename @param renames the [[RenameMap]] to query when renaming
  * @define implicitNamespace @param ns an encolosing [[Namespace]] with which new names must not conflict
  * @define implicitScope @param scope the enclosing scope of this name. If [[None]], then this is a [[Circuit]] name
  */
class RemoveKeywordCollisions(keywords: Set[String]) extends Transform {
  val inputForm: CircuitForm = LowForm
  val outputForm: CircuitForm = LowForm
  private type Renames = mutable.HashMap[String, String]
  private type ModuleType = mutable.HashMap[String, ir.Type]
  private val inlineDelim = "_"

  /** Generate a new name, by appending underscores, that will not conflict with the existing namespace
    * @param n a name
    * @param ns a [[Namespace]]
    * @return a conflict-free name
    * @note prefix uniqueness is not respected
    */
  private def safeName(n: String, ns: Namespace): String =
    Uniquify.findValidPrefix(n + inlineDelim, Seq(""), ns.cloneUnderlying ++ keywords)

  /** Modify a name to not conflict with a Verilog keywords while respecting existing renames and a namespace
    * @param n the name to rename
    * @param renames the [[RenameMap]] to query when renaming
    * $implicitRename
    * $implicitNamespace
    * $implicitScope
    * @return a name without keyword conflicts
    */
  private def onName(n: String)(implicit renames: RenameMap, ns: Namespace, scope: Option[Named]): String = {

    // Convert a [[String]] into [[Named]] based on the provided scope.
    def wrap(name: String, scope: Option[Named]): Named = scope match {
      case None                     => CircuitName(name)
      case Some(cir: CircuitName)   => ModuleName(name, cir)
      case Some(mod: ModuleName)    => ComponentName(name, mod)
      case Some(com: ComponentName) => ComponentName(s"${com.name}.$name", com.module)
    }

    val named = wrap(n, scope)

    // If this has already been renamed use that name. If it conflicts with a keyword, determine a new, safe name and
    // update the renames. Otherwise, leave it alone.
    val namedx: Seq[Named] = renames.get(named) match {
      case Some(x) => x
      case None if keywords(n) =>
        val sn = wrap(safeName(n, ns), scope)
        renames.rename(named, sn)
        Seq(sn)
      case _ => Seq(wrap(n, scope))
    }

    namedx match {
      case Seq(ComponentName(n, _)) => n
      case Seq(ModuleName(n, _))    => n
      case Seq(CircuitName(n))      => n
      case x => throw new PassException(
        s"Verilog renaming shouldn't result in multiple renames, but found '$named -> $namedx'")
    }
  }

  /** Rename the fields of a [[Type]] to match the ports of an instance
    * @param t the type to rename
    * $implicitRename
    * $implicitNamespace
    * $implicitScope
    * @return a [[Type]] with updated names
    * @note This is not intended for fixing arbitrary types, only [[BundleType]] in instance [[WRef]]s
    */
  private def onType(t: ir.Type)
                    (implicit renames: RenameMap,
                     ns: Namespace,
                     scope: Option[ModuleName]): ir.Type = t match {
    case b: ir.BundleType => b.copy(fields = b.fields.map(f => f.copy(name = onName(f.name))))
    case _                 => t
  }

  /** Rename an [[Expression]] to respect existing renames and avoid keyword collisions
    * @param e the [[Expression]] to rename
    * $implicitRename
    * $implicitNamespace
    * $implicitScope
    * @return an [[Expression]] without keyword conflicts
    */
  private def onExpression(e: ir.Expression)
                          (implicit renames: RenameMap,
                           ns: Namespace,
                           scope: Option[ModuleName],
                           iToM: mutable.Map[ComponentName, ModuleName],
                           modType: ModuleType): ir.Expression = e match {
    case wsf@ WSubField(wr@ WRef(name, _, InstanceKind, _), port, _, _) =>
      val subInst = ComponentName(name, scope.get)
      val subModule = iToM(subInst)
      val subPort = ComponentName(port, subModule)

      val wrx = wr.copy(
        name = renames.get(subInst).orElse(Some(Seq(subInst))).get.head.name,
        tpe = modType(subModule.name))

      wsf.copy(
        expr = wrx,
        name = renames.get(subPort).orElse(Some(Seq(subPort))).get.head.name)
    case wr: WRef => wr.copy(name=onName(wr.name))
    case ex       => ex.map(onExpression)
  }

  /** Rename a [[Statement]] to respect existing renames and avoid keyword collisions
    * $implicitRename
    * $implicitNamespace
    * $implicitScope
    * @return a [[Statement]] without keyword conflicts
    */
  private def onStatement(s: ir.Statement)
                         (implicit renames: RenameMap,
                          ns: Namespace,
                          scope: Option[ModuleName],
                          iToM: mutable.Map[ComponentName, ModuleName],
                          modType: ModuleType): ir.Statement = s match {
    case wdi: WDefInstance =>
      val subModule = ModuleName(wdi.module, scope.get.circuit)
      val modulex = renames.get(subModule).orElse(Some(Seq(subModule))).get.head.name
      val wdix = wdi.copy(module = modulex,
                          name = onName(wdi.name),
                          tpe = onType(wdi.tpe)(renames, ns, Some(ModuleName(modulex, scope.get.circuit))))
      iToM(ComponentName(wdi.name, scope.get)) = ModuleName(wdix.module, scope.get.circuit)
      wdix
    case _ => s
        .map(onStatement)
        .map(onExpression)
        .map(onName)
  }

  /** Rename a [[Port]] to avoid keyword collisions
    * $implicitRename
    * $implicitNamespace
    * $implicitScope
    * @return a [[Port]] without keyword conflicts
    */
  private def onPort(p: ir.Port)(implicit renames: RenameMap, ns: Namespace, scope: Option[ModuleName]): ir.Port =
    p.copy(name = onName(p.name))

  /** Rename a [[DefModule]] and it's internals (ports and statements) to fix keyword collisions and update instance
    * references to respect previous renames
    * @param renames a [[RenameMap]]
    * @param circuit the enclosing [[CircuitName]]
    * @return a [[DefModule]] without keyword conflicts
    */
  private def onModule(renames: RenameMap,
                       circuit: CircuitName,
                       modType: ModuleType)
                      (m: ir.DefModule): ir.DefModule = {
    implicit val moduleNamespace: Namespace = Namespace(m)
    implicit val scope: Option[ModuleName] = Some(ModuleName(m.name, circuit))
    implicit val r: RenameMap = renames
    implicit val mType: ModuleType = modType

    // Store local renames of refs to instances to their renamed modules. This is needed when renaming port connections
    // on subfields where only the local instance name is available.
    implicit val iToM: mutable.Map[ComponentName, ModuleName] = mutable.Map.empty

    val mx = m
      .map(onPort)
      .map(onStatement)
      .map(onName(_: String)(renames, moduleNamespace, Some(circuit)))

    // Must happen after renaming the name and ports of the module itself
    mType += (mx.name -> onType(Utils.module_type(mx)))
    mx
  }

  /** Fix any Verilog keyword collisions in a [[firrtl.ir Circuit]]
    * @param c a [[firrtl.ir Circuit]] with possible name collisions
    * @param renames a [[RenameMap]] to update. If you don't want to propagate renames, this can be ignored.
    * @return a [[firrtl.ir Circuit]] without keyword conflicts
    */
  def run(c: ir.Circuit, renames: RenameMap = RenameMap()): ir.Circuit = {
    implicit val circuitNamespace: Namespace = Namespace(c)
    implicit val scope: Option[CircuitName] = Some(CircuitName(c.main))
    val modType: ModuleType = new ModuleType()

    // Rename all modules from leafs to root in one pass while updating a shared rename map. Going from leafs to roots
    // ensures that the rename map is safe for parents to blindly consult.
    val modulesx: Map[ModuleName, Seq[ir.DefModule]] = new InstanceGraph(c).moduleOrder.reverse
      .map(onModule(renames, scope.get, modType))
      .groupBy(m => ModuleName(m.name, scope.get))

    // Reorder the renamed modules into the original circuit order.
    val modulesxx: Seq[ir.DefModule] = c.modules.flatMap{ orig =>
      val named = ModuleName(orig.name, scope.get)
      modulesx(renames.get(named).orElse(Some(Seq(named))).get.head)
    }

    // Rename the circuit if the top module was renamed
    val mainx = renames.get(ModuleName(c.main, CircuitName(c.main))) match {
      case Some(Seq(ModuleName(m, _))) =>
        renames.rename(CircuitName(c.main), CircuitName(m))
        m
      case x@ Some(_) => throw new PassException(
        s"Verilog renaming shouldn't result in multiple renames, but found '${c.main} -> $x'")
      case None =>
        c.main
    }

    // Apply all updates
    c.copy(modules = modulesxx, main = mainx)
  }

  /** Fix any Verilog keyword name collisions in a [[CircuitState]] while propagating renames
    * @param state the [[CircuitState]] with possible name collisions
    * @return a [[CircuitState]] without name collisions
    */
  def execute(state: CircuitState): CircuitState = {
    val renames = RenameMap()
    renames.setCircuit(state.circuit.main)
    state.copy(circuit = run(state.circuit, renames), renames = Some(renames))
  }
}

/** Transform that removes collisions with Verilog keywords */
class VerilogRename extends RemoveKeywordCollisions(v_keywords)
