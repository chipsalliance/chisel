// See LICENSE for license details.

package firrtl.transforms

import firrtl.{CircuitState, Namespace, PrimOps, Transform, UnknownForm, Utils, WRef}
import firrtl.ir._
import firrtl.Mappers._
import firrtl.options.{Dependency, PreservesAll}
import firrtl.PrimOps.{Add, AsSInt, Sub, Tail}
import firrtl.stage.Forms

import scala.collection.mutable

object FixAddingNegativeLiterals {

  /** Returns the maximum negative number represented by given width
    * @param width width of the negative number
    * @return maximum negative number
    */
  def minNegValue(width: BigInt): BigInt = -(1 << (width.toInt - 1))

  /** Updates the type of the DoPrim from its arguments (e.g. if is UnknownType)
    * @param d input DoPrim
    * @return updated DoPrim with calculated type
    */
  def setType(d: DoPrim): DoPrim = {
    PrimOps.set_primop_type(d)
  }

  /** Returns a module with fixed additions of negative literals
    * @param m input module
    * @return updated module
    */
  def fixupModule(m: DefModule): DefModule = {
    val namespace = Namespace(m)
    m map fixupStatement(namespace)
  }

  /** Returns a statement with fixed additions of negative literals
    * @param namespace object to enabling creating unique names
    * @param s input statement
    * @return updated statement
    */
  def fixupStatement(namespace: Namespace)(s: Statement): Statement = {
    val stmtBuffer = mutable.ListBuffer[Statement]()
    val ret = s map fixupStatement(namespace) map fixupOnExpr(Utils.get_info(s), namespace, stmtBuffer)
    if(stmtBuffer.isEmpty) {
      ret
    } else {
      stmtBuffer += ret
      Block(stmtBuffer.toList)
    }
  }

  /** Returns a statement with fixed additions of negative literals
    * @param info Info of statement containing this expression
    * @param namespace object to enabling creating unique names
    * @param e expression to fixup
    * @return generated statements and the fixed expression
    */
  def fixupExpression(info: Info, namespace: Namespace)
                     (e: Expression): (Seq[Statement], Expression) = {
    val stmtBuffer = mutable.ListBuffer[Statement]()
    val retExpr = fixupOnExpr(info, namespace, stmtBuffer)(e)
    (stmtBuffer.toList, retExpr)
  }

  /** Returns a statement with fixed additions of negative literals
    * @param info Info of statement containing this expression
    * @param namespace object to enabling creating unique names
    * @param stmtBuffer mutable buffer of statements - append to this for it to be inlined in the module
    * @param e expression to fixup
    * @return fixed expression
    */
  private def fixupOnExpr(info: Info, namespace: Namespace, stmtBuffer: mutable.ListBuffer[Statement])
                         (e: Expression): Expression = {

    // Helper function to create the subtraction expression
    def fixupAdd(expr: Expression, litValue: BigInt, litWidth: BigInt): DoPrim = {
      if(litValue == minNegValue(litWidth)) {
        val posLiteral = SIntLiteral(-litValue)
        assert(posLiteral.width.asInstanceOf[IntWidth].width - 1 == litWidth)
        val sub = DefNode(info, namespace.newTemp, setType(DoPrim(Sub, Seq(expr, posLiteral), Nil, UnknownType)))
        val tail = DefNode(info, namespace.newTemp, setType(DoPrim(Tail, Seq(WRef(sub)), Seq(1), UnknownType)))
        stmtBuffer += sub
        stmtBuffer += tail
        setType(DoPrim(AsSInt, Seq(WRef(tail)), Nil, UnknownType))
      } else {
        val posLiteral = SIntLiteral(-litValue)
        setType(DoPrim(Sub, Seq(expr, SIntLiteral(-litValue, IntWidth(litWidth))), Nil, UnknownType))
      }
    }

    e map fixupOnExpr(info, namespace, stmtBuffer) match {
      case DoPrim(Add, Seq(arg, lit@SIntLiteral(value, w@IntWidth(width))), Nil, t: SIntType) if value < 0 =>
        fixupAdd(arg, value, width)
      case DoPrim(Add, Seq(lit@SIntLiteral(value, w@IntWidth(width)), arg), Nil, t: SIntType) if value < 0 =>
        fixupAdd(arg, value, width)
      case other => other
    }
  }
}

/** Replaces adding a negative literal with subtracting that literal
  *
  * Verilator has a lint warning if a literal is negated in an expression, because it adds a bit to
  * the literal and thus not all expressions in the add are the same. This is fixed here when we directly
  * subtract the literal instead.
  */
class FixAddingNegativeLiterals extends Transform with PreservesAll[Transform] {
  def inputForm = UnknownForm
  def outputForm = UnknownForm

  override val prerequisites = Forms.LowFormMinimumOptimized :+ Dependency[BlackBoxSourceHelper]

  override val optionalPrerequisites = firrtl.stage.Forms.LowFormOptimized

  override val dependents = Seq.empty

  def execute(state: CircuitState): CircuitState = {
    val modulesx = state.circuit.modules.map(FixAddingNegativeLiterals.fixupModule)
    state.copy(circuit = state.circuit.copy(modules = modulesx))
  }
}
