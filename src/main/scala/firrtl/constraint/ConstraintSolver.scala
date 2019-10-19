// See LICENSE for license details.

package firrtl.constraint

import firrtl._
import firrtl.ir._
import firrtl.Utils.throwInternalError
import firrtl.annotations.ReferenceTarget

import scala.collection.mutable

/** Forwards-Backwards Constraint Solver
  *
  * Used for computing [[Width]] and [[Bound]] constraints
  *
  * Note - this is an O(N) algorithm, but requires exponential memory. We rely on aggressive early optimization
  *   of constraint expressions to (usually) get around this.
  */
class ConstraintSolver {

  /** Initial, mutable constraint list, with function to add the constraint */
  private val constraints = mutable.ArrayBuffer[Inequality]()

  /** Solved constraints */
  type ConstraintMap = mutable.HashMap[String, (Constraint, Boolean)]
  private val solvedConstraintMap = new ConstraintMap()


  /** Clear all previously recorded/solved constraints */
  def clear(): Unit = {
    constraints.clear()
    solvedConstraintMap.clear()
  }

  /** Updates internal list of inequalities with a new [[GreaterOrEqual]]
    * @param big The larger constraint, must be either known or a variable
    * @param small The smaller constraint
    */
  def addGeq(big: Constraint, small: Constraint, r1: String, r2: String): Unit = (big, small) match {
    case (IsVar(name), other: Constraint) => add(GreaterOrEqual(name, other))
    case _ => // Constraints on widths should never error, e.g. attach adds lots of unnecessary constraints
  }

  /** Updates internal list of inequalities with a new [[GreaterOrEqual]]
    * @param big The larger constraint, must be either known or a variable
    * @param small The smaller constraint
    */
  def addGeq(big: Width, small: Width, r1: String, r2: String): Unit = (big, small) match {
    case (IsVar(name), other: CalcWidth) => add(GreaterOrEqual(name, other.arg))
    case (IsVar(name), other: IsVar) => add(GreaterOrEqual(name, other))
    case (IsVar(name), other: IntWidth) => add(GreaterOrEqual(name, Implicits.width2constraint(other)))
    case _ => // Constraints on widths should never error, e.g. attach adds lots of unnecessary constraints
  }

  /** Updates internal list of inequalities with a new [[LesserOrEqual]]
    * @param small The smaller constraint, must be either known or a variable
    * @param big The larger constraint
    */
  def addLeq(small: Constraint, big: Constraint, r1: String, r2: String): Unit = (small, big) match {
    case (IsVar(name), other: Constraint) => add(LesserOrEqual(name, other))
    case _ => // Constraints on widths should never error, e.g. attach adds lots of unnecessary constraints
  }

  /** Updates internal list of inequalities with a new [[LesserOrEqual]]
    * @param small The smaller constraint, must be either known or a variable
    * @param big The larger constraint
    */
  def addLeq(small: Width, big: Width, r1: String, r2: String): Unit = (small, big) match {
    case (IsVar(name), other: CalcWidth) => add(LesserOrEqual(name, other.arg))
    case (IsVar(name), other: IsVar) => add(LesserOrEqual(name, other))
    case (IsVar(name), other: IntWidth) => add(LesserOrEqual(name, Implicits.width2constraint(other)))
    case _ => // Constraints on widths should never error, e.g. attach adds lots of unnecessary constraints
  }

  /** Returns a solved constraint, if it exists and is solved
    * @param b
    * @return
    */
  def get(b: Constraint): Option[IsKnown] = {
    val name = b match {
      case IsVar(name) => name
      case x => ""
    }
    solvedConstraintMap.get(name) match {
      case None => None
      case Some((k: IsKnown, _)) => Some(k)
      case Some(_) => None
    }
  }

  /** Returns a solved width, if it exists and is solved
    * @param b
    * @return
    */
  def get(b: Width): Option[IsKnown] = {
    val name = b match {
      case IsVar(name) => name
      case x => ""
    }
    solvedConstraintMap.get(name) match {
      case None => None
      case Some((k: IsKnown, _)) => Some(k)
      case Some(_) => None
    }
  }


  private def add(c: Inequality) = constraints += c


  /** Creates an Inequality given a variable name, constraint, and whether its >= or <=
    * @param left
    * @param right
    * @param geq
    * @return
    */
  private def genConst(left: String, right: Constraint, geq: Boolean): Inequality = geq match {
    case true => GreaterOrEqual(left, right)
    case false => LesserOrEqual(left, right)
  }

  /** For debugging, can serialize the initial constraints */
  def serializeConstraints: String = constraints.mkString("\n")

  /** For debugging, can serialize the solved constraints */
  def serializeSolutions: String = solvedConstraintMap.map{
    case (k, (v, true))  => s"$k >= ${v.serialize}"
    case (k, (v, false)) => s"$k <= ${v.serialize}"
  }.mkString("\n")



  /************* Constraint Solver Engine ****************/

  /** Merges constraints on the same variable
    *
    * Returns a new list of Inequalities with a single Inequality per variable
    *
    * For example, given:
    *   a >= 1 + b
    *   a >= 3
    *
    * Will return:
    *   a >= max(3, 1 + b)
    *
    * @param constraints
    * @return
    */
  private def mergeConstraints(constraints: Seq[Inequality]): Seq[Inequality] = {
    val mergedMap = mutable.HashMap[String, Inequality]()
    constraints.foreach {
        case c if c.geq  && mergedMap.contains(c.left) =>
          mergedMap(c.left) = genConst(c.left, IsMax(mergedMap(c.left).right, c.right), true)
        case c if !c.geq && mergedMap.contains(c.left) =>
          mergedMap(c.left) = genConst(c.left, IsMin(mergedMap(c.left).right, c.right), false)
        case c =>
          mergedMap(c.left) = c
    }
    mergedMap.values.toList
  }


  /** Attempts to substitute variables with their corresponding forward-solved constraints
    * If no corresponding constraint has been visited yet, keep variable as is
    *
    * @param forwardSolved ConstraintMap containing earlier forward-solved constraints
    * @param constraint Constraint to forward solve
    * @return Forward solved constraint
    */
  private def forwardSubstitution(forwardSolved: ConstraintMap)(constraint: Constraint): Constraint = {
    val x = constraint map forwardSubstitution(forwardSolved)
    x match {
      case isVar: IsVar => forwardSolved get isVar.name match {
        case None => isVar.asInstanceOf[Constraint]
        case Some((p, geq)) =>
          val newT = forwardSubstitution(forwardSolved)(p)
          forwardSolved(isVar.name) = (newT, geq)
          newT
      }
      case other => other
    }
  }

  /** Attempts to substitute variables with their corresponding backwards-solved constraints
    * If no corresponding constraint is solved, keep variable as is (as an unsolved constraint,
    *   which will be reported later)
    *
    * @param backwardSolved ConstraintMap containing earlier backward-solved constraints
    * @param constraint Constraint to backward solve
    * @return Backward solved constraint
    */
  private def backwardSubstitution(backwardSolved: ConstraintMap)(constraint: Constraint): Constraint = {
    constraint match {
      case isVar: IsVar => backwardSolved.get(isVar.name) match {
        case Some((p, geq)) => p
        case _ => isVar
      }
      case other => other map backwardSubstitution(backwardSolved)
    }
  }

  /** Remove solvable cycles in an inequality
    *
    * For example:
    *   a >= max(1, a)
    *
    * Can be simplified to:
    *   a >= 1
    * @param name Name of the variable on left side of inequality
    * @param geq Whether inequality is >= or <=
    * @param constraint Constraint expression
    * @return
    */
  private def removeCycle(name: String, geq: Boolean)(constraint: Constraint): Constraint =
    if(geq) removeGeqCycle(name)(constraint) else removeLeqCycle(name)(constraint)

  /** Removes solvable cycles of <= inequalities
    * @param name Name of the variable on left side of inequality
    * @param constraint Constraint expression
    * @return
    */
  private def removeLeqCycle(name: String)(constraint: Constraint): Constraint = constraint match {
    case x if greaterEqThan(name)(x) => VarCon(name)
    case isMin: IsMin => IsMin(isMin.children.filter{ c => !greaterEqThan(name)(c)})
    case x => x
  }

  /** Removes solvable cycles of >= inequalities
    * @param name Name of the variable on left side of inequality
    * @param constraint Constraint expression
    * @return
    */
  private def removeGeqCycle(name: String)(constraint: Constraint): Constraint = constraint match {
    case x if lessEqThan(name)(x) => VarCon(name)
    case isMax: IsMax => IsMax(isMax.children.filter{c => !lessEqThan(name)(c)})
    case x => x
  }

  private def greaterEqThan(name: String)(constraint: Constraint): Boolean = constraint match {
    case isMin: IsMin => isMin.children.map(greaterEqThan(name)).reduce(_ && _)
    case isAdd: IsAdd => isAdd.children match {
      case Seq(isVar: IsVar, isVal: IsKnown) if (isVar.name == name) && (isVal.value >= 0) => true
      case Seq(isVal: IsKnown, isVar: IsVar) if (isVar.name == name) && (isVal.value >= 0) => true
      case _ => false
    }
    case isMul: IsMul => isMul.children match {
      case Seq(isVar: IsVar, isVal: IsKnown) if (isVar.name == name) && (isVal.value >= 0) => true
      case Seq(isVal: IsKnown, isVar: IsVar) if (isVar.name == name) && (isVal.value >= 0) => true
      case _ => false
    }
    case isVar: IsVar if isVar.name == name => true
    case _ => false
  }

  private def lessEqThan(name: String)(constraint: Constraint): Boolean = constraint match {
    case isMax: IsMax => isMax.children.map(lessEqThan(name)).reduce(_ && _)
    case isAdd: IsAdd => isAdd.children match {
      case Seq(isVar: IsVar, isVal: IsKnown) if (isVar.name == name) && (isVal.value <= 0) => true
      case Seq(isVal: IsKnown, isVar: IsVar) if (isVar.name == name) && (isVal.value <= 0) => true
      case _ => false
    }
    case isMul: IsMul => isMul.children match {
      case Seq(isVar: IsVar, isVal: IsKnown) if (isVar.name == name) && (isVal.value <= 0) => true
      case Seq(isVal: IsKnown, isVar: IsVar) if (isVar.name == name) && (isVal.value <= 0) => true
      case _ => false
    }
    case isVar: IsVar if isVar.name == name => true
    case isNeg: IsNeg => isNeg.child match {
      case isVar: IsVar if isVar.name == name => true
      case _ => false
    }
    case _ => false
  }

  /** Whether a constraint contains the named variable
    * @param name Name of variable
    * @param constraint Constraint to check
    * @return
    */
  private def hasVar(name: String)(constraint: Constraint): Boolean = {
    var has = false
    def rec(constraint: Constraint): Constraint = {
      constraint match {
        case isVar: IsVar if isVar.name == name => has = true
        case _ =>
      }
      constraint map rec
    }
    rec(constraint)
    has
  }

  /** Returns illegal constraints, where both a >= and <= inequality are used on the same variable
    * @return
    */
  def check(): Seq[Inequality] = {
    val checkMap = new mutable.HashMap[String, Inequality]()
    constraints.foldLeft(Seq[Inequality]()) { (seq, c) =>
      checkMap.get(c.left) match {
        case None =>
          checkMap(c.left) = c
          seq ++ Nil
        case Some(x) if x.geq != c.geq => seq ++ Seq(x, c)
        case Some(x) => seq ++ Nil
      }
    }
  }

  /** Solves constraints present in collected inequalities
    *
    * Constraint solving steps:
    *   1) Assert no variable has both >= and <= inequalities (it can have multiple of the same kind of inequality)
    *   2) Merge constraints of variables having multiple inequalities
    *   3) Forward solve inequalities
    *     a. Iterate through inequalities top-to-bottom, replacing previously seen variables with corresponding
    *        constraint
    *     b. For each forward-solved inequality, attempt to remove circular constraints
    *     c. Forward-solved inequalities without circular constraints are recorded
    *   4) Backwards solve inequalities
    *     a. Iterate through successful forward-solved inequalities bottom-to-top, replacing previously seen variables
    *        with corresponding constraint
    *     b. Record solved constraints
    */
  def solve(): Unit = {
    // 1) Check if any variable has both >= and <= inequalities (which is illegal)
    val illegals = check()
    if (illegals != Nil) throwInternalError(s"Constraints cannot have both >= and <= inequalities: $illegals")

    // 2) Merge constraints
    val uniqueConstraints = mergeConstraints(constraints.toSeq)

    // 3) Forward Solve
    val forwardConstraintMap = new ConstraintMap
    val orderedVars = mutable.HashMap[Int, String]()

    var index = 0
    for (constraint <- uniqueConstraints) {
      //TODO: Risky if used improperly... need to check whether substitution from a leq to a geq is negated (always).
      val subbedRight = forwardSubstitution(forwardConstraintMap)(constraint.right)
      val name = constraint.left
      val finishedRight = removeCycle(name, constraint.geq)(subbedRight)
      if (!hasVar(name)(finishedRight)) {
        forwardConstraintMap(name) = (finishedRight, constraint.geq)
        orderedVars(index) = name
        index += 1
      }
    }

    // 4) Backwards Solve
    for (i <- (orderedVars.size - 1) to 0 by -1) {
      val name = orderedVars(i) // Should visit `orderedVars` backward
      val (forwardRight, forwardGeq) = forwardConstraintMap(name)
      val solvedRight = backwardSubstitution(solvedConstraintMap)(forwardRight)
      solvedConstraintMap(name) = (solvedRight, forwardGeq)
    }
  }
}
