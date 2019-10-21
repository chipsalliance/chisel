package firrtlTests.execution

import firrtl._
import firrtl.ir._

object DUTRules {
  val dutName = "dut"
  val clock = Reference("clock", ClockType)
  val reset = Reference("reset", Utils.BoolType)
  val counter = Reference("step", UnknownType)

  // Need a flat name for the register that latches poke values
  val illegal = raw"[\[\]\.]".r
  val pokeRegSuffix = "_poke"
  def pokeRegName(e: Expression) = illegal.replaceAllIn(e.serialize, "_") + pokeRegSuffix

  // Naming patterns are static, so DUT has to be checked for proper form + collisions
  def hasNameConflicts(c: Circuit): Boolean = {
    val top = c.modules.find(_.name == c.main).get
    val names = Namespace(top).cloneUnderlying
    names.contains(counter.name) || names.exists(_.contains(pokeRegSuffix))
  }
}

object ExecutionTestHelper {
  val counterType = UIntType(IntWidth(32))
  def apply(body: String): ExecutionTestHelper = {
    // Parse input and check that it complies with test syntax rules
    val c = ParseStatement.makeDUT(body)
    require(!DUTRules.hasNameConflicts(c), "Avoid using 'step' or 'poke' in DUT component names")

    // Generate test step counter, create ExecutionTestHelper that represents initial test state
    val cnt = DefRegister(NoInfo, DUTRules.counter.name, counterType, DUTRules.clock, DUTRules.reset, Utils.zero)
    val inc = Connect(NoInfo, DUTRules.counter, DoPrim(PrimOps.Add, Seq(DUTRules.counter, UIntLiteral(1)), Nil, UnknownType))
    ExecutionTestHelper(c, Seq(cnt, inc), Map.empty[Expression, Expression], Nil, Nil)
  }
}

case class ExecutionTestHelper(
  dut: Circuit,
  setup: Seq[Statement],
  pokeRegs: Map[Expression, Expression],
  completedSteps: Seq[Conditionally],
  activeStep: Seq[Statement]
) {

  def step(n: Int): ExecutionTestHelper = {
    require(n > 0, "Step length must be positive")
    (0 until n).foldLeft(this) { case (eth, int) => eth.next }
  }

  def poke(expString: String, value: Literal): ExecutionTestHelper = {
    val pokeExp = ParseExpression(expString)
    val pokeable = ensurePokeable(pokeExp)
    pokeable.addStatements(
      Connect(NoInfo, pokeExp, value),
      Connect(NoInfo, pokeable.pokeRegs(pokeExp), value))
  }

  def invalidate(expString: String): ExecutionTestHelper = {
    addStatements(IsInvalid(NoInfo, ParseExpression(expString)))
  }

  def expect(expString: String, value: Literal): ExecutionTestHelper = {
    val peekExp = ParseExpression(expString)
    val neq = DoPrim(PrimOps.Neq, Seq(peekExp, value), Nil, Utils.BoolType)
    addStatements(Stop(NoInfo, 1, DUTRules.clock, neq))
  }

  def finish(): ExecutionTestHelper = {
    addStatements(Stop(NoInfo, 0, DUTRules.clock, Utils.one)).next
  }

  // Private helper methods

  private def t = completedSteps.length

  private def addStatements(stmts: Statement*) = copy(activeStep = activeStep ++ stmts)

  private def next: ExecutionTestHelper = {
    val count = Reference(DUTRules.counter.name, DUTRules.counter.tpe)
    val ifStep = DoPrim(PrimOps.Eq, Seq(count, UIntLiteral(t)), Nil, Utils.BoolType)
    val onThisStep = Conditionally(NoInfo, ifStep, Block(activeStep), EmptyStmt)
    copy(completedSteps = completedSteps :+ onThisStep, activeStep = Nil)
  }

  private def top: Module = {
    dut.modules.collectFirst({ case m: Module if m.name == dut.main  => m }).get
  }

  private[execution] def emit: Circuit = {
    val finished = finish()
    val modulesX = dut.modules.collect {
      case m: Module if m.name == dut.main =>
        m.copy(body = Block(m.body +: (setup ++ finished.completedSteps)))
      case m => m
    }
    dut.copy(modules = modulesX)
  }

  private def ensurePokeable(pokeExp: Expression): ExecutionTestHelper = {
    if (pokeRegs.contains(pokeExp)) {
      this
    } else {
      val pName = DUTRules.pokeRegName(pokeExp)
      val pRef = Reference(pName, UnknownType)
      val pReg = DefRegister(NoInfo, pName, UIntType(UnknownWidth), DUTRules.clock, Utils.zero, pRef)
      val defaultConn = Connect(NoInfo, pokeExp, pRef)
      copy(setup = setup ++ Seq(pReg, defaultConn), pokeRegs = pokeRegs + (pokeExp -> pRef))
    }
  }
}
