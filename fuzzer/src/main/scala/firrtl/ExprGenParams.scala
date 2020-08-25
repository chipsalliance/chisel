package firrtl.fuzzer

import com.pholser.junit.quickcheck.generator.{Generator, GenerationStatus}
import com.pholser.junit.quickcheck.random.SourceOfRandomness

import firrtl.{Namespace, Utils}
import firrtl.ir._

import scala.language.higherKinds

/** A set of parameters for randomly generating [[firrtl.ir.Expression Expression]]s
  */
sealed trait ExprGenParams {

  /** The maximum levels of nested sub-expressions that may be generated
    */
  def maxDepth: Int

  /** The maximum width of any generated expression, including sub-expressions
    */
  def maxWidth: Int

  /** A mapping of expression generator to frequency
    *
    * The frequency number determines the probability that the corresponding
    * generator will be chosen. i.g. for Map(A -> 1, B -> 2, C -> B), the
    * probabilities for A, B, and C are 1/6, 2/6, and 3/6 respectively.  This
    * map must be non-empty and all frequency numbers must be greater than
    * zero.
    */
  def generators: Map[ExprGen[_ <: Expression], Int]

  /** The set of generated references that don't have a corresponding declaration
    */
  protected def unboundRefs: Set[Reference]

  /** The namespace to use for generating new [[firrtl.ir.Reference Reference]]s
    */
  protected def namespace: Namespace

  /** Returns a copy of this [[ExprGenParams]] with the maximum depth decremented
    */
  protected def decrementDepth: ExprGenParams

  /** Returns a copy of this [[ExprGenParams]] with the maximum depth incremented
    */
  protected def incrementDepth: ExprGenParams

  /** Returns a copy of this [[ExprGenParams]] with the reference added to the set of unbound references
    */
  protected def withRef(ref: Reference): ExprGenParams


  import GenMonad.syntax._

  /** Generator that generates an expression and wraps it in a Module
    *
    *  The generated references are bound to input ports and the generated
    *  expression is assigned to an output port.
    */
  private def exprMod[G[_]: GenMonad]: StateGen[ExprGenParams, G, Module] = {
    for {
      width <- StateGen.inspectG((s: ExprGenParams) => ExprGen.genWidth(1, ExprState[ExprGenParams].maxWidth(s)))
      tpe <- StateGen.liftG(GenMonad.frequency(
        2 -> UIntType(width),
        2 -> SIntType(width),
        1 -> Utils.BoolType
      ))
      expr <- ExprState[ExprGenParams].exprGen(tpe)
      outputPortRef <- tpe match {
        case UIntType(IntWidth(width)) if width == BigInt(1) => ExprGen.ReferenceGen.boolUIntGen[ExprGenParams, G].get
        case UIntType(IntWidth(width)) => ExprGen.ReferenceGen.uintGen[ExprGenParams, G].get(width)
        case SIntType(IntWidth(width)) if width == BigInt(1) => ExprGen.ReferenceGen.boolSIntGen[ExprGenParams, G].get
        case SIntType(IntWidth(width)) => ExprGen.ReferenceGen.sintGen[ExprGenParams, G].get(width)
      }
      unboundRefs <- StateGen.inspect { ExprState[ExprGenParams].unboundRefs }
    } yield {
      val outputPort = Port(
        NoInfo,
        outputPortRef.name,
        Output,
        outputPortRef.tpe
      )
      Module(
        NoInfo,
        "foo",
        unboundRefs.flatMap {
          case ref if ref.name == outputPortRef.name => None
          case ref => Some(Port(NoInfo, ref.name, Input, ref.tpe))
        }.toSeq.sortBy(_.name) :+ outputPort,
        Connect(NoInfo, outputPortRef, expr)
      )
    }
  }

  /** Runs the expression generator once and returns the generated expression
    * wrapped in a Module and Circuit
    */
  def generateSingleExprCircuit[G[_]: GenMonad](): Circuit = {
    exprMod.map { m =>
      Circuit(NoInfo, Seq(m), m.name)
    }.run(this).map(_._2).generate()
  }
}

object ExprGenParams {

  val defaultGenerators: Map[ExprGen[_ <: Expression], Int] = {
    import ExprGen._
    Map(
      AddDoPrimGen -> 1,
      SubDoPrimGen -> 1,
      MulDoPrimGen -> 1,
      DivDoPrimGen -> 1,
      LtDoPrimGen -> 1,
      LeqDoPrimGen -> 1,
      GtDoPrimGen -> 1,
      GeqDoPrimGen -> 1,
      EqDoPrimGen -> 1,
      NeqDoPrimGen -> 1,
      PadDoPrimGen -> 1,
      ShlDoPrimGen -> 1,
      ShrDoPrimGen -> 1,
      DshlDoPrimGen -> 1,
      CvtDoPrimGen -> 1,
      NegDoPrimGen -> 1,
      NotDoPrimGen -> 1,
      AndDoPrimGen -> 1,
      OrDoPrimGen -> 1,
      XorDoPrimGen -> 1,
      AndrDoPrimGen -> 1,
      OrrDoPrimGen -> 1,
      XorrDoPrimGen -> 1,
      CatDoPrimGen -> 1,
      BitsDoPrimGen -> 1,
      HeadDoPrimGen -> 1,
      TailDoPrimGen -> 1,
      AsUIntDoPrimGen -> 1,
      AsSIntDoPrimGen -> 1,
      MuxGen -> 1
    )
  }

  private case class ExprGenParamsImp(
    maxDepth: Int,
    maxWidth: Int,
    generators: Map[ExprGen[_ <: Expression], Int],
    protected val unboundRefs: Set[Reference],
    protected val namespace: Namespace) extends ExprGenParams {

    protected def decrementDepth: ExprGenParams = this.copy(maxDepth = maxDepth - 1)
    protected def incrementDepth: ExprGenParams = this.copy(maxDepth = maxDepth + 1)
    protected def withRef(ref: Reference): ExprGenParams = this.copy(unboundRefs = unboundRefs + ref)
  }

  /** Constructs an [[ExprGenParams]] with the given parameters
    */
  def apply(
    maxDepth: Int,
    maxWidth: Int,
    generators: Map[ExprGen[_ <: Expression], Int]
  ): ExprGenParams = {
    require(maxWidth > 0, "maxWidth must be greater than zero")
    ExprGenParamsImp(
      maxDepth,
      maxWidth,
      generators,
      Set.empty,
      Namespace()
    )
  }

  import GenMonad.syntax._

  private def combineExprGens[S: ExprState, G[_]: GenMonad](
    exprGenerators: Seq[(Int, ExprGen[_ <: Expression])]
  )(tpe: Type): StateGen[S, G, Option[Expression]] = {
    val boolUIntStateGens = exprGenerators.flatMap {
      case (freq, gen) => gen.boolUIntGen[S, G].map(freq -> _.widen[Expression])
    }
    val uintStateGenFns = exprGenerators.flatMap {
      case (freq, gen) => gen.uintGen[S, G].map { fn =>
        (width: BigInt) => freq -> fn(width).widen[Expression]
      }
    }
    val boolSIntStateGens = exprGenerators.flatMap {
      case (freq, gen) => gen.boolSIntGen[S, G].map(freq -> _.widen[Expression])
    }
    val sintStateGenFns = exprGenerators.flatMap {
      case (freq, gen) => gen.sintGen[S, G].map { fn =>
        (width: BigInt) => freq -> fn(width).widen[Expression]
      }
    }
    val stateGens: Seq[(Int, StateGen[S, G, Expression])] = tpe match {
      case Utils.BoolType => boolUIntStateGens
      case UIntType(IntWidth(width)) => uintStateGenFns.map(_(width))
      case SIntType(IntWidth(width)) if width.toInt == 1 => boolSIntStateGens
      case SIntType(IntWidth(width)) => sintStateGenFns.map(_(width))
    }
    StateGen { (s: S) =>
      if (stateGens.isEmpty) {
        GenMonad[G].const(s -> None)
      } else if (stateGens.size == 1) {
        stateGens(0)._2.run(s).map { case (ss, expr) => ss -> Some(expr) }
      } else {
        GenMonad.frequency(stateGens: _*).flatMap { stateGen =>
          stateGen.run(s).map { case (ss, expr) => ss -> Some(expr) }
        }
      }
    }
  }

  implicit val exprGenParamsExprStateInstance: ExprState[ExprGenParams] = new ExprState[ExprGenParams] {
    def withRef[G[_]: GenMonad](ref: Reference): StateGen[ExprGenParams, G, Reference] = {
      StateGen { (s: ExprGenParams) =>
        val refx = ref.copy(name = s.namespace.newName(ref.name))
        GenMonad[G].const(s.withRef(refx) -> refx)
      }
    }
    def unboundRefs(s: ExprGenParams): Set[Reference] = s.unboundRefs
    def maxWidth(s: ExprGenParams): Int = s.maxWidth

    def exprGen[G[_]: GenMonad](tpe: Type): StateGen[ExprGenParams, G, Expression] = {
      StateGen { (s: ExprGenParams) =>

        val leafGen: Type => StateGen[ExprGenParams, G, Expression] = (tpe: Type) => combineExprGens(Seq(
          1 -> ExprGen.LiteralGen,
          1 -> ExprGen.ReferenceGen
        ))(tpe).map(e => e.get) // should be safe because leaf generators are defined for all types

        val branchGen: Type => StateGen[ExprGenParams, G, Expression] = (tpe: Type) => {
          val gens = s.generators.toSeq.map { case (gen, freq) => (freq, gen) }
          combineExprGens(gens)(tpe).flatMap {
            case None => leafGen(tpe)
            case Some(e) => StateGen.pure(e)
          }
        }

        if (s.maxDepth > 0) {
          // for recrusive generators, decrement maxDepth before recursing then increment when finished
          GenMonad.frequency(
            5 -> (branchGen(_)),
            1 -> (leafGen(_))
          ).flatMap(_(tpe).run(s.decrementDepth).map {
            case (ss, e) => ss.incrementDepth -> e
          })
        } else {
          leafGen(tpe).run(s)
        }
      }
    }
  }
}

abstract class SingleExpressionCircuitGenerator(val params: ExprGenParams) extends Generator[Circuit](classOf[Circuit]) {
  override def generate(random: SourceOfRandomness, status: GenerationStatus): Circuit = {
    implicit val r = random
    params.generateSingleExprCircuit[SourceOfRandomnessGen]()
  }
}
