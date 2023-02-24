// SPDX-License-Identifier: Apache-2.0

package firrtlTests.options

import firrtl.AnnotationSeq
import firrtl.options.{Dependency, DependencyManagerException, Phase, PhaseManager}

import java.io.{File, PrintWriter}

import sys.process._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

trait IdentityPhase extends Phase {
  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations
}

trait PreservesAll { this: Phase =>
  override def invalidates(phase: Phase) = false
}

/** Default [[Phase]] that has no prerequisites and invalidates nothing */
class A extends IdentityPhase {

  override def invalidates(phase: Phase): Boolean = false
}

/** [[Phase]] that requires [[A]] and invalidates nothing */
class B extends IdentityPhase {
  override def prerequisites = Seq(Dependency[A])
  override def invalidates(phase: Phase): Boolean = false
}

/** [[Phase]] that requires [[B]] and invalidates nothing */
class C extends IdentityPhase {
  override def prerequisites = Seq(Dependency[A])
  override def invalidates(phase: Phase): Boolean = false
}

/** [[Phase]] that requires [[A]] and invalidates [[A]] */
class D extends IdentityPhase {
  override def prerequisites = Seq(Dependency[A])
  override def invalidates(phase: Phase): Boolean = phase match {
    case _: A => true
    case _ => false
  }
}

/** [[Phase]] that requires [[B]] and invalidates nothing */
class E extends IdentityPhase {
  override def prerequisites = Seq(Dependency[B])
  override def invalidates(phase: Phase): Boolean = false
}

/** [[Phase]] that requires [[B]] and [[C]] and invalidates [[E]] */
class F extends IdentityPhase {
  override def prerequisites = Seq(Dependency[B], Dependency[C])
  override def invalidates(phase: Phase): Boolean = phase match {
    case _: E => true
    case _ => false
  }
}

/** [[Phase]] that requires [[C]] and invalidates [[F]] */
class G extends IdentityPhase {
  override def prerequisites = Seq(Dependency[C])
  override def invalidates(phase: Phase): Boolean = phase match {
    case _: F => true
    case _ => false
  }
}

class CyclicA extends IdentityPhase with PreservesAll {
  override def prerequisites = Seq(Dependency[CyclicB])
}

class CyclicB extends IdentityPhase with PreservesAll {
  override def prerequisites = Seq(Dependency[CyclicA])
}

class CyclicC extends IdentityPhase {
  override def invalidates(a: Phase): Boolean = a match {
    case _: CyclicD => true
    case _ => false
  }
}

class CyclicD extends IdentityPhase {
  override def invalidates(a: Phase): Boolean = a match {
    case _: CyclicC => true
    case _ => false
  }
}

object ComplicatedFixture {

  class A extends IdentityPhase {
    override def invalidates(phase: Phase): Boolean = false
  }
  class B extends IdentityPhase {
    override def prerequisites = Seq(Dependency[A])
    override def invalidates(phase: Phase): Boolean = false
  }
  class C extends IdentityPhase {
    override def prerequisites = Seq(Dependency[A])
    override def invalidates(phase: Phase): Boolean = phase match {
      case _: B => true
      case _ => false
    }
  }
  class D extends IdentityPhase {
    override def prerequisites = Seq(Dependency[B])
    override def invalidates(phase: Phase): Boolean = phase match {
      case _: C | _: E => true
      case _ => false
    }
  }
  class E extends IdentityPhase {
    override def prerequisites = Seq(Dependency[B])
    override def invalidates(phase: Phase): Boolean = false
  }

}

object RepeatedAnalysisFixture {

  trait InvalidatesAnalysis extends IdentityPhase {
    override def invalidates(phase: Phase): Boolean = phase match {
      case _: Analysis => true
      case _ => false
    }
  }

  class Analysis extends IdentityPhase {
    override def invalidates(phase: Phase): Boolean = false
  }
  class A extends InvalidatesAnalysis {
    override def prerequisites = Seq(Dependency[Analysis])
  }
  class B extends InvalidatesAnalysis {
    override def prerequisites = Seq(Dependency[A], Dependency[Analysis])
  }
  class C extends InvalidatesAnalysis {
    override def prerequisites = Seq(Dependency[B], Dependency[Analysis])
  }

}

object InvertedAnalysisFixture {

  class Analysis extends IdentityPhase {
    override def invalidates(phase: Phase): Boolean = false
  }
  class A extends IdentityPhase {
    override def prerequisites = Seq(Dependency[Analysis])
    override def invalidates(phase: Phase): Boolean = phase match {
      case _: Analysis => true
      case _ => false
    }
  }
  class B extends IdentityPhase {
    override def prerequisites = Seq(Dependency[Analysis])
    override def invalidates(phase: Phase): Boolean = phase match {
      case _: Analysis | _: A => true
      case _ => false
    }
  }
  class C extends IdentityPhase {
    override def prerequisites = Seq(Dependency[Analysis])
    override def invalidates(phase: Phase): Boolean = phase match {
      case _: Analysis | _: B => true
      case _ => false
    }
  }

}

object OptionalPrerequisitesOfFixture {

  class First extends IdentityPhase {
    override def invalidates(phase: Phase): Boolean = false
  }

  class Second extends IdentityPhase {
    override def prerequisites = Seq(Dependency[First])
    override def invalidates(phase: Phase): Boolean = false
  }

  /* This models a situation where a user has a custom Phase that they need to run before some other Phase. This is an
   * abstract example of writing a Transform that cleans up combinational loops. This needs to run before combinational
   * loop detection.
   */
  class Custom extends IdentityPhase {
    override def prerequisites = Seq(Dependency[First])
    override def optionalPrerequisiteOf = Seq(Dependency[Second])
    override def invalidates(phase: Phase): Boolean = false
  }

}

object ChainedInvalidationFixture {

  class A extends IdentityPhase {
    override def invalidates(phase: Phase): Boolean = phase match {
      case _: B => true
      case _ => false
    }
  }
  class B extends IdentityPhase {
    override def invalidates(phase: Phase): Boolean = phase match {
      case _: C => true
      case _ => false
    }
  }
  class C extends IdentityPhase {
    override def invalidates(phase: Phase): Boolean = phase match {
      case _: D => true
      case _ => false
    }
  }
  class D extends IdentityPhase {
    override def invalidates(phase: Phase): Boolean = false
  }
  class E extends IdentityPhase {
    override def prerequisites = Seq(Dependency[A], Dependency[B], Dependency[C], Dependency[D])
    override def invalidates(phase: Phase): Boolean = false
  }

}

object UnrelatedFixture {

  trait InvalidatesB8Dep { this: Phase =>
    override def invalidates(a: Phase) = a match {
      case _: B8Dep => true
      case _ => false
    }
  }

  class B0 extends IdentityPhase with InvalidatesB8Dep
  class B1 extends IdentityPhase with PreservesAll
  class B2 extends IdentityPhase with PreservesAll
  class B3 extends IdentityPhase with PreservesAll
  class B4 extends IdentityPhase with PreservesAll
  class B5 extends IdentityPhase with PreservesAll
  class B6 extends IdentityPhase with PreservesAll
  class B7 extends IdentityPhase with PreservesAll

  class B8 extends IdentityPhase with PreservesAll
  class B9 extends IdentityPhase with PreservesAll
  class B10 extends IdentityPhase with PreservesAll
  class B11 extends IdentityPhase with PreservesAll
  class B12 extends IdentityPhase with PreservesAll
  class B13 extends IdentityPhase with PreservesAll
  class B14 extends IdentityPhase with PreservesAll
  class B15 extends IdentityPhase with PreservesAll

  class B6Sub extends B6 {
    override def prerequisites = Seq(Dependency[B6])
    override def optionalPrerequisiteOf = Seq(Dependency[B7])
  }

  class B6_0 extends B6Sub
  class B6_1 extends B6Sub
  class B6_2 extends B6Sub
  class B6_3 extends B6Sub
  class B6_4 extends B6Sub
  class B6_5 extends B6Sub
  class B6_6 extends B6Sub
  class B6_7 extends B6Sub
  class B6_8 extends B6Sub
  class B6_9 extends B6Sub
  class B6_10 extends B6Sub
  class B6_11 extends B6Sub
  class B6_12 extends B6Sub
  class B6_13 extends B6Sub
  class B6_14 extends B6Sub
  class B6_15 extends B6Sub

  class B8Dep extends B8 {
    override def optionalPrerequisiteOf = Seq(Dependency[B8])
  }

  class B8_0 extends B8Dep
  class B8_1 extends B8Dep
  class B8_2 extends B8Dep
  class B8_3 extends B8Dep
  class B8_4 extends B8Dep
  class B8_5 extends B8Dep
  class B8_6 extends B8Dep
  class B8_7 extends B8Dep
  class B8_8 extends B8Dep
  class B8_9 extends B8Dep
  class B8_10 extends B8Dep
  class B8_11 extends B8Dep
  class B8_12 extends B8Dep
  class B8_13 extends B8Dep
  class B8_14 extends B8Dep
  class B8_15 extends B8Dep

}

object CustomAfterOptimizationFixture {

  class Root extends IdentityPhase with PreservesAll

  class OptMinimum extends IdentityPhase with PreservesAll {
    override def prerequisites = Seq(Dependency[Root])
    override def optionalPrerequisiteOf = Seq(Dependency[AfterOpt])
  }

  class OptFull extends IdentityPhase with PreservesAll {
    override def prerequisites = Seq(Dependency[Root], Dependency[OptMinimum])
    override def optionalPrerequisiteOf = Seq(Dependency[AfterOpt])
  }

  class AfterOpt extends IdentityPhase with PreservesAll

  class DoneMinimum extends IdentityPhase with PreservesAll {
    override def prerequisites = Seq(Dependency[OptMinimum])
  }

  class DoneFull extends IdentityPhase with PreservesAll {
    override def prerequisites = Seq(Dependency[OptFull])
  }

  class Custom extends IdentityPhase with PreservesAll {
    override def prerequisites = Seq(Dependency[Root], Dependency[AfterOpt])
    override def optionalPrerequisiteOf = Seq(Dependency[DoneMinimum], Dependency[DoneFull])
  }

}

object OptionalPrerequisitesFixture {

  class Root extends IdentityPhase

  class OptMinimum extends IdentityPhase with PreservesAll {
    override def prerequisites = Seq(Dependency[Root])
  }

  class OptFull extends IdentityPhase with PreservesAll {
    override def prerequisites = Seq(Dependency[Root], Dependency[OptMinimum])
  }

  class DoneMinimum extends IdentityPhase with PreservesAll {
    override def prerequisites = Seq(Dependency[OptMinimum])
  }

  class DoneFull extends IdentityPhase with PreservesAll {
    override def prerequisites = Seq(Dependency[OptFull])
  }

  class Custom extends IdentityPhase with PreservesAll {
    override def prerequisites = Seq(Dependency[Root])
    override def optionalPrerequisites = Seq(Dependency[OptMinimum], Dependency[OptFull])
    override def optionalPrerequisiteOf = Seq(Dependency[DoneMinimum], Dependency[DoneFull])
  }

}

object OrderingFixture {

  class A extends IdentityPhase with PreservesAll

  class B extends IdentityPhase {
    override def invalidates(phase: Phase): Boolean = phase match {
      case _: A => true
      case _ => false
    }
  }

  class C extends IdentityPhase {
    override def prerequisites = Seq(Dependency[A], Dependency[B])
    override def invalidates(phase: Phase): Boolean = phase match {
      case _: B => true
      case _ => false
    }
  }

  class Cx extends C {
    override def prerequisites = Seq(Dependency[B], Dependency[A])
  }

}

class PhaseManagerSpec extends AnyFlatSpec with Matchers {

  def writeGraphviz(pm: PhaseManager, dir: String): Unit = {

    /** Convert a Graphviz file to PNG using */
    def maybeToPng(f: File): Unit = try {
      s"dot -Tpng -O ${f}".!
    } catch {
      case _: java.io.IOException =>
    }

    val d = new File(dir)
    d.mkdirs()

    {
      val f = new File(d, "dependencyGraph.dot")
      val w = new PrintWriter(f)
      w.write(pm.dependenciesToGraphviz)
      w.close
      maybeToPng(f)
    }

    {
      val f = new File(d, "transformOrder.dot")
      val w = new PrintWriter(new File(d, "transformOrder.dot"))
      try {
        info("transform order:\n" + pm.prettyPrint("    "))
        w.write(pm.transformOrderToGraphviz())
        w.close
        maybeToPng(f)
      } catch {
        case _: DependencyManagerException =>
      }
    }

  }

  behavior.of(this.getClass.getName)

  it should "do nothing if all targets are reached" in {
    val targets = Seq(Dependency[A], Dependency[B], Dependency[C], Dependency[D])
    val pm = new PhaseManager(targets, targets)

    writeGraphviz(pm, "test_run_dir/PhaseManagerSpec/DoNothing")

    pm.flattenedTransformOrder should be(empty)
  }

  it should "handle a simple dependency" in {
    val targets = Seq(Dependency[B])
    val order = Seq(classOf[A], classOf[B])
    val pm = new PhaseManager(targets)

    writeGraphviz(pm, "test_run_dir/PhaseManagerSpec/SimpleDependency")

    pm.flattenedTransformOrder.map(_.getClass) should be(order)
  }

  it should "handle a simple dependency with an invalidation" in {
    val targets = Seq(Dependency[A], Dependency[B], Dependency[C], Dependency[D])
    val order = Seq(classOf[A], classOf[D], classOf[A], classOf[B], classOf[C])
    val pm = new PhaseManager(targets)

    writeGraphviz(pm, "test_run_dir/PhaseManagerSpec/OneInvalidate")

    pm.flattenedTransformOrder.map(_.getClass) should be(order)
  }

  it should "handle a dependency with two invalidates optimally" in {
    val targets = Seq(Dependency[A], Dependency[B], Dependency[C], Dependency[E], Dependency[F], Dependency[G])
    val pm = new PhaseManager(targets)

    writeGraphviz(pm, "test_run_dir/PhaseManagerSpec/TwoInvalidates")

    pm.flattenedTransformOrder.size should be(targets.size)
  }

  it should "throw an exception for cyclic prerequisites" in {
    val targets = Seq(Dependency[CyclicA], Dependency[CyclicB])
    val pm = new PhaseManager(targets)

    writeGraphviz(pm, "test_run_dir/PhaseManagerSpec/CyclicPrerequisites")

    intercept[DependencyManagerException] { pm.flattenedTransformOrder }.getMessage should startWith(
      "No transform ordering possible"
    )
  }

  it should "throw an exception for cyclic invalidates" in {
    val targets = Seq(Dependency[CyclicC], Dependency[CyclicD])
    val pm = new PhaseManager(targets)

    writeGraphviz(pm, "test_run_dir/PhaseManagerSpec/CyclicInvalidates")

    intercept[DependencyManagerException] { pm.flattenedTransformOrder }.getMessage should startWith(
      "No transform ordering possible"
    )
  }

  it should "handle a complicated graph" in {
    val f = ComplicatedFixture
    val targets = Seq(Dependency[f.A], Dependency[f.B], Dependency[f.C], Dependency[f.D], Dependency[f.E])
    val pm = new PhaseManager(targets)

    writeGraphviz(pm, "test_run_dir/PhaseManagerSpec/Complicated")

    info("only one phase was recomputed")
    pm.flattenedTransformOrder.size should be(targets.size + 1)
  }

  it should "handle repeated recomputed analyses" in {
    val f = RepeatedAnalysisFixture
    val targets = Seq(Dependency[f.A], Dependency[f.B], Dependency[f.C])
    val order =
      Seq(classOf[f.Analysis], classOf[f.A], classOf[f.Analysis], classOf[f.B], classOf[f.Analysis], classOf[f.C])
    val pm = new PhaseManager(targets)

    writeGraphviz(pm, "test_run_dir/PhaseManagerSpec/RepeatedAnalysis")

    pm.flattenedTransformOrder.map(_.getClass) should be(order)
  }

  it should "handle inverted repeated recomputed analyses" in {
    val f = InvertedAnalysisFixture
    val targets = Seq(Dependency[f.A], Dependency[f.B], Dependency[f.C])
    val order =
      Seq(classOf[f.Analysis], classOf[f.C], classOf[f.Analysis], classOf[f.B], classOf[f.Analysis], classOf[f.A])
    val pm = new PhaseManager(targets)

    writeGraphviz(pm, "test_run_dir/PhaseManagerSpec/InvertedRepeatedAnalysis")

    pm.flattenedTransformOrder.map(_.getClass) should be(order)
  }

  /** This test shows how the optionalPrerequisiteOf member can be used to run one transform before another. */
  it should "handle a custom Phase with an optionalPrerequisiteOf" in {
    val f = OptionalPrerequisitesOfFixture

    info("without the custom transform it runs: First -> Second")
    val pm = new PhaseManager(Seq(Dependency[f.Second]))
    val orderNoCustom = Seq(classOf[f.First], classOf[f.Second])
    pm.flattenedTransformOrder.map(_.getClass) should be(orderNoCustom)

    info("with the custom transform it runs:    First -> Custom -> Second")
    val pmCustom = new PhaseManager(Seq(Dependency[f.Custom], Dependency[f.Second]))
    val orderCustom = Seq(classOf[f.First], classOf[f.Custom], classOf[f.Second])

    writeGraphviz(pmCustom, "test_run_dir/PhaseManagerSpec/SingleDependent")

    pmCustom.flattenedTransformOrder.map(_.getClass) should be(orderCustom)
  }

  it should "handle chained invalidation" in {
    val f = ChainedInvalidationFixture

    val targets = Seq(Dependency[f.A], Dependency[f.E])
    val current = Seq(Dependency[f.B], Dependency[f.C], Dependency[f.D])

    val pm = new PhaseManager(targets, current)
    val order = Seq(classOf[f.A], classOf[f.B], classOf[f.C], classOf[f.D], classOf[f.E])

    writeGraphviz(pm, "test_run_dir/PhaseManagerSpec/ChainedInvalidate")

    pm.flattenedTransformOrder.map(_.getClass) should be(order)
  }

  it should "maintain the order of input targets" in {
    val f = UnrelatedFixture

    /** A bunch of unrelated Phases. This ensures that these run in the order in which they are specified. */
    val targets =
      Seq(
        Dependency[f.B0],
        Dependency[f.B1],
        Dependency[f.B2],
        Dependency[f.B3],
        Dependency[f.B4],
        Dependency[f.B5],
        Dependency[f.B6],
        Dependency[f.B7],
        Dependency[f.B8],
        Dependency[f.B9],
        Dependency[f.B10],
        Dependency[f.B11],
        Dependency[f.B12],
        Dependency[f.B13],
        Dependency[f.B14],
        Dependency[f.B15]
      )

    /** A sequence of custom transforms that should all run after B6 and before B7. This exercises correct ordering of the
      * prerequisiteGraph and optionalPrerequisiteOfGraph.
      */
    val prerequisiteTargets =
      Seq(
        Dependency[f.B6_0],
        Dependency[f.B6_1],
        Dependency[f.B6_2],
        Dependency[f.B6_3],
        Dependency[f.B6_4],
        Dependency[f.B6_5],
        Dependency[f.B6_6],
        Dependency[f.B6_7],
        Dependency[f.B6_8],
        Dependency[f.B6_9],
        Dependency[f.B6_10],
        Dependency[f.B6_11],
        Dependency[f.B6_12],
        Dependency[f.B6_13],
        Dependency[f.B6_14],
        Dependency[f.B6_15]
      )

    /** A sequence of transforms that are invalidated by B0 and only define optionalPrerequisiteOf on B8. This exercises
      * the ordering defined by "otherPrerequisites".
      */
    val current =
      Seq(
        Dependency[f.B8_0],
        Dependency[f.B8_1],
        Dependency[f.B8_2],
        Dependency[f.B8_3],
        Dependency[f.B8_4],
        Dependency[f.B8_5],
        Dependency[f.B8_6],
        Dependency[f.B8_7],
        Dependency[f.B8_8],
        Dependency[f.B8_9],
        Dependency[f.B8_10],
        Dependency[f.B8_11],
        Dependency[f.B8_12],
        Dependency[f.B8_13],
        Dependency[f.B8_14],
        Dependency[f.B8_15]
      )

    /** The resulting order: B0--B6, B6_0--B6_B15, B7, B8_0--B8_15, B8--B15 */
    val expectedDeps = targets.slice(0, 7) ++ prerequisiteTargets ++ Some(targets(7)) ++ current ++ targets.drop(8)
    val expectedClasses = expectedDeps.collect { case Dependency(Left(c)) => c }

    val pm = new PhaseManager(targets ++ prerequisiteTargets ++ current, current.reverse)

    writeGraphviz(pm, "test_run_dir/PhaseManagerSpec/DeterministicOrder")

    pm.flattenedTransformOrder.map(_.getClass) should be(expectedClasses)
  }

  it should "allow conditional placement of custom transforms" in {
    val f = CustomAfterOptimizationFixture

    val targetsMinimum = Seq(Dependency[f.Custom], Dependency[f.DoneMinimum])
    val pmMinimum = new PhaseManager(targetsMinimum)

    val targetsFull = Seq(Dependency[f.Custom], Dependency[f.DoneFull])
    val pmFull = new PhaseManager(targetsFull)

    val expectedMinimum =
      Seq(classOf[f.Root], classOf[f.OptMinimum], classOf[f.AfterOpt], classOf[f.Custom], classOf[f.DoneMinimum])
    writeGraphviz(pmMinimum, "test_run_dir/PhaseManagerSpec/CustomAfterOptimization/minimum")
    pmMinimum.flattenedTransformOrder.map(_.getClass) should be(expectedMinimum)

    val expectedFull = Seq(
      classOf[f.Root],
      classOf[f.OptMinimum],
      classOf[f.OptFull],
      classOf[f.AfterOpt],
      classOf[f.Custom],
      classOf[f.DoneFull]
    )
    writeGraphviz(pmFull, "test_run_dir/PhaseManagerSpec/CustomAfterOptimization/full")
    pmFull.flattenedTransformOrder.map(_.getClass) should be(expectedFull)
  }

  it should "support optional prerequisites" in {
    val f = OptionalPrerequisitesFixture

    val targetsMinimum = Seq(Dependency[f.Custom], Dependency[f.DoneMinimum])
    val pmMinimum = new PhaseManager(targetsMinimum)

    val targetsFull = Seq(Dependency[f.Custom], Dependency[f.DoneFull])
    val pmFull = new PhaseManager(targetsFull)

    val expectedMinimum = Seq(classOf[f.Root], classOf[f.OptMinimum], classOf[f.Custom], classOf[f.DoneMinimum])
    writeGraphviz(pmMinimum, "test_run_dir/PhaseManagerSpec/CustomAfterOptimization/minimum")
    pmMinimum.flattenedTransformOrder.map(_.getClass) should be(expectedMinimum)

    val expectedFull =
      Seq(classOf[f.Root], classOf[f.OptMinimum], classOf[f.OptFull], classOf[f.Custom], classOf[f.DoneFull])
    writeGraphviz(pmFull, "test_run_dir/PhaseManagerSpec/CustomAfterOptimization/full")
    pmFull.flattenedTransformOrder.map(_.getClass) should be(expectedFull)
  }

  /** This tests a situation the ordering of edges matters. Namely, this test is dependent on the ordering in which
    * DiGraph.linearize walks the edges of each node.
    */
  it should "choose the optimal solution irregardless of prerequisite ordering" in {
    val f = OrderingFixture

    {
      val targets = Seq(Dependency[f.A], Dependency[f.B], Dependency[f.C])
      val order = Seq(classOf[f.B], classOf[f.A], classOf[f.C], classOf[f.B], classOf[f.A])
      (new PhaseManager(targets)).flattenedTransformOrder.map(_.getClass) should be(order)
    }

    {
      val targets = Seq(Dependency[f.A], Dependency[f.B], Dependency[f.Cx])
      val order = Seq(classOf[f.B], classOf[f.A], classOf[f.Cx], classOf[f.B], classOf[f.A])
      (new PhaseManager(targets)).flattenedTransformOrder.map(_.getClass) should be(order)
    }
  }

}
