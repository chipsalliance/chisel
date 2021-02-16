// SPDX-License-Identifier: Apache-2.0

package firrtlTests.transforms

import org.scalatest.featurespec.AnyFeatureSpec
import org.scalatest.GivenWhenThen
import firrtl.testutils.FirrtlMatchers
import java.io.File

import firrtl.graph.DiGraph
import firrtl.analyses.InstanceKeyGraph
import firrtl.annotations.CircuitTarget
import firrtl.annotations.TargetToken.OfModule
import firrtl.transforms._
import firrtl.transforms.MustDeduplicateTransform._
import firrtl.transforms.MustDeduplicateTransform.DisjointChildren._
import firrtl.util.BackendCompilationUtilities.createTestDirectory
import firrtl.stage.{FirrtlSourceAnnotation, RunFirrtlTransformAnnotation}
import firrtl.options.{TargetDirAnnotation}
import logger.{LogLevel, LogLevelAnnotation, Logger}

class MustDedupSpec extends AnyFeatureSpec with FirrtlMatchers with GivenWhenThen {

  Feature("When you have a simple non-deduping hierarcy") {
    val text = """
                 |circuit A :
                 |  module C :
                 |    output io : { flip in : UInt<8>, out : UInt<8> }
                 |    io.out <= io.in
                 |  module C_1 :
                 |    output io : { flip in : UInt<8>, out : UInt<8> }
                 |    io.out <= and(io.in, UInt("hff"))
                 |  module B :
                 |    output io : { flip in : UInt<8>, out : UInt<8> }
                 |    inst c of C
                 |    io <= c.io
                 |  module B_1 :
                 |    output io : { flip in : UInt<8>, out : UInt<8> }
                 |    inst c of C_1
                 |    io <= c.io
                 |  module A :
                 |    output io : { flip in : UInt<8>, out : UInt<8> }
                 |    inst b of B
                 |    inst b_1 of B_1
                 |    io.out <= and(b.io.out, b_1.io.out)
                 |    b.io.in <= io.in
                 |    b_1.io.in <= io.in
    """.stripMargin
    val top = CircuitTarget("A")
    val bdedup = MustDeduplicateAnnotation(Seq(top.module("B"), top.module("B_1")))
    val igraph = InstanceKeyGraph(parse(text))

    Scenario("Full compilation should fail and dump reports to disk") {
      val testDir = createTestDirectory("must_dedup")
      val reportDir = new File(testDir, "reports")
      val annos = Seq(
        TargetDirAnnotation(testDir.toString),
        FirrtlSourceAnnotation(text),
        RunFirrtlTransformAnnotation(new MustDeduplicateTransform),
        MustDeduplicateReportDirectory(reportDir.toString),
        bdedup
      )

      a[DeduplicationFailureException] shouldBe thrownBy {
        (new firrtl.stage.FirrtlPhase).transform(annos)
      }

      reportDir should exist

      val report0 = new File(reportDir, "report_0.rpt")
      report0 should exist

      val expectedModules = Seq("B", "B_1", "C", "C_1")
      for (mod <- expectedModules) {
        new File(reportDir, s"modules/$mod.fir") should exist
      }
    }

    Scenario("Non-deduping children should give actionable debug information") {
      When("Finding dedup failures")
      val failure = findDedupFailures(Seq(OfModule("B"), OfModule("B_1")), igraph)

      Then("The children should appear as a failure candidate")
      failure.candidates should be(Seq(LikelyShouldMatch(OfModule("C"), OfModule("C_1"))))

      And("There should be a pretty DiGraph showing context")
      val got = makeDedupFailureDiGraph(failure, igraph.graph.transformNodes(_.module))
      val expected = DiGraph("A" -> "(B)", "A" -> "(B_1)", "(B)" -> "C [0]", "(B_1)" -> "C_1 [0]")
      // DiGraph uses referential equality so compare serialized form
      got.prettyTree() should be(expected.prettyTree())
    }

    Scenario("Unrelated hierarchies should give actionable debug information") {
      When("Finding dedup failures")
      val failure = findDedupFailures(Seq(OfModule("B"), OfModule("C_1")), igraph)

      Then("The failure should note the hierarchies don't match")
      failure.candidates should be(Seq(DisjointChildren(OfModule("B"), OfModule("C_1"), Left)))

      And("There should be a pretty DiGraph showing context")
      val got = makeDedupFailureDiGraph(failure, igraph.graph.transformNodes(_.module))
      val expected = DiGraph("A" -> "(B) [0]", "(B) [0]" -> "C", "B_1" -> "(C_1) [0]")
      // DiGraph uses referential equality so compare serialized form
      got.prettyTree() should be(expected.prettyTree())
    }
  }

  Feature("When you have a deep, non-deduping hierarchy") {
    // Shadow hierarchy just to get an InstanceKeyGraph which can only be made from a circuit
    val text = parse("""
                       |circuit A :
                       |  module E:
                       |   skip
                       |  module F :
                       |   skip
                       |  module F_1 :
                       |   inst e of E
                       |  module D :
                       |   skip
                       |  module D_1 :
                       |   skip
                       |  module C :
                       |    inst d of D
                       |    inst f of F
                       |  module C_1 :
                       |    inst d of D_1
                       |    inst f of F_1
                       |  module B :
                       |    inst c of C
                       |    inst e of E
                       |  module B_1 :
                       |    inst c of C_1
                       |    inst e of E
                       |  module A :
                       |    inst b of B
                       |    inst b_1 of B_1
                       |""".stripMargin)
    val igraph = InstanceKeyGraph(text)

    Scenario("Non-deduping children should give actionable debug information") {
      When("Finding dedup failures")
      val failure = findDedupFailures(Seq(OfModule("B"), OfModule("B_1")), igraph)

      Then("The children should appear as a failure candidate")
      failure.candidates should be(
        Seq(LikelyShouldMatch(OfModule("D"), OfModule("D_1")), DisjointChildren(OfModule("F"), OfModule("F_1"), Right))
      )

      And("There should be a pretty DiGraph showing context")
      val got = makeDedupFailureDiGraph(failure, igraph.graph.transformNodes(_.module))
      val expected = DiGraph(
        "A" -> "(B)",
        "A" -> "(B_1)",
        "(B)" -> "C",
        "C" -> "D [0]",
        "C" -> "F [1]",
        "(B_1)" -> "C_1",
        "C_1" -> "D_1 [0]",
        "C_1" -> "F_1 [1]",
        "F_1 [1]" -> "E",
        // These last 2 are undesirable but E is included because it's a submodule of disjoint F and F_1
        "(B)" -> "E",
        "(B_1)" -> "E"
      )
      // DiGraph uses referential equality so compare serialized form
      got.prettyTree() should be(expected.prettyTree())
    }
  }

  Feature("When you have multiple modules that should dedup, but don't") {
    // Shadow hierarchy just to get an InstanceKeyGraph which can only be made from a circuit
    val text = parse("""
                       |circuit A :
                       |  module D :
                       |    skip
                       |  module D_1 :
                       |    skip
                       |  module C :
                       |    skip
                       |  module C_1 :
                       |    skip
                       |  module B :
                       |    inst c of C
                       |    inst d of D
                       |  module B_1 :
                       |    inst c of C_1
                       |    inst d of D
                       |  module B_2 :
                       |    inst c of C
                       |    inst d of D_1
                       |  module A :
                       |    inst b of B
                       |    inst b_1 of B_1
                       |    inst b_2 of B_2
                       |""".stripMargin)
    val igraph = InstanceKeyGraph(text)

    Scenario("Non-deduping children should give actionable debug information") {
      When("Finding dedup failures")
      val failure = findDedupFailures(Seq(OfModule("B"), OfModule("B_1"), OfModule("B_2")), igraph)

      Then("The children should appear as a failure candidate")
      failure.candidates should be(
        Seq(LikelyShouldMatch(OfModule("C"), OfModule("C_1")), LikelyShouldMatch(OfModule("D"), OfModule("D_1")))
      )

      And("There should be a pretty DiGraph showing context")
      val got = makeDedupFailureDiGraph(failure, igraph.graph.transformNodes(_.module))
      val expected = DiGraph(
        "A" -> "(B)",
        "A" -> "(B_1)",
        "A" -> "(B_2)",
        "(B)" -> "C [0]",
        "(B)" -> "D [1]",
        "(B_1)" -> "C_1 [0]",
        "(B_1)" -> "D [1]",
        "(B_2)" -> "C [0]",
        "(B_2)" -> "D_1 [1]"
      )
      // DiGraph uses referential equality so compare serialized form
      got.prettyTree() should be(expected.prettyTree())
    }
  }

  Feature("When you have modules that should dedup, and they do") {
    val text = """
                 |circuit A :
                 |  module C :
                 |    output io : { flip in : UInt<8>, out : UInt<8> }
                 |    io.out <= io.in
                 |  module C_1 :
                 |    output io : { flip in : UInt<8>, out : UInt<8> }
                 |    io.out <= io.in
                 |  module B :
                 |    output io : { flip in : UInt<8>, out : UInt<8> }
                 |    inst c of C
                 |    io <= c.io
                 |  module B_1 :
                 |    output io : { flip in : UInt<8>, out : UInt<8> }
                 |    inst c of C_1
                 |    io <= c.io
                 |  module A :
                 |    output io : { flip in : UInt<8>, out : UInt<8> }
                 |    inst b of B
                 |    inst b_1 of B_1
                 |    io.out <= and(b.io.out, b_1.io.out)
                 |    b.io.in <= io.in
                 |    b_1.io.in <= io.in
    """.stripMargin
    val top = CircuitTarget("A")
    val bdedup = MustDeduplicateAnnotation(Seq(top.module("B"), top.module("B_1")))

    Scenario("Full compilation should succeed") {
      val testDir = createTestDirectory("must_dedup")
      val reportDir = new File(testDir, "reports")
      val annos = Seq(
        TargetDirAnnotation(testDir.toString),
        FirrtlSourceAnnotation(text),
        RunFirrtlTransformAnnotation(new MustDeduplicateTransform),
        MustDeduplicateReportDirectory(reportDir.toString),
        bdedup
      )

      (new firrtl.stage.FirrtlPhase).transform(annos)
    }
  }
}
