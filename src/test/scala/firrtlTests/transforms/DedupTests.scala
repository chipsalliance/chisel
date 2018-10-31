// See LICENSE for license details.

package firrtlTests
package transforms

import firrtl.RenameMap
import firrtl.annotations._
import firrtl.transforms.DedupModules


/**
 * Tests inline instances transformation
 */
class DedupModuleTests extends HighTransformSpec {
  case class MultiTargetDummyAnnotation(targets: Seq[Target], tag: Int) extends Annotation {
    override def update(renames: RenameMap): Seq[Annotation] = {
      val newTargets = targets.flatMap(renames(_))
      Seq(MultiTargetDummyAnnotation(newTargets, tag))
    }
  }
  case class SingleTargetDummyAnnotation(target: ComponentName) extends SingleTargetAnnotation[ComponentName] {
    override def duplicate(n: ComponentName): Annotation = SingleTargetDummyAnnotation(n)
  }
  def transform = new DedupModules
  "The module A" should "be deduped" in {
     val input =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A_
          |  module A :
          |    output x: UInt<1>
          |    x <= UInt(1)
          |  module A_ :
          |    output x: UInt<1>
          |    x <= UInt(1)
          """.stripMargin
     val check =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A
          |  module A :
          |    output x: UInt<1>
          |    x <= UInt(1)
          """.stripMargin
     execute(input, check, Seq.empty)
  }
  "The module A and B" should "be deduped" in {
     val input =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A_
          |  module A :
          |    output x: UInt<1>
          |    inst b of B
          |    x <= b.x
          |  module A_ :
          |    output x: UInt<1>
          |    inst b of B_
          |    x <= b.x
          |  module B :
          |    output x: UInt<1>
          |    x <= UInt(1)
          |  module B_ :
          |    output x: UInt<1>
          |    x <= UInt(1)
          """.stripMargin
     val check =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A
          |  module A :
          |    output x: UInt<1>
          |    inst b of B
          |    x <= b.x
          |  module B :
          |    output x: UInt<1>
          |    x <= UInt(1)
          """.stripMargin
     execute(input, check, Seq.empty)
  }
  "The module A and B with comments" should "be deduped" in {
     val input =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A_
          |  module A : @[yy 2:2]
          |    output x: UInt<1> @[yy 2:2]
          |    inst b of B @[yy 2:2]
          |    x <= b.x @[yy 2:2]
          |  module A_ : @[xx 1:1]
          |    output x: UInt<1> @[xx 1:1]
          |    inst b of B_ @[xx 1:1]
          |    x <= b.x @[xx 1:1]
          |  module B :
          |    output x: UInt<1>
          |    x <= UInt(1)
          |  module B_ :
          |    output x: UInt<1>
          |    x <= UInt(1)
          """.stripMargin
     val check =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A
          |  module A : @[yy 2:2]
          |    output x: UInt<1> @[yy 2:2]
          |    inst b of B @[yy 2:2]
          |    x <= b.x @[yy 2:2]
          |  module B :
          |    output x: UInt<1>
          |    x <= UInt(1)
          """.stripMargin
     execute(input, check, Seq.empty)
  }
  "A_ but not A" should "be deduped if not annotated" in {
     val input =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A_
          |  module A : @[yy 2:2]
          |    output x: UInt<1> @[yy 2:2]
          |    x <= UInt(1)
          |  module A_ : @[xx 1:1]
          |    output x: UInt<1> @[xx 1:1]
          |    x <= UInt(1)
          """.stripMargin
     val check =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A_
          |  module A : @[yy 2:2]
          |    output x: UInt<1> @[yy 2:2]
          |    x <= UInt(1)
          |  module A_ : @[xx 1:1]
          |    output x: UInt<1> @[xx 1:1]
          |    x <= UInt(1)
          """.stripMargin
     execute(input, check, Seq(dontDedup("A")))
  }
  "The module A and A_" should "be deduped even with different port names and info, and annotations should remapped" in {
     val input =
        """circuit Top :
          |  module Top :
          |    output out: UInt<1>
          |    inst a1 of A
          |    inst a2 of A_
          |    out <= and(a1.x, a2.y)
          |  module A : @[yy 2:2]
          |    output x: UInt<1> @[yy 2:2]
          |    x <= UInt(1)
          |  module A_ : @[xx 1:1]
          |    output y: UInt<1> @[xx 1:1]
          |    y <= UInt(1)
        """.stripMargin
     val check =
        """circuit Top :
          |  module Top :
          |    output out: UInt<1>
          |    inst a1 of A
          |    inst a2 of A
          |    out <= and(a1.x, a2.x)
          |  module A : @[yy 2:2]
          |    output x: UInt<1> @[yy 2:2]
          |    x <= UInt(1)
        """.stripMargin

     val mname = ModuleName("Top", CircuitName("Top"))
     val finalState = execute(input, check, Seq(SingleTargetDummyAnnotation(ComponentName("a2.y", mname))))
     finalState.annotations.collect({ case d: SingleTargetDummyAnnotation => d }).head should be(SingleTargetDummyAnnotation(ComponentName("a2.x", mname)))
  }

  "Extmodules" should "with the same defname and parameters should dedup" in {
     val input =
        """circuit Top :
          |  module Top :
          |    output out: UInt<1>
          |    inst a1 of A
          |    inst a2 of A_
          |    out <= and(a1.x, a2.y)
          |  module A : @[yy 2:2]
          |    output x: UInt<1> @[yy 2:2]
          |    inst b of B
          |    x <= b.u
          |  module A_ : @[xx 1:1]
          |    output y: UInt<1> @[xx 1:1]
          |    inst c of C
          |    y <= c.v
          |  extmodule B : @[aa 3:3]
          |    output u : UInt<1> @[aa 4:4]
          |    defname = BB
          |    parameter N = 0
          |  extmodule C : @[bb 5:5]
          |    output v : UInt<1> @[bb 6:6]
          |    defname = BB
          |    parameter N = 0
        """.stripMargin
     val check =
        """circuit Top :
          |  module Top :
          |    output out: UInt<1>
          |    inst a1 of A
          |    inst a2 of A
          |    out <= and(a1.x, a2.x)
          |  module A : @[yy 2:2]
          |    output x: UInt<1> @[yy 2:2]
          |    inst b of B
          |    x <= b.u
          |  extmodule B : @[aa 3:3]
          |    output u : UInt<1> @[aa 4:4]
          |    defname = BB
          |    parameter N = 0
        """.stripMargin

     execute(input, check, Seq.empty)
  }

  "Extmodules" should "with the different defname or parameters should NOT dedup" in {
     def mkfir(defnames: (String, String), params: (String, String)) =
       s"""circuit Top :
          |  module Top :
          |    output out: UInt<1>
          |    inst a1 of A
          |    inst a2 of A_
          |    out <= and(a1.x, a2.y)
          |  module A : @[yy 2:2]
          |    output x: UInt<1> @[yy 2:2]
          |    inst b of B
          |    x <= b.u
          |  module A_ : @[xx 1:1]
          |    output y: UInt<1> @[xx 1:1]
          |    inst c of C
          |    y <= c.v
          |  extmodule B : @[aa 3:3]
          |    output u : UInt<1> @[aa 4:4]
          |    defname = ${defnames._1}
          |    parameter N = ${params._1}
          |  extmodule C : @[bb 5:5]
          |    output v : UInt<1> @[bb 6:6]
          |    defname = ${defnames._2}
          |    parameter N = ${params._2}
        """.stripMargin
     val diff_defname = mkfir(("BB", "CC"), ("0", "0"))
     execute(diff_defname, diff_defname, Seq.empty)
     val diff_params = mkfir(("BB", "BB"), ("0", "1"))
     execute(diff_params, diff_params, Seq.empty)
  }
  "The module A and B" should "be deduped with the first module in order" in {
    val input =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A_
        |  module A :
        |    output x: UInt<1>
        |    inst b of B
        |    x <= b.x
        |  module A_ :
        |    output x: UInt<1>
        |    inst b of B_
        |    x <= b.x
        |  module B :
        |    output x: UInt<1>
        |    x <= UInt(1)
        |  module B_ :
        |    output x: UInt<1>
        |    x <= UInt(1)
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A
        |  module A :
        |    output x: UInt<1>
        |    inst b of B
        |    x <= b.x
        |  module B :
        |    output x: UInt<1>
        |    x <= UInt(1)
      """.stripMargin
    execute(input, check, Seq.empty)
  }

  "The module A and A_" should "be deduped with fields that sort of match" in {
    val input =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A_
        |  module A :
        |    output x: UInt<1>
        |    wire b: {c: UInt<1>}
        |    x <= b.c
        |  module A_ :
        |    output x: UInt<1>
        |    wire b: {b: UInt<1>}
        |    x <= b.b
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A
        |  module A :
        |    output x: UInt<1>
        |    wire b: {c: UInt<1>}
        |    x <= b.c
      """.stripMargin
    execute(input, check, Seq.empty)
  }

  "The module A and A_" should "not be deduped with different annotation targets" in {
    val input =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A_
        |  module A :
        |    output x: UInt<1>
        |    wire b: UInt<1>
        |    x <= b
        |  module A_ :
        |    output x: UInt<1>
        |    wire b: UInt<1>
        |    x <= b
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A_
        |  module A :
        |    output x: UInt<1>
        |    wire b: UInt<1>
        |    x <= b
        |  module A_ :
        |    output x: UInt<1>
        |    wire b: UInt<1>
        |    x <= b
      """.stripMargin
    execute(input, check, Seq(dontTouch("A.b")))
  }

  "The module A and A_" should "be deduped with same annotation targets" in {
    val input =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A_
        |  module A :
        |    output x: UInt<1>
        |    wire b: UInt<1>
        |    x <= b
        |  module A_ :
        |    output x: UInt<1>
        |    wire b: UInt<1>
        |    x <= b
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A
        |  module A :
        |    output x: UInt<1>
        |    wire b: UInt<1>
        |    x <= b
      """.stripMargin
    execute(input, check, Seq(dontTouch("A.b"), dontTouch("A_.b")))
  }
  "The module A and A_" should "not be deduped with same annotations with same multi-targets, but which have different root modules" in {
    val input =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A_
        |  module A :
        |    output x: UInt<1>
        |    inst b of B
        |    x <= b.x
        |  module A_ :
        |    output x: UInt<1>
        |    inst b of B_
        |    x <= b.x
        |  module B :
        |    output x: UInt<1>
        |    x <= UInt(1)
        |  module B_ :
        |    output x: UInt<1>
        |    x <= UInt(1)
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A_
        |  module A :
        |    output x: UInt<1>
        |    inst b of B
        |    x <= b.x
        |  module A_ :
        |    output x: UInt<1>
        |    inst b of B_
        |    x <= b.x
        |  module B :
        |    output x: UInt<1>
        |    x <= UInt(1)
        |  module B_ :
        |    output x: UInt<1>
        |    x <= UInt(1)
      """.stripMargin
    val Top = CircuitTarget("Top")
    val A = Top.module("A")
    val B = Top.module("B")
    val A_ = Top.module("A_")
    val B_ = Top.module("B_")
    val annoAB = MultiTargetDummyAnnotation(Seq(A, B), 0)
    val annoA_B_ = MultiTargetDummyAnnotation(Seq(A_, B_), 0)
    val cs = execute(input, check, Seq(annoAB, annoA_B_))
    cs.annotations.toSeq should contain (annoAB)
    cs.annotations.toSeq should contain (annoA_B_)
  }
  "The module A and A_" should "be deduped with same annotations with same multi-targets, that share roots" in {
    val input =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A_
        |  module A :
        |    output x: UInt<1>
        |    inst b of B
        |    x <= b.x
        |  module A_ :
        |    output x: UInt<1>
        |    inst b of B_
        |    x <= b.x
        |  module B :
        |    output x: UInt<1>
        |    x <= UInt(1)
        |  module B_ :
        |    output x: UInt<1>
        |    x <= UInt(1)
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A
        |  module A :
        |    output x: UInt<1>
        |    inst b of B
        |    x <= b.x
        |  module B :
        |    output x: UInt<1>
        |    x <= UInt(1)
      """.stripMargin
    val Top = CircuitTarget("Top")
    val A = Top.module("A")
    val A_ = Top.module("A_")
    val annoA = MultiTargetDummyAnnotation(Seq(A, A.instOf("b", "B")), 0)
    val annoA_ = MultiTargetDummyAnnotation(Seq(A_, A_.instOf("b", "B_")), 0)
    val cs = execute(input, check, Seq(annoA, annoA_))
    cs.annotations.toSeq should contain (annoA)
    cs.annotations.toSeq should not contain (annoA_)
    cs.deletedAnnotations.isEmpty should be (true)
  }
  "The deduping module A and A_" should "renamed internal signals that have different names" in {
    val input =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A_
        |  module A :
        |    output y: UInt<1>
        |    y <= UInt(1)
        |  module A_ :
        |    output x: UInt<1>
        |    x <= UInt(1)
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A
        |  module A :
        |    output y: UInt<1>
        |    y <= UInt<1>("h1")
      """.stripMargin
    val Top = CircuitTarget("Top")
    val A = Top.module("A")
    val A_ = Top.module("A_")
    val annoA = SingleTargetDummyAnnotation(A.ref("y"))
    val annoA_ = SingleTargetDummyAnnotation(A_.ref("x"))
    val cs = execute(input, check, Seq(annoA, annoA_))
    cs.annotations.toSeq should contain (annoA)
    cs.annotations.toSeq should not contain (SingleTargetDummyAnnotation(A.ref("x")))
    cs.deletedAnnotations.isEmpty should be (true)
  }
}

