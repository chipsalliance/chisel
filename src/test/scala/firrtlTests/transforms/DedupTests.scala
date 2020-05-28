// See LICENSE for license details.

package firrtlTests
package transforms

import firrtl.RenameMap
import firrtl.annotations._
import firrtl.transforms.{DedupModules, NoCircuitDedupAnnotation}
import firrtl.testutils._

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

  "The module A and A_" should "dedup with different annotation targets" in {
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
    val cs = execute(input, check, Seq(
      dontTouch(ReferenceTarget("Top", "A", Nil, "b", Nil)),
      dontTouch(ReferenceTarget("Top", "A_", Nil, "b", Nil))
    ))
    cs.annotations.toSeq should contain (dontTouch(ModuleTarget("Top", "Top").instOf("a1", "A").ref("b")))
    cs.annotations.toSeq should contain (dontTouch(ModuleTarget("Top", "Top").instOf("a2", "A").ref("b")))
    cs.annotations.toSeq should not contain dontTouch(ReferenceTarget("Top", "A_", Nil, "b", Nil))
  }
  "The module A and A_" should "be deduped with same annotation targets when there are a lot" in {
    val input =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A_
        |  module A :
        |    output x: UInt<1>[100]
        |    wire b: UInt<1>[100]
        |    x <= b
        |  module A_ :
        |    output x: UInt<1>[100]
        |    wire b: UInt<1>[100]
        |    x <= b
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A
        |  module A :
        |    output x: UInt<1>[100]
        |    wire b: UInt<1>[100]
        |    x <= b
      """.stripMargin
    val annos = (0 until 100).flatMap(i => Seq(dontTouch(s"A.b[$i]"), dontTouch(s"A_.b[$i]")))
    execute(input, check, annos)
  }
  "The module A and A_" should "be deduped with same annotations with same multi-targets" in {
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
    val B = Top.module("B")
    val A_ = Top.module("A_")
    val B_ = Top.module("B_")
    val Top_a1 = Top.module("Top").instOf("a1", "A")
    val Top_a2 = Top.module("Top").instOf("a2", "A")
    val Top_a1_b = Top_a1.instOf("b", "B")
    val Top_a2_b = Top_a2.instOf("b", "B")
    val annoAB = MultiTargetDummyAnnotation(Seq(A, B), 0)
    val annoA_B_ = MultiTargetDummyAnnotation(Seq(A_, B_), 1)
    val cs = execute(input, check, Seq(annoAB, annoA_B_))
    cs.annotations.toSeq should contain (MultiTargetDummyAnnotation(Seq(
      Top_a1, Top_a1_b
    ), 0))
    cs.annotations.toSeq should contain (MultiTargetDummyAnnotation(Seq(
      Top_a2, Top_a2_b
    ), 1))
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
    val annoA_ = MultiTargetDummyAnnotation(Seq(A_, A_.instOf("b", "B_")), 1)
    val cs = execute(input, check, Seq(annoA, annoA_))
    cs.annotations.toSeq should contain (MultiTargetDummyAnnotation(Seq(
      Top.module("Top").instOf("a1", "A"),
      Top.module("Top").instOf("a1", "A").instOf("b", "B")
    ),0))
    cs.annotations.toSeq should contain (MultiTargetDummyAnnotation(Seq(
      Top.module("Top").instOf("a2", "A"),
      Top.module("Top").instOf("a2", "A").instOf("b", "B")
    ),1))
    cs.deletedAnnotations.isEmpty should be (true)
  }
  "The deduping module A and A_" should "rename internal signals that have different names" in {
    val input =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    a1 is invalid
        |    inst a2 of A_
        |    a2 is invalid
        |  module A :
        |    input x: UInt<1>
        |    output y: UInt<1>
        |    node a = add(x, UInt(1))
        |    y <= add(a, a)
        |  module A_ :
        |    input x: UInt<1>
        |    output y: UInt<1>
        |    node b = add(x, UInt(1))
        |    y <= add(b, b)
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    a1 is invalid
        |    inst a2 of A
        |    a2 is invalid
        |  module A :
        |    input x: UInt<1>
        |    output y: UInt<1>
        |    node a = add(x, UInt<1>("h1"))
        |    y <= add(a, a)
      """.stripMargin
    val Top = CircuitTarget("Top")
    val A = Top.module("A")
    val A_ = Top.module("A_")
    val annoA  = SingleTargetDummyAnnotation(A.ref("a"))
    val annoA_ = SingleTargetDummyAnnotation(A_.ref("b"))
    val cs = execute(input, check, Seq(annoA, annoA_))
    cs.annotations.toSeq should contain (annoA)
    cs.annotations.toSeq should not contain (SingleTargetDummyAnnotation(A.ref("b")))
    cs.deletedAnnotations.isEmpty should be (true)
  }
  "main" should "not be deduped even if it's the last module" in {
    val input =
      """circuit main:
        |  module dupe:
        |    input in: UInt<8>
        |    output out: UInt<8>
        |    out <= in
        |  module main:
        |    input in:  UInt<8>
        |    output out: UInt<8>
        |    out <= in
      """.stripMargin
    val check =
      """circuit main:
        |  module dupe:
        |    input in: UInt<8>
        |    output out: UInt<8>
        |    out <= in
        |  module main:
        |    input in:  UInt<8>
        |    output out: UInt<8>
        |    out <= in
      """.stripMargin
    execute(input, check, Seq.empty)
  }
  "modules" should "not be deduped if the NoCircuitDedupAnnotation (or --no-dedup option) is supplied" in {
    val input =
      """circuit main:
        |  module dupe:
        |    input in: UInt<8>
        |    output out: UInt<8>
        |    out <= in
        |  module main:
        |    input in:  UInt<8>
        |    output out: UInt<8>
        |    out <= in
      """.stripMargin
    val check =
      """circuit main:
        |  module dupe:
        |    input in: UInt<8>
        |    output out: UInt<8>
        |    out <= in
        |  module main:
        |    input in:  UInt<8>
        |    output out: UInt<8>
        |    out <= in
      """.stripMargin
    execute(input, check, Seq(NoCircuitDedupAnnotation))
  }

  "The deduping module A and A_" should "rename instances and signals that have different names" in {
    val input =
      """circuit Top :
        |  module Top :
        |    inst a of A
        |    inst a_ of A_
        |  module A :
        |    inst b of B
        |  module A_ :
        |    inst b_ of B_
        |  module B :
        |    node foo = UInt<1>(0)
        |  module B_ :
        |    node bar = UInt<1>(0)
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    inst a of A
        |    inst a_ of A
        |  module A :
        |    inst b of B
        |  module B :
        |    node foo = UInt<1>(0)
      """.stripMargin
    val Top = CircuitTarget("Top")
    val inst1 = Top.module("Top").instOf("a", "A").instOf("b", "B")
    val inst2 = Top.module("Top").instOf("a_", "A_").instOf("b_", "B_")
    val ref1 = Top.module("Top").instOf("a", "A").instOf("b", "B").ref("foo")
    val ref2 = Top.module("Top").instOf("a_", "A_").instOf("b_", "B_").ref("bar")
    val anno1 = MultiTargetDummyAnnotation(Seq(inst1, ref1), 0)
    val anno2 = MultiTargetDummyAnnotation(Seq(inst2, ref2), 1)
    val cs = execute(input, check, Seq(anno1, anno2))
    cs.annotations.toSeq should contain (MultiTargetDummyAnnotation(Seq(
      inst1, ref1
    ),0))
    cs.annotations.toSeq should contain (MultiTargetDummyAnnotation(Seq(
      Top.module("Top").instOf("a_", "A").instOf("b", "B"),
      Top.module("Top").instOf("a_", "A").instOf("b", "B").ref("foo")
    ),1))
    cs.deletedAnnotations.isEmpty should be (true)
  }

  "The deduping module A and A_" should "rename nested instances that have different names" in {
    val input =
      """circuit Top :
        |  module Top :
        |    inst a of A
        |    inst a_ of A_
        |  module A :
        |    inst b of B
        |  module A_ :
        |    inst b_ of B_
        |  module B :
        |    inst c of C
        |  module B_ :
        |    inst c_ of C_
        |  module C :
        |    inst d of D
        |  module C_ :
        |    inst d_ of D_
        |  module D :
        |    node foo = UInt<1>(0)
        |  module D_ :
        |    node bar = UInt<1>(0)
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    inst a of A
        |    inst a_ of A
        |  module A :
        |    inst b of B
        |  module B :
        |    inst c of C
        |  module C :
        |    inst d of D
        |  module D :
        |    node foo = UInt<1>(0)
      """.stripMargin
    val Top = CircuitTarget("Top")
    val inst1 = Top.module("Top").instOf("a", "A").instOf("b", "B").instOf("c", "C").instOf("d", "D")
    val inst2 = Top.module("Top").instOf("a_", "A_").instOf("b_", "B_").instOf("c_", "C_").instOf("d_", "D_")
    val ref1 = Top.module("Top").instOf("a", "A").instOf("b", "B").instOf("c", "C").instOf("d", "D").ref("foo")
    val ref2 = Top.module("Top").instOf("a_", "A_").instOf("b_", "B_").instOf("c_", "C_").instOf("d_", "D_").ref("bar")
    val anno1 = MultiTargetDummyAnnotation(Seq(inst1, ref1), 0)
    val anno2 = MultiTargetDummyAnnotation(Seq(inst2, ref2), 1)
    val cs = execute(input, check, Seq(anno1, anno2))
    cs.annotations.toSeq should contain (MultiTargetDummyAnnotation(Seq(
      inst1, ref1
    ),0))
    cs.annotations.toSeq should contain (MultiTargetDummyAnnotation(Seq(
      Top.module("Top").instOf("a_", "A").instOf("b", "B").instOf("c", "C").instOf("d", "D"),
      Top.module("Top").instOf("a_", "A").instOf("b", "B").instOf("c", "C").instOf("d", "D").ref("foo")
    ),1))
    cs.deletedAnnotations.isEmpty should be (true)
  }

  "Deduping modules with multiple instances" should "corectly rename instances" in {
    val input =
      """circuit Top :
        |  module Top :
        |    inst b of B
        |    inst b_ of B_
        |    inst a1 of A
        |    inst a2 of A
        |  module A :
        |    inst b of B
        |    inst b_ of B_
        |  module B :
        |    inst c of C
        |  module B_ :
        |    inst c of C
        |  module C :
        |    skip
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    inst b of B
        |    inst b_ of B
        |    inst a1 of A
        |    inst a2 of A
        |  module A :
        |    inst b of B
        |    inst b_ of B
        |  module B :
        |    inst c of C
        |  module C :
        |    skip
      """.stripMargin
    val Top = CircuitTarget("Top").module("Top")
    val bInstances = Seq(
      Top.instOf("b", "B"),
      Top.instOf("b_", "B_"),
      Top.instOf("a1", "A").instOf("b_", "B_"),
      Top.instOf("a2", "A").instOf("b_", "B_"),
      Top.instOf("a1", "A").instOf("b", "B"),
      Top.instOf("a2", "A").instOf("b", "B")
    )
    val cInstances = bInstances.map(_.instOf("c", "C"))
    val annos = MultiTargetDummyAnnotation(bInstances ++ cInstances, 0)
    val cs = execute(input, check, Seq(annos))
    cs.annotations.toSeq should contain (MultiTargetDummyAnnotation(Seq(
      Top.instOf("b", "B"),
      Top.instOf("b_", "B"),
      Top.instOf("a1", "A").instOf("b_", "B"),
      Top.instOf("a2", "A").instOf("b_", "B"),
      Top.instOf("a1", "A").instOf("b", "B"),
      Top.instOf("a2", "A").instOf("b", "B"),
      Top.instOf("b", "B").instOf("c", "C"),
      Top.instOf("b_", "B").instOf("c", "C"),
      Top.instOf("a1", "A").instOf("b_", "B").instOf("c", "C"),
      Top.instOf("a2", "A").instOf("b_", "B").instOf("c", "C"),
      Top.instOf("a1", "A").instOf("b", "B").instOf("c", "C"),
      Top.instOf("a2", "A").instOf("b", "B").instOf("c", "C")
    ),0))
    cs.deletedAnnotations.isEmpty should be (true)
  }
}

