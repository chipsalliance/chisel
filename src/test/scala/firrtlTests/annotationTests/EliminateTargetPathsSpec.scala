// See LICENSE for license details.

package firrtlTests.annotationTests

import firrtl._
import firrtl.annotations._
import firrtl.annotations.analysis.DuplicationHelper
import firrtl.annotations.transforms.{NoSuchTargetException}
import firrtl.transforms.{DedupedResult, DontTouchAnnotation}
import firrtl.testutils.{FirrtlMatchers, FirrtlPropSpec}

object EliminateTargetPathsSpec {

  case class DummyAnnotation(target: Target) extends SingleTargetAnnotation[Target] {
    override def duplicate(n: Target): Annotation = DummyAnnotation(n)
  }
  class DummyTransform() extends Transform with ResolvedAnnotationPaths {
    override def inputForm:  CircuitForm = LowForm
    override def outputForm: CircuitForm = LowForm

    override val annotationClasses: Traversable[Class[_]] = Seq(classOf[DummyAnnotation])

    override def execute(state: CircuitState): CircuitState = state
  }

}

class EliminateTargetPathsSpec extends FirrtlPropSpec with FirrtlMatchers {
  import EliminateTargetPathsSpec._

  val input =
    """circuit Top:
      |  module Leaf:
      |    input i: UInt<1>
      |    output o: UInt<1>
      |    o <= i
      |    node a = i
      |  module Middle:
      |    input i: UInt<1>
      |    output o: UInt<1>
      |    inst l1 of Leaf
      |    inst l2 of Leaf
      |    l1.i <= i
      |    l2.i <= l1.o
      |    o <= l2.o
      |  module Top:
      |    input i: UInt<1>
      |    output o: UInt<1>
      |    inst m1 of Middle
      |    inst m2 of Middle
      |    m1.i <= i
      |    m2.i <= m1.o
      |    o <= m2.o
    """.stripMargin

  val TopCircuit = CircuitTarget("Top")
  val Top = TopCircuit.module("Top")
  val Middle = TopCircuit.module("Middle")
  val Leaf = TopCircuit.module("Leaf")

  val Top_m1_l1_a = Top.instOf("m1", "Middle").instOf("l1", "Leaf").ref("a")
  val Top_m2_l1_a = Top.instOf("m2", "Middle").instOf("l1", "Leaf").ref("a")
  val Top_m1_l2_a = Top.instOf("m1", "Middle").instOf("l2", "Leaf").ref("a")
  val Top_m2_l2_a = Top.instOf("m2", "Middle").instOf("l2", "Leaf").ref("a")
  val Middle_l1_a = Middle.instOf("l1", "Leaf").ref("a")
  val Middle_l2_a = Middle.instOf("l2", "Leaf").ref("a")
  val Leaf_a = Leaf.ref("a")

  val customTransforms = Seq(new DummyTransform())

  val inputState = CircuitState(parse(input), ChirrtlForm)
  property("Hierarchical tokens should be expanded properly") {
    val dupMap = DuplicationHelper(inputState.circuit.modules.map(_.name).toSet)

    // Only a few instance references
    dupMap.expandHierarchy(Top_m1_l1_a)
    dupMap.expandHierarchy(Top_m2_l1_a)
    dupMap.expandHierarchy(Middle_l1_a)

    dupMap.makePathless(Top_m1_l1_a).foreach { Set(TopCircuit.module("Leaf___Top_m1_l1").ref("a")) should contain(_) }
    dupMap.makePathless(Top_m2_l1_a).foreach { Set(TopCircuit.module("Leaf___Top_m2_l1").ref("a")) should contain(_) }
    dupMap.makePathless(Top_m1_l2_a).foreach { Set(Leaf_a) should contain(_) }
    dupMap.makePathless(Top_m2_l2_a).foreach { Set(Leaf_a) should contain(_) }
    dupMap.makePathless(Middle_l1_a).foreach {
      Set(
        TopCircuit.module("Leaf___Top_m1_l1").ref("a"),
        TopCircuit.module("Leaf___Top_m2_l1").ref("a"),
        TopCircuit.module("Leaf___Middle_l1").ref("a")
      ) should contain(_)
    }
    dupMap.makePathless(Middle_l2_a).foreach { Set(Leaf_a) should contain(_) }
    dupMap.makePathless(Leaf_a).foreach {
      Set(
        TopCircuit.module("Leaf___Top_m1_l1").ref("a"),
        TopCircuit.module("Leaf___Top_m2_l1").ref("a"),
        TopCircuit.module("Leaf___Middle_l1").ref("a"),
        Leaf_a
      ) should contain(_)
    }
    dupMap.makePathless(Top).foreach { Set(Top) should contain(_) }
    dupMap.makePathless(Middle).foreach {
      Set(
        TopCircuit.module("Middle___Top_m1"),
        TopCircuit.module("Middle___Top_m2"),
        Middle
      ) should contain(_)
    }
    dupMap.makePathless(Leaf).foreach {
      Set(
        TopCircuit.module("Leaf___Top_m1_l1"),
        TopCircuit.module("Leaf___Top_m2_l1"),
        TopCircuit.module("Leaf___Middle_l1"),
        Leaf
      ) should contain(_)
    }
  }

  property("Hierarchical donttouch should be resolved properly") {
    val inputState = CircuitState(parse(input), ChirrtlForm, Seq(DontTouchAnnotation(Top_m1_l1_a)))
    val customTransforms = Seq(new LowFirrtlOptimization())
    val outputState = new LowFirrtlCompiler().compile(inputState, customTransforms)
    val check =
      """circuit Top :
        |  module Leaf :
        |    input i : UInt<1>
        |    output o : UInt<1>

        |    node a = i
        |    o <= i
        |
        |  module Middle :
        |    input i : UInt<1>
        |    output o : UInt<1>

        |    inst l1 of Leaf
        |    inst l2 of Leaf
        |    o <= l2.o
        |    l1.i <= i
        |    l2.i <= l1.o

        |  module Top :
        |    input i : UInt<1>
        |    output o : UInt<1>

        |    inst m1 of Middle
        |    inst m2 of Middle
        |    o <= m2.o
        |    m1.i <= i
        |    m2.i <= m1.o""".stripMargin

    canonicalize(outputState.circuit).serialize should be(canonicalize(parse(check)).serialize)
    outputState.annotations.collect {
      case x: DontTouchAnnotation => x.target
    } should be(Seq(Top.circuitTarget.module("Top").instOf("m1", "Middle").instOf("l1", "Leaf").ref("a")))
  }

  property("No name conflicts between old and new modules") {
    val input =
      """circuit Top:
        |  module Middle:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    o <= i
        |  module Top:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    inst m1 of Middle
        |    inst m2 of Middle
        |    inst x of Middle___Top_m1
        |    x.i <= i
        |    m1.i <= i
        |    m2.i <= m1.o
        |    o <= m2.o
        |  module Middle___Top_m1:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    o <= i
        |    node a = i
      """.stripMargin
    val checks =
      """circuit Top :
        |  module Middle :
        |  module Top :
        |  module Middle___Top_m1 :
        |  module Middle____Top_m1 :""".stripMargin.split("\n")
    val Top_m1 = Top.instOf("m1", "Middle")
    val inputState = CircuitState(parse(input), ChirrtlForm, Seq(DummyAnnotation(Top_m1)))
    val outputState = new LowFirrtlCompiler().compile(inputState, customTransforms)
    val outputLines = outputState.circuit.serialize.split("\n")
    checks.foreach { line =>
      outputLines should contain(line)
    }
  }

  property("Previously unused modules should remain, but newly unused modules should be eliminated") {
    val input =
      """circuit Top:
        |  module Leaf:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    o <= i
        |    node a = i
        |  module Middle:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    o <= i
        |  module Top:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    inst m1 of Middle
        |    inst m2 of Middle
        |    m1.i <= i
        |    m2.i <= m1.o
        |    o <= m2.o
      """.stripMargin

    val checks =
      """circuit Top :
        |  module Leaf :
        |  module Top :
        |  module Middle___Top_m1 :
        |  module Middle___Top_m2 :""".stripMargin.split("\n")

    val Top_m1 = Top.instOf("m1", "Middle")
    val Top_m2 = Top.instOf("m2", "Middle")
    val inputState = CircuitState(parse(input), ChirrtlForm, Seq(DummyAnnotation(Top_m1), DummyAnnotation(Top_m2)))
    val outputState = new LowFirrtlCompiler().compile(inputState, customTransforms)
    val outputLines = outputState.circuit.serialize.split("\n")

    checks.foreach { line =>
      outputLines should contain(line)
    }
    checks.foreach { line =>
      outputLines should not contain ("  module Middle :")
    }
  }

  property("Paths with incorrect names should error") {
    val input =
      """circuit Top:
        |  module Leaf:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    o <= i
        |    node a = i
        |  module Middle:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    o <= i
        |  module Top:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    inst m1 of Middle
        |    inst m2 of Middle
        |    m1.i <= i
        |    m2.i <= m1.o
        |    o <= m2.o
      """.stripMargin
    val e1 = the[CustomTransformException] thrownBy {
      val Top_m1 = Top.instOf("m1", "MiddleX")
      val inputState = CircuitState(parse(input), ChirrtlForm, Seq(DummyAnnotation(Top_m1)))
      new LowFirrtlCompiler().compile(inputState, customTransforms)
    }
    e1.cause shouldBe a[NoSuchTargetException]

    val e2 = the[CustomTransformException] thrownBy {
      val Top_m2 = Top.instOf("x2", "Middle")
      val inputState = CircuitState(parse(input), ChirrtlForm, Seq(DummyAnnotation(Top_m2)))
      new LowFirrtlCompiler().compile(inputState, customTransforms)
    }
    e2.cause shouldBe a[NoSuchTargetException]
  }

  property("No name conflicts between two new modules") {
    val input =
      """circuit Top:
        |  module Top:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    inst m1 of Middle_
        |    inst m2 of Middle
        |    m1.i <= i
        |    m2.i <= m1.o
        |    o <= m2.o
        |  module Middle:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    inst _l of Leaf
        |    _l.i <= i
        |    o <= _l.o
        |  module Middle_:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    inst l of Leaf
        |    l.i <= i
        |    node x = i
        |    o <= l.o
        |  module Leaf:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    o <= i
      """.stripMargin
    val checks =
      """circuit Top :
        |  module Middle :
        |  module Middle_ :
        |  module Top :
        |  module Leaf___Middle__l :
        |  module Leaf____Middle__l :""".stripMargin.split("\n")
    val Middle_l1 = CircuitTarget("Top").module("Middle").instOf("_l", "Leaf")
    val Middle_l2 = CircuitTarget("Top").module("Middle_").instOf("l", "Leaf")
    val inputState =
      CircuitState(parse(input), ChirrtlForm, Seq(DummyAnnotation(Middle_l1), DummyAnnotation(Middle_l2)))
    val outputState = new LowFirrtlCompiler().compile(inputState, customTransforms)
    val outputLines = outputState.circuit.serialize.split("\n")
    checks.foreach { line =>
      outputLines should contain(line)
    }
  }

  property("Keep annotations of modules not instantiated") {
    val input =
      """circuit Top:
        |  module Top:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    inst m1 of Middle
        |    inst m2 of Middle
        |    m1.i <= i
        |    m2.i <= m1.o
        |    o <= m2.o
        |  module Middle:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    inst _l of Leaf
        |    _l.i <= i
        |    o <= _l.o
        |  module Middle_:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    o <= UInt(0)
        |  module Leaf:
        |    input i: UInt<1>
        |    output o: UInt<1>
        |    o <= i
      """.stripMargin
    val checks =
      """circuit Top :
        |  module Middle_ :""".stripMargin.split("\n")
    val Middle_ = CircuitTarget("Top").module("Middle_").ref("i")
    val inputState = CircuitState(parse(input), ChirrtlForm, Seq(DontTouchAnnotation(Middle_)))
    val outputState = new VerilogCompiler().compile(inputState, customTransforms)
    val outputLines = outputState.circuit.serialize.split("\n")
    checks.foreach { line =>
      outputLines should contain(line)
    }
  }

  property("It should remove ResolvePaths annotations") {
    val input =
      """|circuit Foo:
         |  module Bar:
         |    skip
         |  module Foo:
         |    inst bar of Bar
         |""".stripMargin

    CircuitState(passes.ToWorkingIR.run(Parser.parse(input)), UnknownForm, Nil)
      .resolvePaths(Seq(CircuitTarget("Foo").module("Foo").instOf("bar", "Bar")))
      .annotations
      .collect { case a: firrtl.annotations.transforms.ResolvePaths => a } should be(empty)
  }

  property("It should rename module annotations") {
    val input =
      """|circuit Foo:
         |  module Bar:
         |    node x = UInt<1>(0)
         |  module Foo:
         |    inst bar of Bar
         |    inst baz of Bar""".stripMargin
    val check =
      """|circuit Foo:
         |  module Bar___Foo_bar:
         |    node x = UInt<1>(0)
         |  module Bar:
         |    node x = UInt<1>(0)
         |  module Foo:
         |    inst bar of Bar___Foo_bar
         |    inst baz of Bar""".stripMargin
    val Bar_x = CircuitTarget("Foo").module("Bar").ref("x")
    val output = CircuitState(passes.ToWorkingIR.run(Parser.parse(input)), UnknownForm, Seq(DontTouchAnnotation(Bar_x)))
      .resolvePaths(Seq(CircuitTarget("Foo").module("Foo").instOf("bar", "Bar")))

    val parsedCheck = Parser.parse(check)
    info(output.circuit.serialize)
    (output.circuit.serialize) should be(parsedCheck.serialize)

    val newBar_x = CircuitTarget("Foo").module("Bar___Foo_bar").ref("x")

    (output.annotations.filter {
      case _: DeletedAnnotation => false
      case _ => true
    } should contain).allOf(DontTouchAnnotation(newBar_x), DontTouchAnnotation(Bar_x))
  }

  property("It should not rename lone instances") {
    val input =
      """|circuit Foo:
         |  module Baz:
         |    node foo = UInt<1>(0)
         |  module Bar:
         |    node foo = UInt<1>(0)
         |    inst baz of Baz
         |  module Foo:
         |    node foo = UInt<1>(0)
         |    inst bar of Bar
         |""".stripMargin
    val targets = Seq(
      CircuitTarget("Foo").module("Foo").instOf("bar", "Bar").instOf("baz", "Baz"),
      CircuitTarget("Foo").module("Foo").instOf("bar", "Bar"),
      CircuitTarget("Foo").module("Foo")
    )
    val dontTouches = targets.map(t => DontTouchAnnotation(t.ref("foo")))
    val inputCircuit = Parser.parse(input)
    val output = CircuitState(passes.ToWorkingIR.run(inputCircuit), UnknownForm, dontTouches)
      .resolvePaths(targets)

    info(output.circuit.serialize)

    output.circuit.serialize should be(inputCircuit.serialize)
    (output.annotations.collect {
      case a: DontTouchAnnotation => a
    } should contain).allOf(
      DontTouchAnnotation(ModuleTarget("Foo", "Foo").ref("foo")),
      DontTouchAnnotation(ModuleTarget("Foo", "Bar").ref("foo")),
      DontTouchAnnotation(ModuleTarget("Foo", "Baz").ref("foo"))
    )
  }

  property("It should not rename parent lone instances but still rename children") {
    val input =
      """|circuit FooBar:
         |  module Bar:
         |    node baz = UInt<1>(0)
         |  module FooBar:
         |    inst foo of Foo
         |  module Foo:
         |    inst bar of Bar
         |    inst barBar of Bar
         |""".stripMargin

    // modules Foo and FooBar should not be renamed
    val checks =
      """circuit FooBar :
        |  module Foo :
        |  module FooBar :
        |    inst foo of Foo""".stripMargin.split("\n")

    val targets = Seq(
      CircuitTarget("FooBar").module("FooBar").instOf("foo", "Foo").instOf("bar", "Bar"),
      CircuitTarget("FooBar").module("FooBar").instOf("foo", "Foo").instOf("barBar", "Bar")
    )
    val dontTouches = targets.map(t => DontTouchAnnotation(t.ref("baz")))
    val output = CircuitState(passes.ToWorkingIR.run(Parser.parse(input)), UnknownForm, dontTouches)
      .resolvePaths(targets)

    info(output.circuit.serialize)

    val outputLines = output.circuit.serialize.split("\n")
    checks.foreach { line =>
      outputLines should contain(line)
    }

    (output.annotations.collect {
      case a: DontTouchAnnotation => a
    } should contain).allOf(
      DontTouchAnnotation(ModuleTarget("FooBar", "Bar___Foo_bar").ref("baz")),
      DontTouchAnnotation(ModuleTarget("FooBar", "Bar___Foo_barBar").ref("baz"))
    )
  }

  property("It should use DedupedResult names") {
    val input =
      """|circuit Top:
         |  module Baz:
         |    skip
         |  module Bar:
         |    inst baz of Baz
         |    inst bazzz of Baz
         |    skip
         |  module Top:
         |    inst bar of Bar
         |""".stripMargin
    val checks =
      """|circuit Top :
         |  module Baz_0 :
         |  module Baz_1 :
         |    inst baz of Baz_0
         |    inst bazzz of Baz_1
         |""".stripMargin.split("\n")
    val baz = CircuitTarget("Top").module("Top").instOf("bar", "Bar").instOf("baz", "Baz")
    val bazzz = CircuitTarget("Top").module("Top").instOf("bar", "Bar").instOf("bazzz", "Baz")
    val annos = Seq(
      DedupedResult(ModuleTarget("Top", "Baz_0"), Some(baz), 0),
      DedupedResult(ModuleTarget("Top", "Baz_1"), Some(bazzz), 1),
      DontTouchAnnotation(baz.ref("foo")),
      DontTouchAnnotation(bazzz.ref("foo"))
    )
    val inputCircuit = Parser.parse(input)
    val output = CircuitState(passes.ToWorkingIR.run(inputCircuit), UnknownForm, annos)
      .resolvePaths(Seq(baz, bazzz))

    info(output.circuit.serialize)

    val outputLines = output.circuit.serialize.split("\n")
    checks.foreach { line =>
      outputLines should contain(line)
    }
    (output.annotations.collect {
      case a: DontTouchAnnotation => a
    } should contain).allOf(
      DontTouchAnnotation(ModuleTarget("Top", "Baz_0").ref("foo")),
      DontTouchAnnotation(ModuleTarget("Top", "Baz_1").ref("foo"))
    )
  }

  property("It should not rename untouched modules") {
    val input =
      """|circuit Top:
         |  module Baz:
         |    node foo = UInt<1>(0)
         |  module Bar:
         |    inst lkj of Baz
         |    inst asdf of Baz
         |  module Top:
         |    inst bar of Bar
         |    inst baz of Baz
         |""".stripMargin
    val asdf = ModuleTarget("Top", "Top").instOf("bar", "Bar").instOf("asdf", "Baz")
    val lkj = ModuleTarget("Top", "Top").instOf("bar", "Bar").instOf("lkj", "Baz")
    val baz = ModuleTarget("Top", "Top").instOf("baz", "Baz")
    val annos = Seq(
      DontTouchAnnotation(asdf.ref("foo")),
      DontTouchAnnotation(lkj.ref("foo")),
      DontTouchAnnotation(baz.ref("foo"))
    )
    val inputCircuit = Parser.parse(input)
    val output = CircuitState(passes.ToWorkingIR.run(inputCircuit), UnknownForm, annos)
      .resolvePaths(Seq(asdf, lkj))

    info(output.circuit.serialize)

    output.annotations.collect { case a: DontTouchAnnotation => a } should be(
      Seq(
        DontTouchAnnotation(ModuleTarget("Top", "Baz___Bar_asdf").ref("foo")),
        DontTouchAnnotation(ModuleTarget("Top", "Baz___Bar_lkj").ref("foo")),
        DontTouchAnnotation(baz.ref("foo"))
      )
    )
  }

  property("It should properly rename modules with multiple instances") {
    val input =
      """|circuit Top:
         |  module Core:
         |    node clock = UInt<1>(0)
         |  module System:
         |    inst core_1 of Core
         |    inst core_2 of Core
         |    inst core_3 of Core
         |    inst core_4 of Core
         |  module Top:
         |    inst system of System
         |""".stripMargin
    val absCoreInstances = (1 to 4).map { i =>
      ModuleTarget("Top", "Top").instOf("system", "System").instOf(s"core_$i", "Core")
    }
    val relCoreInstances = (1 to 4).map { i =>
      ModuleTarget("Top", "System").instOf(s"core_$i", "Core")
    }
    val coreModule = ModuleTarget("Top", "Core")
    val annos = (coreModule +: (relCoreInstances ++ absCoreInstances)).map(DummyAnnotation(_))
    val inputCircuit = Parser.parse(input)
    val output = CircuitState(passes.ToWorkingIR.run(inputCircuit), UnknownForm, annos)
      .resolvePaths(relCoreInstances ++ absCoreInstances)

    info(output.circuit.serialize)

    val checkDontTouches = (1 to 4).map { i =>
      DummyAnnotation(ModuleTarget("Top", s"Core___System_core_$i"))
    }
    output.annotations.collect { case a: DummyAnnotation => a } should be(checkDontTouches)
  }
}
