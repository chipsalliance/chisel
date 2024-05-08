package circtTests.tywavesTests.scalaReflectionTests

import chisel3.tywaves.{ClassParam, TywavesChiselAnnotation}
import circtTests.tywavesTests.TestUtils.countSubstringOccurrences
import org.scalatest.AppendedClues.convertToClueful
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

import java.math.BigInteger

class ParametersAnnotationSpec extends AnyFunSpec with Matchers with chiselTests.Utils {

  describe("Parameters of a scala class") {

    it("should return empty list without params") {
      class A
      val a = new A
      TywavesChiselAnnotation.getConstructorParams(a) should be(Seq())
    }

    it("should access the val of a class constructor") {
      val x = 10
      class A(val a: Int)
      val a = new A(2)
      val expectA = Seq(ClassParam("a", "Int", Some(2.toString)))
      TywavesChiselAnnotation.getConstructorParams(a) should be(expectA)

      class B(val a: Boolean, val b: Float)
      val b = new B(true, 1.1f)
      val expectB =
        Seq(
          ClassParam("a", "Boolean", Some("true")),
          ClassParam("b", "Float", Some(1.1.toString))
        )
      TywavesChiselAnnotation.getConstructorParams(b) should be(expectB)

      class C(val a: Int, val b: String)
      val c = new C(1, "hello")
      val expectC = Seq(ClassParam("a", "Int", Some(1.toString)), ClassParam("b", "String", Some("hello")))
      TywavesChiselAnnotation.getConstructorParams(c) should be(expectC)
    }

    it("should access normal parameters of a class constructor") {
      class A(a: Int)
      val a = new A(1)
      val expectA = Seq(ClassParam("a", "Int", None))
      TywavesChiselAnnotation.getConstructorParams(a) should be(expectA)

      class B(a: Boolean, b: Float)
      val b = new B(true, 1.1f)
      val expectB = Seq(ClassParam("a", "Boolean", None), ClassParam("b", "Float", None))
      TywavesChiselAnnotation.getConstructorParams(b) should be(expectB)

      class C(a: Int, b: String)
      val c = new C(1, "hello")
      val expectC = Seq(ClassParam("a", "Int", None), ClassParam("b", "String", None))
      TywavesChiselAnnotation.getConstructorParams(c) should be(expectC)

    }

    it("should NOT access the internal val of a class") {
      class A(a: Int) {
        val c = 10
      }
      val a = new A(1)
      val expectA = Seq(ClassParam("a", "Int", None))
      TywavesChiselAnnotation.getConstructorParams(a) should be(expectA)
    }

    it("should access private val of a class constructor") {
      class A(private val a: Int) {
        val c = 10
      }
      val a = new A(1)
      val expectA = Seq(ClassParam("a", "Int", Some(1.toString)))
      TywavesChiselAnnotation.getConstructorParams(a) should be(expectA)

      class B(protected val b: Int) {
        val c = 10
      }
      val b = new B(1)
      val expectB = Seq(ClassParam("b", "Int", Some(1.toString)))
      TywavesChiselAnnotation.getConstructorParams(b) should be(expectB)
    }

    it("should NOT access val of a super class") {
      class A(private val a: Int) {
        val c = 10
      }
      val a = new A(1)
      val expectA = Seq(ClassParam("a", "Int", Some(1.toString)))
      TywavesChiselAnnotation.getConstructorParams(a) should be(expectA)

      class B(protected val b: Int) extends A(b) {
        override val c = 10
      }
      val b = new B(1)
      val expectB = Seq(ClassParam("b", "Int", Some(1.toString)))
      TywavesChiselAnnotation.getConstructorParams(b) should be(expectB)

      class C extends A(1) {
        val d = 10
      }
      val c = new C
      TywavesChiselAnnotation.getConstructorParams(c) should be(Seq())
    }

    it("should parameter of complex types of a class") {
      class A(private val a: Int, val x: Char) {
        val c = 10
      }
      val a = new A(1, 'c')

      class B(protected val aClass: A) {
        val c = 10
      }
      val b = new B(a)
      val expectB = Seq(ClassParam("aClass", "A", Some("A(a: 1, x: c)")))
      TywavesChiselAnnotation.getConstructorParams(b) should be(expectB)
    }

    it("should parameter of nested complex types") {
      class Base(val a: Int, c: Float)
      class A(private val a: Int, val x: String, val base: Base) {
        val c = 10
      }
      val a = new A(1, "ciao", new Base(3, 1.1f))

      class B(protected val aClass: A) {
        val c = 10
      }
      val b = new B(a)
      val expectB = Seq(ClassParam("aClass", "A", Some("A(a: 1, x: ciao, base: Base(a: 3, c))")))
      TywavesChiselAnnotation.getConstructorParams(b) should be(expectB)
    }

    it("should work with case classes") {
      case class Base(val a: Int, c: Float)
      case class A(private val a: Int, val x: String, val base: Base) { val c = 10 }
      val a = A(1, "ciao", Base(3, 1.1f))

      case class B(protected val aClass: A) {
        val c = 10
      }
      val b = B(a)
      val expectB = Seq(ClassParam("aClass", "A", Some("A(a: 1, x: ciao, base: Base(a: 3, c: 1.1))")))
      TywavesChiselAnnotation.getConstructorParams(b) should be(expectB)
    }
  }

  describe("Parameters with Chisel constructs") {
    import circt.stage.ChiselStage
    import chisel3.stage.ChiselGeneratorAnnotation
    import circtTests.tywavesTests.TywavesAnnotationCircuits.ParamCircuits._
    import circtTests.tywavesTests.TestUtils.{checkAnno, createExpected}

    val targetDir = os.pwd / "test_run_dir" / "TywavesAnnotationSpec" / "Chisel Constructs with Parameters Annotations"
    val args: Array[String] = Array("--target", "chirrtl", "--target-dir", targetDir.toString)

    it("should annotate a circuit with parameters") {
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitWithParams(8, 16))))
      // The file should include it
      val string = os.read(targetDir / "TopCircuitWithParams.fir")
      val expectedMatches = Seq(
        (createExpected("~TopCircuitWithParams\\|TopCircuitWithParams>uint", "UInt<8>", "IO"), 1),
        (createExpected("~TopCircuitWithParams\\|TopCircuitWithParams>sint", "SInt<16>", "IO"), 1),
        (createExpected("~TopCircuitWithParams\\|TopCircuitWithParams>bool", "Bool", "IO"), 1),
        (createExpected("~TopCircuitWithParams\\|TopCircuitWithParams>bits", "UInt<24>", "IO"), 1),
        (
          createExpected(
            "~TopCircuitWithParams\\|TopCircuitWithParams",
            "TopCircuitWithParams",
            params = Some(Seq(ClassParam("width1", "Int", Some("8")), ClassParam("width2", "Int", Some("16"))))
          ),
          1
        )
      )
      checkAnno(expectedMatches, string, includeConstructor = true)
    }

    it("should annotate variants of Parametrized module in a circuit") {
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitWithParamModules)))
      // The file should include it
      val string = os.read(targetDir / "TopCircuitWithParamModules.fir")
      countSubstringOccurrences(string, "\"class\":\"chisel3.tywaves.TywavesAnnotation\"") should be(4)
      countSubstringOccurrences(string, "\"typeName\":\"MyModule") should be(3)

      val expected1 = "\"typeName\":\"MyModule\"," + "\\s*\"params\":\\[\\s*\\{\\s*" +
        "\"name\":\"width\",\\s*\"typeName\":\"Int\",\\s*\"value\":\"8\"" +
        "\\s*\\}\\s*\\]\\s*\\}"
      (countSubstringOccurrences(string, expected1) should be(1)).withClue("Expected: " + expected1)
      val expected2 = "\"typeName\":\"MyModule\"," + "\\s*\"params\":\\[\\s*\\{\\s*" +
        "\"name\":\"width\",\\s*\"typeName\":\"Int\",\\s*\"value\":\"16\"" +
        "\\s*\\}\\s*\\]\\s*\\}"
      (countSubstringOccurrences(string, expected2) should be(1)).withClue("Expected: " + expected2)
      val expected3 = "\"typeName\":\"MyModule\"," + "\\s*\"params\":\\[\\s*\\{\\s*" +
        "\"name\":\"width\",\\s*\"typeName\":\"Int\",\\s*\"value\":\"32\"" +
        "\\s*\\}\\s*\\]\\s*\\}"
      (countSubstringOccurrences(string, expected3) should be(1)).withClue("Expected: " + expected3)
    }

    it("should annotate Parametrized bundles") {
      new ChiselStage(true).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitWithParamBundle)))
      // The file should include it
      val string = os.read(targetDir / "TopCircuitWithParamBundle.fir")
      // format: off
      val expectedMatches = Seq(
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>baseBundle", "BaseBundle", "IO",
          params = Some(Seq(ClassParam("n", "Int", Some(1.toString))))), 1),
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>baseBundle.b", "UInt<1>", "IO"), 1),
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>otherBundle", "OtherBundle", "IO",
          params = Some(Seq(
            ClassParam("a", "UInt", Some("IO\\[UInt<1>\\]")),
            ClassParam("b", "BaseBundle", Some("BaseBundle\\(n: 1\\)"))))), 1),
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>otherBundle.a", "UInt<1>", "IO"), 1),
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>otherBundle.b", "BaseBundle", "IO",
          params = Some(Seq(ClassParam("n", "Int", Some(1.toString))))), 1),
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>otherBundle.b.b", "UInt<1>", "IO"), 1),

        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>topBundle", "TopBundle", "IO",
          params = Some(Seq(
            ClassParam("a", "Bool", Some("IO\\[Bool\\]")),
            ClassParam("b", "String", Some("hello")),
            ClassParam("c", "Char", Some("c")),
            ClassParam("d", "Boolean", Some("true")),
            ClassParam("o", "OtherBundle", Some("OtherBundle\\(a: IO\\[UInt<1>\\], b: BaseBundle\\(n: 1\\)\\)"))  ))), 1),
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>topBundle.inner_a", "Bool", "IO"), 1),
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>topBundle.o", "OtherBundle", "IO",
            params = Some(Seq(ClassParam("a", "UInt", Some("IO\\[UInt<1>\\]")),
            ClassParam("b", "BaseBundle", Some("BaseBundle\\(n: 1\\)"))))), 1),
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>topBundle.o.a", "UInt<1>", "IO"), 1),
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>topBundle.o.b", "BaseBundle", "IO",
          params = Some(Seq(ClassParam("n", "Int", Some(1.toString))))), 1),
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>topBundle.o.b.b", "UInt<1>", "IO"), 1),

        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>caseClassBundle", "CaseClassExample", "IO",
          params = Some(Seq(
            ClassParam("a", "Int", Some("1")),
            ClassParam("o", "OtherBundle", Some("OtherBundle\\(a: IO\\[UInt<2>\\], b: BaseBundle\\(n: 1\\)\\)"))
          ))), 1),
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>caseClassBundle.o", "OtherBundle", "IO",
          params = Some(Seq(
            ClassParam("a", "UInt", Some("IO\\[UInt<2>\\]")),
            ClassParam("b", "BaseBundle", Some("BaseBundle\\(n: 1\\)"))))), 1),
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>caseClassBundle.o.a", "UInt<2>", "IO"), 1),
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>caseClassBundle.o.b", "BaseBundle", "IO",
          params = Some(Seq(ClassParam("n", "Int", Some(1.toString))))), 1),
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>caseClassBundle.o.b.b", "UInt<1>", "IO"), 1),
        (createExpected("~TopCircuitWithParamBundle\\|TopCircuitWithParamBundle>anonBundle", "AnonymousBundle", "IO"), 1),
      )
      // format: on
      checkAnno(expectedMatches, string)
    }

    it("should annotate Parametrized modules and bundles with scala classes as params") {
      new ChiselStage(true).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitWithParamScalaClasses)))
      // The file should include it
      val string = os.read(targetDir / "TopCircuitWithParamScalaClasses.fir")
      // format: off
      val expectedMatches = Seq(
        (createExpected("~TopCircuitWithParamScalaClasses\\|MyModule", "MyModule",
          params = Some(Seq(
            ClassParam("a", "MyScalaClass", Some("MyScalaClass\\(a: 1, b\\)")),
            ClassParam("b", "MyScalaCaseClass", Some("MyScalaCaseClass\\(a: 2, b: world\\)"))
          ))), 1),
        (createExpected("~TopCircuitWithParamScalaClasses\\|TopCircuitWithParamScalaClasses>bundle", "MyBundle", "IO",
          params = Some(Seq(
            ClassParam("a", "MyScalaClass", Some("MyScalaClass\\(a: 1, b\\)")),
            ClassParam("b", "MyScalaCaseClass", Some("MyScalaCaseClass\\(a: 2, b: world\\)"))
          ))), 1),
      )
      // format: on
      checkAnno(expectedMatches, string)
    }
  }

}
