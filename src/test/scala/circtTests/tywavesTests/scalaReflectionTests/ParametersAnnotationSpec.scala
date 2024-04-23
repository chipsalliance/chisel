package circtTests.tywavesTests.scalaReflectionTests

import chisel3.tywaves.{ClassParam, TywavesChiselAnnotation}
import chisel3.util.{SRAM, SRAMInterface}
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

}
