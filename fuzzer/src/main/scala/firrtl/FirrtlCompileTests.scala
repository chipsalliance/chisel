// SPDX-License-Identifier: Apache-2.0

package firrtl.fuzzer

import com.pholser.junit.quickcheck.From
import com.pholser.junit.quickcheck.generator.{Generator, GenerationStatus}
import com.pholser.junit.quickcheck.random.SourceOfRandomness

import firrtl.{ChirrtlForm, CircuitState, LowFirrtlCompiler}
import firrtl.ir.Circuit

import org.junit.Assert
import org.junit.runner.RunWith

import java.io.{PrintWriter, StringWriter}

import edu.berkeley.cs.jqf.fuzz.Fuzz;
import edu.berkeley.cs.jqf.fuzz.JQF;

/** a GenMonad backed by [[com.pholser.junit.quickcheck.random.SourceOfRandomness SourceOfRandomness]]
  */
trait SourceOfRandomnessGen[A] {
  def apply(): A

  def flatMap[B](f: A => SourceOfRandomnessGen[B]): SourceOfRandomnessGen[B] =
    SourceOfRandomnessGen { f(apply())() }

  def map[B](f: A => B): SourceOfRandomnessGen[B] =
    SourceOfRandomnessGen { f(apply()) }

  def widen[B >: A]: SourceOfRandomnessGen[B] =
    SourceOfRandomnessGen { apply() }
}

object SourceOfRandomnessGen {
  implicit def sourceOfRandomnessGenGenMonadInstance(implicit r: SourceOfRandomness): GenMonad[SourceOfRandomnessGen] = new GenMonad[SourceOfRandomnessGen] {
    import scala.collection.JavaConverters.seqAsJavaListConverter
    type G[T] = SourceOfRandomnessGen[T]
    def flatMap[A, B](a: G[A])(f: A => G[B]): G[B] = a.flatMap(f)
    def map[A, B](a: G[A])(f: A => B): G[B] = a.map(f)
    def choose(min: Int, max: Int): G[Int] = SourceOfRandomnessGen {
      r.nextLong(min, max).toInt // use r.nextLong instead of r.nextInt because r.nextInt is exclusive of max
    }
    def oneOf[T](items: T*): G[T] = {
      val arr = seqAsJavaListConverter(items)
      const(arr.asJava).map(r.choose(_))
    }
    def const[T](c: T): G[T] = SourceOfRandomnessGen(c)
    def widen[A, B >: A](ga: G[A]): G[B] = ga.widen[B]
    def generate[A](ga: G[A]): A = ga.apply()
  }

  def apply[T](f: => T): SourceOfRandomnessGen[T] = new SourceOfRandomnessGen[T] {
    def apply(): T = f
  }
}

import ExprGen._
class FirrtlCompileCircuitGenerator extends SingleExpressionCircuitGenerator (
  ExprGenParams(
    maxDepth = 50,
    maxWidth = 31,
    generators = ExprGenParams.defaultGenerators
  )
)

@RunWith(classOf[JQF])
class FirrtlCompileTests {
  private val lowFirrtlCompiler = new LowFirrtlCompiler()
  private val header = "=" * 50 + "\n"
  private val footer = header
  private def message(c: Circuit, t: Throwable): String = {
    val sw = new StringWriter()
    val pw = new PrintWriter(sw)
    t.printStackTrace(pw)
    pw.flush()
    header + c.serialize + "\n" + sw.toString + footer
  }

  @Fuzz
  def compileSingleModule(@From(value = classOf[FirrtlCompileCircuitGenerator]) c: Circuit) = {
    compile(CircuitState(c, ChirrtlForm, Seq()))
  }

  // adapted from chisel3.Driver.execute and firrtl.Driver.execute
  def compile(c: CircuitState) = {
    val compiler = lowFirrtlCompiler
    try {
      val res = compiler.compile(c, Seq())
    } catch {
      case e: firrtl.CustomTransformException =>
        Assert.assertTrue(message(c.circuit, e.cause), false)
      case any : Throwable =>
        Assert.assertTrue(message(c.circuit, any), false)
    }
  }
}
