// SPDX-License-Identifier: Apache-2.0

package firrtlTests.options

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import java.util.ServiceLoader

import firrtl.options.{RegisteredLibrary, RegisteredTransform, ShellOption}
import firrtl.passes.Pass
import firrtl.ir.Circuit
import firrtl.annotations.NoTargetAnnotation
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

case object HelloAnnotation extends NoTargetAnnotation

class FooTransform extends Pass with RegisteredTransform {
  def run(c: Circuit): Circuit = c

  val options = Seq(
    new ShellOption[Unit](longOption = "hello", toAnnotationSeq = _ => Seq(HelloAnnotation), helpText = "Hello option")
  )

}

class BarLibrary extends RegisteredLibrary {
  def name: String = "Bar"

  val options = Seq(
    new ShellOption[Unit](longOption = "world", toAnnotationSeq = _ => Seq(HelloAnnotation), helpText = "World option")
  )
}

class RegistrationSpec extends AnyFlatSpec with Matchers {

  behavior.of("RegisteredTransform")

  it should "FooTransform should be discovered by Java.util.ServiceLoader" in {
    val iter = ServiceLoader.load(classOf[RegisteredTransform]).iterator()
    val transforms = scala.collection.mutable.ArrayBuffer[RegisteredTransform]()
    while (iter.hasNext) {
      transforms += iter.next()
    }
    transforms.map(_.getClass.getName) should contain("firrtlTests.options.FooTransform")
  }

  behavior.of("RegisteredLibrary")

  it should "BarLibrary be discovered by Java.util.ServiceLoader" in {
    val iter = ServiceLoader.load(classOf[RegisteredLibrary]).iterator()
    val transforms = scala.collection.mutable.ArrayBuffer[RegisteredLibrary]()
    while (iter.hasNext) {
      transforms += iter.next()
    }
    transforms.map(_.getClass.getName) should contain("firrtlTests.options.BarLibrary")
  }
}
