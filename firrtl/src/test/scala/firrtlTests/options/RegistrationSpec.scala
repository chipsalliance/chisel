// SPDX-License-Identifier: Apache-2.0

package firrtlTests.options

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import java.util.ServiceLoader

import firrtl.options.{RegisteredLibrary, ShellOption}
import firrtl.ir.Circuit
import firrtl.annotations.NoTargetAnnotation
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

case object HelloAnnotation extends NoTargetAnnotation

class BarLibrary extends RegisteredLibrary {
  def name: String = "Bar"

  val options = Seq(
    new ShellOption[Unit](longOption = "world", toAnnotationSeq = _ => Seq(HelloAnnotation), helpText = "World option")
  )
}

class RegistrationSpec extends AnyFlatSpec with Matchers {

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
