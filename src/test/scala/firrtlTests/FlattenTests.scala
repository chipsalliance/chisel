// See LICENSE for license details.

package firrtlTests

import org.scalatest.FlatSpec
import org.scalatest.Matchers
import org.scalatest.junit.JUnitRunner
import firrtl.ir.Circuit
import firrtl.Parser
import firrtl.passes.PassExceptions
import firrtl.annotations.{Annotation, CircuitName, ComponentName, ModuleName, Named}
import firrtl.transforms.{FlattenAnnotation, Flatten}
import logger.{LogLevel, Logger}
import logger.LogLevel.Debug


/**
 * Tests deep inline transformation
 */
class FlattenTests extends LowTransformSpec {
  def transform = new Flatten
  def flatten(mod: String): Annotation = {
    val parts = mod.split('.')
    val modName = ModuleName(parts.head, CircuitName("Top")) // If this fails, bad input
    val name = if (parts.size == 1) modName else ComponentName(parts.tail.mkString("."), modName)
    FlattenAnnotation(name)
  }
  
  "The modules inside Top " should "be inlined" in {
     val input =
        """circuit Top :
          |  module Top :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    inst i of Inline1
          |    i.a <= a
          |    b <= i.b
          |  module Inline1 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    b <= a""".stripMargin
     val check =
        """circuit Top :
          |  module Top :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    wire i$a : UInt<32>
          |    wire i$b : UInt<32>
          |    i$b <= i$a
          |    b <= i$b
          |    i$a <= a
          |  module Inline1 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    b <= a""".stripMargin
     execute(input, check, Seq(flatten("Top")))
  }
  
  "Two instances of the same module inside Top " should "be inlined" in {
     val input =
        """circuit Top :
          |  module Top :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    inst i1 of Inline1
          |    inst i2 of Inline1
          |    i1.a <= a
          |    node tmp = i1.b
          |    i2.a <= tmp
          |    b <= i2.b
          |  module Inline1 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    b <= a""".stripMargin
     val check =
        """circuit Top :
          |  module Top :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    wire i1$a : UInt<32>
          |    wire i1$b : UInt<32>
          |    i1$b <= i1$a
          |    wire i2$a : UInt<32>
          |    wire i2$b : UInt<32>
          |    i2$b <= i2$a
          |    node tmp = i1$b
          |    b <= i2$b
          |    i1$a <= a
          |    i2$a <= tmp
          |  module Inline1 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    b <= a""".stripMargin
     execute(input, check, Seq(flatten("Top")))
  }

  "The module instance i in Top " should "be inlined" in {
     val input =
        """circuit Top :
          |  module Top :
          |    input a : UInt<32>
          |    input na : UInt<32>
          |    output b : UInt<32>
          |    output nb : UInt<32>
          |    inst i of Inline1
          |    inst ni of NotInline1
          |    i.a <= a
          |    b <= i.b
          |    ni.a <= na
          |    nb <= ni.b
          |  module NotInline1 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    inst i of Inline2
          |    i.a <= a 
          |    b <= i.a 
          |  module Inline1 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    inst i of Inline2
          |    i.a <= a 
          |    b <= i.a 
          |  module Inline2 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    b <= a""".stripMargin
     val check =
        """circuit Top :
          |  module Top :
          |    input a : UInt<32>
          |    input na : UInt<32>
          |    output b : UInt<32>
          |    output nb : UInt<32>
          |    wire i$a : UInt<32>
          |    wire i$b : UInt<32>
          |    wire i$i$a : UInt<32>
          |    wire i$i$b : UInt<32>
          |    i$i$b <= i$i$a
          |    i$b <= i$i$a
          |    i$i$a <= i$a
          |    inst ni of NotInline1
          |    b <= i$b
          |    nb <= ni.b         
          |    i$a <= a
          |    ni.a <= na
          |  module NotInline1 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    inst i of Inline2
          |    b <= i.a 
          |    i.a <= a 
          |  module Inline1 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    inst i of Inline2 
          |    b <= i.a 
          |    i.a <= a
          |  module Inline2 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    b <= a""".stripMargin
     execute(input, check, Seq(flatten("Top.i")))
  }
  "The module Inline1" should "be inlined" in {
    val input =
        """circuit Top :
          |  module Top :
          |    input a : UInt<32>
          |    input na : UInt<32>
          |    output b : UInt<32>
          |    output nb : UInt<32>
          |    inst i of Inline1
          |    inst ni of NotInline1
          |    i.a <= a
          |    b <= i.b
          |    ni.a <= na
          |    nb <= ni.b
          |  module NotInline1 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    inst i of Inline2
          |    i.a <= a 
          |    b <= i.a 
          |  module Inline1 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    inst i of Inline2
          |    i.a <= a 
          |    b <= i.a 
          |  module Inline2 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    b <= a""".stripMargin
     val check =
        """circuit Top :
          |  module Top :
          |    input a : UInt<32>
          |    input na : UInt<32>
          |    output b : UInt<32>
          |    output nb : UInt<32>
          |    inst i of Inline1
          |    inst ni of NotInline1
          |    b <= i.b
          |    nb <= ni.b
          |    i.a <= a
          |    ni.a <= na
          |  module NotInline1 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    inst i of Inline2
          |    b <= i.a 
          |    i.a <= a 
          |  module Inline1 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    wire i$a : UInt<32>
          |    wire i$b : UInt<32>
          |    i$b <= i$a 
          |    b <= i$a 
          |    i$a <= a
          |  module Inline2 :
          |    input a : UInt<32>
          |    output b : UInt<32>
          |    b <= a""".stripMargin
     execute(input, check, Seq(flatten("Inline1")))
  }
}
