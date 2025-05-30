// SPDX-License-Identifier: Apache-2.0

package firrtlTests.annotationTests

import firrtl.annotations._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers
import org.json4s.convertToJsonInput

class LoadMemoryAnnotationSpec extends AnyFreeSpec with Matchers {
  "LoadMemoryAnnotation getFileName" - {
    "add name of subcomponent to file name when a memory was split" in {
      val lma = new LoadMemoryAnnotation(
        ComponentName("init_mem_subdata", ModuleName("b")),
        "somepath/init_mem",
        originalMemoryNameOpt = Some("init_mem")
      )

      lma.getFileName should be("somepath/init_mem_subdata")
    }
    "and do that properly when there are dots in earlier sections of the path" in {
      val lma = new LoadMemoryAnnotation(
        ComponentName("init_mem_subdata", ModuleName("b")),
        "./target/scala-2.12/test-classes/init_mem",
        originalMemoryNameOpt = Some("init_mem")
      )

      lma.getFileName should be("./target/scala-2.12/test-classes/init_mem_subdata")
    }
  }
  "LoadMemoryAnnotation should be correctly parsed from a string" in {
    val lma = new LoadMemoryAnnotation(
      ComponentName("ram", ModuleName("ModuleMem")),
      "CircuitMem.ModuleMem.ram.dat",
      hexOrBinary = MemoryLoadFileType.Binary,
      originalMemoryNameOpt = Some("memory")
    )

    val annoString = JsonProtocol.serializeTry(Seq(lma)).get
    val loadedAnnos = JsonProtocol.deserializeTry(annoString).get
    lma should equal(loadedAnnos.head)
  }
}
