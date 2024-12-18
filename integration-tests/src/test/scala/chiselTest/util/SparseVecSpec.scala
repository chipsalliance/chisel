// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util.{log2Up, Counter, SparseVec}
import chisel3.util.SparseVec.{DefaultValueBehavior, Lookup, OutOfBoundsBehavior}
import chiselTests.{ChiselFlatSpec, Utils}
import _root_.circt.stage.ChiselStage
import java.util.ResourceBundle

/** Tester that checks that a a [[SparseVec]] behaves _exactly_ like a dynamic
  * index into a [[DontCare]]-initialized dense [[Vec]].  This checks
  * out-of-bounds behavior by indexing to the maximum addressable size of both
  * representations.  E.g., if the size is 3, this will check indices `[0, 1, 2,
  * 3]`.
  *
  * @param size the size of the vecs
  * @param tpe the type of the vecs
  * @param mapping a mapping of index to value
  */
class SparseVecDynamicIndexEquivalenceTest(
  size:    Int,
  tpe:     UInt,
  mapping: Seq[(Int, UInt)],
  debug:   Boolean = false)
    extends BasicTester {

  // The number of indices that needs to be checked.  This is larger than `size`
  // if `size` is not a power of 2.  This is done to check out-of-bounds
  // behavior.
  private val paddedSize = BigInt(log2Up(size)).pow(2).toInt

  // This is the reference vector that will be dynamically indexed into.
  // Initialize all elements to DontCare.  Then set specific ones using the
  // provided expected mapping.
  private val denseVec = {
    val w = Wire(Vec(size, tpe))
    w.foreach(_ := DontCare)
    mapping.foreach {
      case (index, data) =>
        w(index) := data
    }
    w
  }

  // Create a wire SparseVec and initialize it to the values in the mapping.
  private val sparseVec = Wire(
    new SparseVec(size, tpe, mapping.map(_._1), DefaultValueBehavior.DynamicIndexEquivalent, OutOfBoundsBehavior.First)
  )
  sparseVec.elements.values.zip(mapping.map(_._2)).foreach { case (a, b) => a :<>= b }

  // Access the dense vector and the sparse vector, using all of the access
  // types, and make sure that the results are exactly the same.
  private val (index, wrap) = Counter(0 until paddedSize)
  private val failed = RegInit(Bool(), false.B)
  private val reference = denseVec(index)
  private val sparseVecResults = Seq(Lookup.Binary, Lookup.OneHot, Lookup.IfElse).map(sparseVec(index, _))
  if (debug) {
    when(RegNext(reset.asBool)) {
      printf("index, dense, binary, onehot, ifelse\n")
    }
    printf("%x: %x, %x, %x, %x", index, reference, sparseVecResults(0), sparseVecResults(1), sparseVecResults(2))
  }
  when(sparseVecResults.map(_ =/= reference).reduce(_ || _)) {
    failed := true.B
    if (debug)
      printf(" <-- error")
    else
      assert(false.B)
  }
  if (debug)
    printf("\n")

  when(wrap) {
    stop()
  }
}

/** This test checks that a [[SparseVec]] returns expected values.  A
  * [[SparseVec]] of size, type, and configuration parameters is created and
  * initialized with a mapping.  It is then checked against an expected sequence
  * of index--value pairs.  Using an expected sequence that checks indices not
  * in the mapping, either default or out-of-bounds behaviors can be checked.
  *
  * @param size the size of the SparseVec
  * @param tpe the element type of the SparseVec
  * @param defaultValueBehavior the default value behavior
  * @param outOfBoundsBehavior the out-of-bounds behavior
  * @param mapping the index--value mapping that is used to initialize the vec
  * @param expected the expected values that are read out of the vec at each index
  */
class SparseVecTest(
  size:                 Int,
  tpe:                  UInt,
  defaultValueBehavior: DefaultValueBehavior.Type,
  outOfBoundsBehavior:  OutOfBoundsBehavior.Type,
  mapping:              Seq[(Int, UInt)],
  expected:             Seq[(Int, Data)],
  debug:                Boolean = false)
    extends BasicTester {
  // Create a wire SparseVec and initialize it to the values in the mapping.
  private val sparseVec = Wire(new SparseVec(size, tpe, mapping.map(_._1), defaultValueBehavior, outOfBoundsBehavior))
  sparseVec.elements.values.zip(mapping.map(_._2)).foreach { case (a, b) => a :<>= b }

  class TestBundle extends Bundle {
    val index = UInt()
    val value = UInt()
  }

  val tests = Wire(
    Vec(
      expected.size,
      new TestBundle
    )
  )
  expected.zipWithIndex.foreach {
    case ((index, value), testNumber) =>
      tests(testNumber).index := index.U
      tests(testNumber).value := value
  }

  // Access the dense vector and the sparse vector, using all of the access
  // types, and make sure that the results are exactly the same.
  private val (index, wrap) = Counter(0 until tests.size)
  private val failed = RegInit(Bool(), false.B)
  private val reference = tests(index)
  private val sparseVecResults = Seq(Lookup.Binary, Lookup.OneHot, Lookup.IfElse).map(sparseVec(reference.index, _))
  if (debug) {
    when(RegNext(reset.asBool)) {
      printf("index, dense, binary, onehot, ifelse\n")
    }
    printf(
      "%x: %x, %x, %x, %x",
      reference.index,
      reference.value,
      sparseVecResults(0),
      sparseVecResults(1),
      sparseVecResults(2)
    )
  }
  when(sparseVecResults.map(_ =/= reference.value).reduce(_ || _)) {
    failed := true.B
    if (debug)
      printf(" <-- error")
    else
      assert(false.B)
  }
  if (debug)
    printf("\n")

  when(wrap) {
    when(RegNext(true.B)) {
      stop()
    }
  }

}

class SparseVecSpec extends ChiselFlatSpec with Utils {
  "SparseVec equivalence to Dynamic Index" should "work for a complete user-specified mapping" in {
    assertTesterPasses(
      new SparseVecDynamicIndexEquivalenceTest(
        4,
        UInt(3.W),
        Seq(
          0 -> 1.U,
          1 -> 2.U,
          2 -> 3.U,
          3 -> 4.U
        )
      )
    )
  }

  it should "work for a mapping that includes default values" in {
    assertTesterPasses(
      new SparseVecDynamicIndexEquivalenceTest(
        4,
        UInt(3.W),
        Seq(
          0 -> 1.U,
          1 -> 2.U,
          3 -> 4.U
        )
      )
    )
  }

  it should "work for a mapping that includes out-of-bounds accesses" in {
    assertTesterPasses(
      new SparseVecDynamicIndexEquivalenceTest(
        3,
        UInt(3.W),
        Seq(
          0 -> 1.U,
          1 -> 2.U,
          2 -> 3.U
        )
      )
    )
  }

  it should "work for a mapping that includes out-of-bounds accesses and no zeroth element" in {
    assertTesterPasses(
      new SparseVecDynamicIndexEquivalenceTest(
        3,
        UInt(3.W),
        Seq(
          1 -> 2.U,
          2 -> 3.U
        )
      )
    )
  }

  "SparseVec" should "work for a complete user-specified mapping" in {
    val mapping = Seq(
      0 -> 1.U,
      1 -> 2.U,
      2 -> 3.U,
      3 -> 4.U
    )
    assertTesterPasses(
      new SparseVecTest(
        4,
        UInt(3.W),
        DefaultValueBehavior.Indeterminate,
        OutOfBoundsBehavior.Indeterminate,
        mapping,
        expected = mapping
      )
    )
  }

  // This test is only checking that the indeterminate values didn't screw
  // anything up.  We can't actually check for an indeterminate value as it
  // could be anything.
  it should "work for a mapping that includes default values with indeterminate behavior" in {
    val mapping = Seq(
      0 -> 1.U,
      1 -> 2.U,
      3 -> 4.U
    )
    assertTesterPasses(
      new SparseVecTest(
        4,
        UInt(3.W),
        DefaultValueBehavior.Indeterminate,
        OutOfBoundsBehavior.Indeterminate,
        mapping,
        expected = mapping
      )
    )
  }

  it should "work for a mapping that includes default values" in {
    val mapping = Seq(
      0 -> 1.U,
      1 -> 2.U,
      3 -> 4.U
    )
    assertTesterPasses(
      new SparseVecTest(
        4,
        UInt(3.W),
        DefaultValueBehavior.UserSpecified(7.U),
        OutOfBoundsBehavior.Indeterminate,
        mapping,
        expected = mapping :+ (2 -> 7.U)
      )
    )
  }

  // As above, there's nothing to test here other than the values put in we get
  // out.
  it should "work for a mapping that includes indeterminate out-of-bounds behvaior" in {
    val mapping = Seq(
      0 -> 1.U,
      1 -> 2.U,
      2 -> 3.U
    )
    assertTesterPasses(
      new SparseVecTest(
        3,
        UInt(3.W),
        DefaultValueBehavior.Indeterminate,
        OutOfBoundsBehavior.Indeterminate,
        mapping,
        expected = mapping
      )
    )
  }

  it should "work for a mapping that includes \"first\" out-of-bounds behavior" in {
    val mapping = Seq(
      0 -> 1.U,
      1 -> 2.U,
      2 -> 3.U
    )
    assertTesterPasses(
      new SparseVecTest(
        3,
        UInt(3.W),
        DefaultValueBehavior.Indeterminate,
        OutOfBoundsBehavior.First,
        mapping,
        expected = mapping :+ (3 -> mapping(0)._2)
      )
    )
  }

  it should "work for an empty mapping" in {
    val mapping = Seq.empty[(Int, UInt)]
    assertTesterPasses(
      new SparseVecTest(
        2,
        UInt(3.W),
        DefaultValueBehavior.UserSpecified(7.U),
        OutOfBoundsBehavior.First,
        mapping,
        expected = mapping ++ Seq(0 -> 7.U, 1 -> 7.U)
      )
    )
  }

  it should "work for a size-zero vec" in {
    val mapping = Seq.empty[(Int, UInt)]
    assertTesterPasses(
      new SparseVecTest(
        0,
        UInt(3.W),
        DefaultValueBehavior.UserSpecified(7.U),
        OutOfBoundsBehavior.Indeterminate,
        mapping,
        expected = mapping ++ Seq(0 -> 7.U)
      )
    )
  }

  "SparseVec error behavior" should "disallow indices large than the size" in {
    val exception = intercept[IllegalArgumentException] {
      ChiselStage.convert(new Module {
        new SparseVec(1, UInt(1.W), Seq(0, 1))
      })
    }
    exception.getMessage should include("the SparseVec indices size (2) must be <= the SparseVec size (1)")
  }

  it should "disallow non-unique indices" in {
    val exception = intercept[ChiselException] {
      ChiselStage.convert(new Module {
        new SparseVec(2, UInt(1.W), Seq(0, 0))
      })
    }
    exception.getMessage should include("Non-unique indices in SparseVec, got duplicates 0")
  }

  it should "disallow a SparseVec write" in {
    val exception = intercept[ChiselException] {
      ChiselStage.convert(new Module {
        val vec = Wire(new SparseVec(2, UInt(1.W), Seq(0, 1)))
        vec(0.U(1.W)) := 1.U
      })
    }
    exception.getMessage should include("ReadOnlyModule cannot be written")
  }

}
