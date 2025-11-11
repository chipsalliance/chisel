// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import chisel3.util.ContinuousHashBuilder

class ContinuousHashBuilderSpec extends AnyFlatSpec with Matchers {

  behavior of "ContinuousHashBuilder"

  it should "create a hash builder with default seed" in {
    val builder = ContinuousHashBuilder()
    builder.getElementCount() should be(0)
    builder.getHash() should not be (0) // Should have some initial hash value
  }

  it should "create a hash builder with custom seed" in {
    val customSeed = 12345
    val builder = ContinuousHashBuilder(customSeed)
    builder.getElementCount() should be(0)
  }

  it should "continuously build hash with single values" in {
    val builder = ContinuousHashBuilder()

    val hash1 = builder.addValue(42)
    builder.getElementCount() should be(1)

    val hash2 = builder.addValue(123)
    builder.getElementCount() should be(2)

    // Hash should change with each addition
    hash1 should not be (hash2)

    val finalHash = builder.getHash()
    finalHash should not be (hash1)
    finalHash should not be (hash2)
  }

  it should "add multiple values at once" in {
    val builder = ContinuousHashBuilder()

    val hash = builder.addValues(1, 2, 3, 4, 5)
    builder.getElementCount() should be(5)

    val finalHash = builder.getHash()
    finalHash should not be (0)
  }

  it should "produce consistent hashes for same sequence" in {
    val builder1 = ContinuousHashBuilder()
    val builder2 = ContinuousHashBuilder()

    val values = Seq(10, 20, 30, 40, 50)

    values.foreach(builder1.addValue)
    values.foreach(builder2.addValue)

    builder1.getHash() should be(builder2.getHash())
  }

  it should "produce different hashes for different sequences" in {
    val builder1 = ContinuousHashBuilder()
    val builder2 = ContinuousHashBuilder()

    builder1.addValues(1, 2, 3)
    builder2.addValues(3, 2, 1)

    builder1.getHash() should not be (builder2.getHash())
  }

  it should "reset to initial state" in {
    val builder = ContinuousHashBuilder()
    val initialHash = builder.getHash()

    builder.addValues(1, 2, 3, 4, 5)
    builder.getElementCount() should be(5)

    builder.reset()
    builder.getElementCount() should be(0)
    builder.getHash() should be(initialHash)
  }

  it should "reset with new seed" in {
    val builder = ContinuousHashBuilder()
    builder.addValue(42)

    val newSeed = 99999
    builder.reset(Some(newSeed))
    builder.getElementCount() should be(0)

    // Should behave like a new builder with the new seed
    val newBuilder = ContinuousHashBuilder(newSeed)
    builder.getHash() should be(newBuilder.getHash())
  }

  it should "create a copy with same state" in {
    val builder = ContinuousHashBuilder()
    builder.addValues(10, 20, 30)

    val copy = builder.copy()
    copy.getHash() should be(builder.getHash())
    copy.getElementCount() should be(builder.getElementCount())

    // Modifying copy should not affect original
    copy.addValue(40)
    copy.getHash() should not be (builder.getHash())
  }

  it should "compute hash from sequence using companion object" in {
    val values = Seq(1, 2, 3, 4, 5)
    val hash = ContinuousHashBuilder.hashSequence(values)

    // Should match manual building
    val builder = ContinuousHashBuilder()
    values.foreach(builder.addValue)
    hash should be(builder.getHash())
  }

  it should "create builder from existing hash state" in {
    val originalBuilder = ContinuousHashBuilder()
    originalBuilder.addValues(1, 2, 3)
    val originalHash = originalBuilder.getHash()
    val originalCount = originalBuilder.getElementCount()

    val newBuilder = ContinuousHashBuilder.fromExistingHash(originalHash, originalCount)
    newBuilder.getHash() should be(originalHash)
    newBuilder.getElementCount() should be(originalCount)

    // Adding same value should produce same result
    originalBuilder.addValue(4)
    newBuilder.addValue(4)
    originalBuilder.getHash() should be(newBuilder.getHash())
  }

  it should "handle negative integers" in {
    val builder = ContinuousHashBuilder()
    val hash1 = builder.addValue(-42)
    val hash2 = builder.addValue(-123)

    hash1 should not be (hash2)
    builder.getElementCount() should be(2)
  }

  it should "handle zero values" in {
    val builder = ContinuousHashBuilder()
    val hash1 = builder.addValue(0)
    val hash2 = builder.addValue(0)

    // Even adding the same value should change the hash due to position
    hash1 should not be (hash2)
    builder.getElementCount() should be(2)
  }
}
