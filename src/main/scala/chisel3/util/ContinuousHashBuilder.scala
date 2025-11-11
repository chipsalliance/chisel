// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import scala.util.hashing.MurmurHash3

/** A utility class for continuously building hash values from integer inputs.
  * 
  * This class maintains an internal hash state that can be continuously updated
  * with new integer values. It uses MurmurHash3 for consistent, high-quality
  * hash distribution.
  * 
  * Example usage:
  * {{{
  * val hashBuilder = new ContinuousHashBuilder()
  * val hash1 = hashBuilder.addValue(42)
  * val hash2 = hashBuilder.addValue(123)
  * val finalHash = hashBuilder.getHash()
  * }}}
  */
class ContinuousHashBuilder(initialSeed: Int = 0x9e3779b9) {
  private var currentHash:  Int = initialSeed
  private var elementCount: Int = 0

  /** Add a new integer value to the hash and return the updated hash.
    * 
    * @param value the integer value to incorporate into the hash
    * @return the updated hash value
    */
  def addValue(value: Int): Int = {
    currentHash = MurmurHash3.mix(currentHash, value)
    elementCount += 1
    currentHash
  }

  /** Add multiple integer values to the hash.
    * 
    * @param values the sequence of integer values to incorporate
    * @return the updated hash value after adding all values
    */
  def addValues(values: Int*): Int = {
    values.foreach(addValue)
    currentHash
  }

  /** Get the current hash value without modifying the state.
    * 
    * @return the current hash value
    */
  def getHash(): Int = {
    MurmurHash3.finalizeHash(currentHash, elementCount)
  }

  /** Reset the hash builder to its initial state.
    * 
    * @param newSeed optional new seed value (defaults to original seed)
    */
  def reset(newSeed: Option[Int] = None): Unit = {
    currentHash = newSeed.getOrElse(initialSeed)
    elementCount = 0
  }

  /** Create a copy of this hash builder with the same state.
    * 
    * @return a new ContinuousHashBuilder with identical state
    */
  def copy(): ContinuousHashBuilder = {
    val newBuilder = new ContinuousHashBuilder(initialSeed)
    newBuilder.currentHash = this.currentHash
    newBuilder.elementCount = this.elementCount
    newBuilder
  }

  /** Get the number of values that have been added to this hash builder.
    * 
    * @return the count of values added
    */
  def getElementCount(): Int = elementCount
}

/** Companion object providing factory methods and utilities for ContinuousHashBuilder. */
object ContinuousHashBuilder {

  /** Create a new ContinuousHashBuilder with default settings.
    * 
    * @return a new ContinuousHashBuilder instance
    */
  def apply(): ContinuousHashBuilder = new ContinuousHashBuilder()

  /** Create a new ContinuousHashBuilder with a custom seed.
    * 
    * @param seed the initial seed value for the hash
    * @return a new ContinuousHashBuilder instance
    */
  def apply(seed: Int): ContinuousHashBuilder = new ContinuousHashBuilder(seed)

  /** Convenience method to compute a hash from a sequence of integers.
    * 
    * @param values the sequence of integer values to hash
    * @param seed optional seed value (defaults to standard seed)
    * @return the computed hash value
    */
  def hashSequence(values: Seq[Int], seed: Int = 0x9e3779b9): Int = {
    val builder = new ContinuousHashBuilder(seed)
    values.foreach(builder.addValue)
    builder.getHash()
  }

  /** Create a hash builder that starts with an existing hash state.
    * 
    * @param existingHash the hash value to start with
    * @param elementCount the number of elements that contributed to the existing hash
    * @return a new ContinuousHashBuilder with the specified state
    */
  def fromExistingHash(existingHash: Int, elementCount: Int): ContinuousHashBuilder = {
    val builder = new ContinuousHashBuilder()
    builder.currentHash = existingHash
    builder.elementCount = elementCount
    builder
  }
}
