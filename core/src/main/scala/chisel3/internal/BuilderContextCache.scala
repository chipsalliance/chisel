// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import scala.collection.mutable

private[chisel3] object BuilderContextCache {

  /** Users of the [[BuilderContextCache]] must use a subclass of this type as a map key */
  abstract class Key[A]

  private[internal] def empty = new BuilderContextCache
}

import BuilderContextCache.Key

/** Internal data structure for caching things during elaboration */
private[chisel3] class BuilderContextCache private () {
  private val cache = mutable.HashMap.empty[Key[_], Any]

  /** Return the value associated with a key if present in the cache */
  def get[A](key: Key[A]): Option[A] = cache.get(key).map(_.asInstanceOf[A])

  /** Returns the value associated with a key, or a default value if the key is not contained in the map. */
  def getOrElse[A](key: Key[A], default: => A): A = cache.getOrElse(key, default).asInstanceOf[A]

  /** If a given key is already in the map, return the value
    *
    * Otherwise, update the map with the default value and return it.
    */
  def getOrElseUpdate[A](key: Key[A], default: => A): A = cache.getOrElseUpdate(key, default).asInstanceOf[A]

  /** Adds a new key/value pair to this map and optionally returns previously bound value. */
  def put[A](key: Key[A], value: A): Option[A] = cache.put(key, value).map(_.asInstanceOf[A])
}
