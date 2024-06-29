// SPDX-License-Identifier: Apache-2.0

package svsim

import java.util.concurrent.ConcurrentLinkedQueue

/**
  * A simple wrapper for a thread-safe queue
  */
final class SyncQueue[A] extends Iterable[A] {
  private val queue = new ConcurrentLinkedQueue[A]()

  def enqueue(elem: A): Unit = queue.add(elem)

  def dequeue(): A = queue.remove()

  def iterator: Iterator[A] = {
    new Iterator[A] {
      def hasNext: Boolean = !queue.isEmpty
      def next():  A = dequeue()
    }
  }

  def clear(): Unit = queue.clear()
}

object SyncQueue {
  def empty[A] = new SyncQueue[A]()
}
