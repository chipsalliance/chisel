// SPDX-License-Identifier: Apache-2.0

package svsim

import java.util.concurrent.ConcurrentLinkedQueue

/** A simple wrapper for a thread-safe queue
  */
final class SyncQueue[A] extends Iterable[A] {
  private val queue = new ConcurrentLinkedQueue[A]()

  @inline def enqueue(elem: A): Unit = queue.add(elem)

  @inline def dequeue(): A = queue.remove()

  @inline def addOne(elem: A): this.type = {
    enqueue(elem)
    this
  }

  def addAll(elems: IterableOnce[A]): this.type = {
    elems.iterator.foreach(enqueue)
    this
  }

  @inline def peekHead: A = queue.peek()

  def iterator: Iterator[A] = {
    new Iterator[A] {
      def hasNext: Boolean = !queue.isEmpty
      def next():  A = dequeue()
    }
  }

  def clear(): Unit = queue.clear()

  @inline def length: Int = queue.size()

}

object SyncQueue {
  def empty[A] = new SyncQueue[A]()
}
