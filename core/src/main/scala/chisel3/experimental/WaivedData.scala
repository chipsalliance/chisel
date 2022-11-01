// SPDX-License-Identifier: Apache-2.0

package chisel3
package experimental

import chisel3.internal.prefix 
import chisel3.internal.sourceinfo.SourceInfo

final case class WaivedData[T <: Data](d: T, waivers: Set[Data])

object WaivedData {

  def waiveUnmatched[T <: Data](consumer: T, producer: T): (WaivedData[T], WaivedData[T]) = {
    val result = DataMirror.collectDeepOverAllForAny(Some((consumer: Data)), Some((producer: Data))) {
      case x@(Some(c), None) => x
      case x@(None, Some(p)) => x
    }
    println(result)
    val cWaived = result.map(_._1).flatten
    val pWaived = result.map(_._2).flatten
    (WaivedData(consumer, cWaived.toSet), WaivedData(producer, pWaived.toSet))
  }

  implicit class WaivableData[T <: Data](d: T) {
    def waive(fields: (T => Data)*): WaivedData[T] = WaivedData(d, fields.map(f => f(d)).toSet)

    def waiveAs[S <: Data](fields: (T => Data)*): WaivedData[S] = WaivedData(d, fields.map(f => f(d)).toSet).asInstanceOf[WaivedData[S]]

    def waiveAll[S <: Data](pf: PartialFunction[Data, Data]): WaivedData[T] = {
      val waivedMembers = DataMirror.collectDeep(d)(pf)
      WaivedData(d, waivedMembers.toSet)
    }
  }

  implicit class ConnectableForWaivedData[T <: Data](wd: WaivedData[T]) extends Connectable.ConnectableDocs {
    val consumer = wd.d
    val cWaivers = wd.waivers

    /** $colonLessEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    final def :<=(producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonLessEq, cWaivers, Set.empty[Data])
      }
    }

    /** $colonLessEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    final def :<=(pWaived: WaivedData[T])(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonLessEq, cWaivers, pWaived.waivers)
      }
    }

    /** $colonGreaterEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    final def :>=(producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonGreaterEq, cWaivers, Set.empty[Data])
      }
    }

    /** $colonGreaterEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    final def :>=(pWaived: WaivedData[T])(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonGreaterEq, cWaivers, pWaived.waivers)
      }
    }

    /** $colonLessGreaterEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    final def :<>=(producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        // cannot call :<= and :>= directly because otherwise prefix is called twice
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonLessGreaterEq, cWaivers, Set.empty[Data])
      }
    }

    /** $colonLessGreaterEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    final def :<>=(pWaived: WaivedData[T])(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonLessGreaterEq, cWaivers, pWaived.waivers)
      }
    }

    /** $colonHashEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection, all fields will be driving, none will be driven-to
      */
    final def :#=(producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonHashEq, cWaivers, Set.empty[Data])
      }
    }

    /** $colonHashEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection, all fields will be driving, none will be driven-to
      */
    final def :#=(pWaived: WaivedData[T])(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonHashEq, cWaivers, pWaived.waivers)
      }
    }

  }
}