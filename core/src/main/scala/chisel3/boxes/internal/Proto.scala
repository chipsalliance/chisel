// SPDX-License-Identifier: Apache-2.0

package chisel3.boxes.internal

// Must be a case class because we want structural equality
case class Proto[+T](protoOpt: Option[T], protoMap: Option[Map[String, Any]])