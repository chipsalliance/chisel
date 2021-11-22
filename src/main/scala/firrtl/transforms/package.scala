// SPDX-License-Identifier: Apache-2.0

package firrtl

package object transforms {
  @deprecated("Replaced by LegalizeClocksAndAsyncResetsTransform", "FIRRTL 1.4.0")
  type LegalizeClocksTransform = LegalizeClocksAndAsyncResetsTransform
  @deprecated("Replaced by LegalizeClocksAndAsyncResetsTransform", "FIRRTL 1.4.0")
  val LegalizeClocksTransform = LegalizeClocksAndAsyncResetsTransform
}
