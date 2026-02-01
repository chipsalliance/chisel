// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

private[chisel3] trait RecordAlias$Intf { self: RecordAlias.type =>
  def apply(id: String)(implicit info: SourceInfo): RecordAlias = _applyImpl(id)
  def apply(id: String, strippedSuffix: String)(implicit info: SourceInfo): RecordAlias =
    _applyWithSuffixImpl(id, strippedSuffix)
}
