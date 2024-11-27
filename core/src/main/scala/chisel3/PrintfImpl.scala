// SPDX-License-Identifier: Apache-2.0

package chisel3

private[chisel3] trait PrintfImpl {

  /** Helper for packing escape characters */
  private[chisel3] def format(formatIn: String): String = {
    require(formatIn.forall(c => c.toInt > 0 && c.toInt < 128), "format strings must comprise non-null ASCII values")
    def escaped(x: Char) = {
      require(x.toInt >= 0, s"char ${x} to Int ${x.toInt} must be >= 0")
      if (x == '"' || x == '\\') {
        s"\\${x}"
      } else if (x == '\n') {
        "\\n"
      } else if (x == '\t') {
        "\\t"
      } else {
        require(
          x.toInt >= 32,
          s"char ${x} to Int ${x.toInt} must be >= 32"
        ) // TODO \xNN once FIRRTL issue #59 is resolved
        x
      }
    }
    formatIn.map(escaped).mkString("")
  }
}