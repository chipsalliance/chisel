// SPDX-License-Identifier: Apache-2.0

package firrtl.ir

/** Object containing the set of FIRRTL keywords that need to be escaped when used as identifiers.
  *
  * This includes keywords from the FIRRTL specification as well as additional keywords recognized
  * by the CIRCT FIRRTL parser (from FIRTokenKinds.def).
  */
object Keywords {

  /** Set of FIRRTL keywords that need to be escaped when used as identifiers. */
  val keywords: Set[String] = Set(
    // Top-level keywords
    "FIRRTL",
    "version",
    "circuit",
    "public",
    "module",
    "extmodule",
    "layer",
    "formal",
    "type",
    // Port and parameter keywords
    "input",
    "output",
    "parameter",
    "defname",
    // Layer keywords
    "bind",
    "inline",
    "enablelayer",
    "knownlayer",
    "layerblock",
    // Statement keywords
    "node",
    "wire",
    "reg",
    "regreset",
    "inst",
    "of",
    "mem",
    "skip",
    // Memory keywords
    "data-type",
    "depth",
    "read-latency",
    "write-latency",
    "read-under-write",
    "reader",
    "writer",
    "readwriter",
    "old",
    "new",
    "undefined",
    // Connect-like keywords
    "connect",
    "invalidate",
    "attach",
    "define",
    "propassign",
    // Conditional keywords
    "when",
    "else",
    "match",
    // Command keywords
    "stop",
    "force",
    "force_initial",
    "release",
    "release_initial",
    "printf",
    "fprintf",
    "fflush",
    "assert",
    "assume",
    "cover",
    "intrinsic",
    // Type keywords
    "const",
    "Clock",
    "Reset",
    "AsyncReset",
    "UInt",
    "SInt",
    "Analog",
    "Probe",
    "RWProbe",
    "flip",
    "Unknown",
    "Bool",
    "Fixed",
    "AnyRef",
    "Path",
    "Inst",
    "Domain",
    "Double",
    "String",
    "Integer",
    "List",
    // Expression keywords
    "mux",
    "read",
    "probe",
    "rwprobe",
    // PrimOp keywords
    "asUInt",
    "asSInt",
    "asClock",
    "asAsyncReset",
    "asReset",
    "cvt",
    "neg",
    "not",
    "andr",
    "orr",
    "xorr",
    "add",
    "sub",
    "mul",
    "div",
    "rem",
    "lt",
    "leq",
    "gt",
    "geq",
    "eq",
    "neq",
    "dshl",
    "dshr",
    "dshlw",
    "and",
    "or",
    "xor",
    "cat",
    "pad",
    "shl",
    "shr",
    "head",
    "tail",
    "bits",
    "integer_add",
    "integer_mul",
    "integer_shr",
    "integer_shl",
    "list_concat",
    "tagExtract",
    // Additional CIRCT keywords not in spec (from FIRTokenKinds.def)
    "case",
    "class",
    "cmem",
    "contract",
    "declgroup",
    "domain_define",
    "domains",
    "extclass",
    "false",
    "group",
    "infer",
    "instchoice",
    "intmodule",
    "invalid",
    "is",
    "mport",
    "object",
    "option",
    "rdwr",
    "ref",
    "requires",
    "reset",
    "simulation",
    "smem",
    "symbolic",
    "true",
    "with",
    "write",
    "unsafe_domain_cast"
  )

  /** Legalize a name by escaping it with backticks if it's a FIRRTL keyword.
    * 
    * @param name the name to legalize
    * @return the legalized name (with backticks if it's a keyword)
    */
  def legalize(name: String): String = {
    if (keywords.contains(name)) s"`$name`" else name
  }
}
