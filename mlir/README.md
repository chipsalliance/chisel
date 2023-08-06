# MLIR, CIRCT and CIRCEL

Chisel uses CIRCT to generate Verilog, which is built on top of MLIR and the LLVM project infrastructure.

## Managing the CIRCT Dependency

Instead of including CIRCT as a submodule, we use a bespoke solution implemented in `mlir/support/circt-helper` to manage our dependency on CIRCT. If you ever wish to update CIRCT, you can simply run `mlir/support/circt-helper update-circt <target commit-ish>` where `<target commit-ish>` is the the `git` commit-ish you wish to update to (most likely `origin/main`). For more information on what `circt-helper` does and why we use it, refer to the help text of `circt-helper` (or the script itself). 
