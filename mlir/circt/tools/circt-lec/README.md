# circt-lec
### a Logical Equivalence Checking tool
#### Building
circt-lec depends on the Z3 theorem prover version 4.8.11 or newer: the build
system will attempt to deduce its location. A custom install directory can
otherwise be specified with the `Z3_DIR` CMake option, in which case it should
also include a CMake package configuration file like `Z3Config.cmake`.

To avoid building the tool set the `CIRCT_LEC_DISABLE` CMake option on.

#### Usage
```circt-lec [options] <input file> [input file]```

The tool will compare circuits from the two input files; if no second file is
specified, both circuits will be parsed from the first and only file.

The circuits to be compared can be specified as their module's name with the
`c1` and `c2` options, otherwise the first module in the appropriate file will
be selected.

`comb` operations are currently supported only on binary state logic.

##### Command-line options
- `--c1=<module name>` specifies a module name for the first circuit
- `--c2=<module name>` specifies a module name for the second circuit
- `-v` turns on printing verbose information about execution
- `-s` turns on printing statistics about the execution of the logical engine
- `-debug` turns on printing debug information
- `-debug-only=<component list>` only prints debug information for the specified
  components (among `lec-exporter`, `lec-solver`, `lec-circuit`)

#### Developement
##### Regression testing
The tool can be tested by running the following command from the circt directory
when built with the default paths:
```./llvm/build/./bin/llvm-lit -vs build/integration_test/circt-lec```
