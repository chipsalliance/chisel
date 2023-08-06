# handshake-runner

## SYNOPSIS

| **handshake-runner** \[_options_] \[_filename_] \[_arguments_]

## DESCRIPTION

This application executes a function in the given MLIR module. Arguments to the 
function are passed on the command line and results are returned on stdout. 
Memref types are specified as a comma-separated list of values. This particular 
tool is use to check the validity of Standard-to-Handshake conversion in CIRCT.

### Example

The following MLIR module first convert to Handshake IR with the circt-opt tool 
as `circt-opt -create-dataflow <file-name>`. And then the converted handshake IR
can be run by the handshake-runner with a command line argument "2" as 
`handshake-runner <file-name> 2` and produces the expected result "1".
```
module {
  func @main(%arg0: index) -> (i8) {
    %0 = alloc() : memref<10xi8>
    %c1 = constant 1 : i8
         store %c1, %0[%arg0] : memref<10xi8>
         %1 = load %0[%arg0] : memref<10xi8>
         return %1 : i8
  }
}
```

### Options

-h, --help

: Prints brief usage information.

--runStats

: Print Execution Statistics

--toplevelFunction=`<string>`

: The toplevel function to execute

-v, --version

: Prints the current version number.

## EXIT STATUS

If handshake-runner succeeds, it will exit with 0.  Otherwise, if an  error
occurs, it will exit with a non-zero value.

## BUGS

See GitHub Issues: <https://github.com/llvm/circt/issues>

## AUTHOR

Maintained by the CIRCT Team (https://circt.org/).

## SEE ALSO

[circt-opt](circt-opt.md)


