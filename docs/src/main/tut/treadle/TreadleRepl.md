---
layout: docs
title:  "treadlerepl"
section: "treadle"
---

## TreadleRepl
The TreadleRepl is a character based debug facility for firrtl circuits. It has a host of features that let a user manual peek and poke values, advance the clock and set monitoring in a number of ways.

## Launching
The TreadleRepl can be launched from a terminal session by running the ./treadle.sh script.
To launch it for a local firrtl file myfile.fir use:
```
./treadle.sh -frfs myfile.fir
```

## Faster launching

The above script uses `sbt` to run and that can be a bit slow to start up.
You can build a executable jar for the TreadleRepl by executing
```$xslt
sbt assembly
```
This will create an executable jar in the utils/bin directory. In that directory there is
a script `treadle` that will launch that jar.
You can copy both the jar and script to your own local bin or make symbolic links to it.

## Command line options

There are many command line options available to see them all, add --help to the command line.

## The Repl commands
To get started at the prompt type the `help` command to see what the other commands are.

Currently the commands are

| command | description |
| ------- | ----------- |
| load fileName | load/replace the current firrtl file |
| script fileName | load a script from a text file |
| run [linesToRun&#124;all&#124;list&#124;reset] | run loaded script |
| vcd [load&#124;run&#124;list&#124;test&#124;help] | control vcd input file |
| record-vcd [<fileName<]&#124;[done] | treadle.vcd loaded script |
| symbol regex | show symbol information |
| watch [+&#124;-] regex [regex ...] | watch (+) or unwatch (-) signals |
| poke inputSymbol value | set an input port to the given integer value |
| force symbol value | hold a wire to value (use value clear to clear forced wire) |
| rpoke regex value | poke value into portSymbols that match regex |
| peek symbol [offset] | show the current value of the signal |
| rpeek regex | show the current value of symbols matching the regex |
| randomize | randomize all symbols except reset) |
| reset [numberOfSteps] | assert reset (if present) for numberOfSteps (default 1) |
| reset [b&#124;d&#124;x&#124;h] | Set the output radix to binary, decimal, or hex |
| step [numberOfSteps] | cycle the clock numberOfSteps (default 1) times, and show state |
| waitfor symbol value [maxNumberOfSteps] | wait for particular value (default 1) of symbol, up to maxNumberOfSteps (default 100) |
| depend [childrenOf&#124;parentsOf] signal [depth] &#124; depend compare symbol1 symbol2 | show dependency relationship to symbol or between two symbols |
| show [state&#124;inputs&#124;outputs&#124;clocks&#124;firrtl&#124;lofirrtl] | show useful things |
| display symbol[, symbol, ...] | show computation of symbols |
| info | show information about the circuit |
| walltime [advance] | show current wall time, or advance it |
| verbose [true&#124;false&#124;toggle] | set evaluator verbose mode (default toggle) during dependency evaluation |
| snapshot | save state of engine |
| restore | restore state of engine from snapshot file |
| waves symbolName ... | generate wavedrom json for viewing waveforms |
| history [pattern] | show command history, with optional regex pattern |
| help [markdown] | show repl commands (in markdown format if desired) |
| quit | exit the engine |