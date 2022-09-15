# What benefits does Chisel offer over classic Hardware Description Languages? [closed]

This is an incredibly difficult question to answer for a number of reasons.
The Chisel community is attempting to unify on one concrete answer that is easy to understand.
There are two main angles:

1. Improved productivity through new language features and availability of libraries
1. Improved specialization due to the hardware-compiler structure

## Language Feature/Power Answer
Think about the following similar question, "What utility does Python provide over C?"
A response indicating no utility would likely bring up the following points:

1. Everything I can write in C I can write in Python.
Both have for, while, etc. and the code I write looks the same: `printf("hello\n")` vs. `print("hello")`
1. C has features that Python doesn't like inline assembly
1. Both are Turing complete

The problem with this line of reasoning is that it's ignoring the fact that Python provides
new programming paradigms that allow you to be more productive:

* Object oriented programming
* Functional programming
* Libraries

In effect, Python can be viewed as a more powerful language from a design productivity and code reuse perspective.
Language power is notoriously difficult to objectively evaluate.
Paul Graham describes this as the "Blub Paradox" in his "Beating the Averages" essay.
Graham's thesis is that an engineer proficient in a less powerful language cannot evaluate the utility of a more powerful language.

Put differently, the existence of the paradigms listed above does not force you to use them.
Hence, it's entirely possible to write Python that looks and feels just like C
(and use that for justification to dismiss Python altogether!).

Chisel is a domain specific language (DSL) for describing hardware circuits embedded in Scala.
By its very nature, it provides equivalent constructs to Verilog like module, Input/Output, Reg, etc.
Problematically for your question, this causes both the most basic Chisel examples to look exactly
like Verilog and also allows people to write Chisel that looks just like Verilog.
This is often used as an argument for dismissing Chisel in favor of Verilog.
However, this is analogous to making programming language choice based on the structure of "Hello World" examples.

That means that a better question is: "What can I build with Python that is insurmountably hard with C?"

Answering this question requires case studies and is (very quickly) out of the scope of "Hello World" comparisons.
Consider building machine learning libraries in C vs. Python, e.g., "How would I build Tensorflow in C?"
This would be incredibly difficult due to the abstractions that something like Tensorflow requires.

You can think about similar exemplar comparisons by looking at mature Chisel projects.
Two prominent examples are Rocket-Chip and Diplomacy.
The former is a generator of arbitrary System-on-Chip (SoC) designs.
The latter is used by Rocket Chip to deal with the problem of parameter negotiation---in an arbitrary
SoC I'd like to have the parameters of the N components I'm connecting together to be a
function of what I'm connecting (e.g., address widths, internal logic, coherency protocol).
Critically, both Rocket-Chip and Diplomacy can be used as libraries.
Concretely, this means that a user is very close to being able to "just import a RISC-V microprocessor"
in the same way that they "just import a graph library".

Due to language feature/power coupled with library availability, the real utility metrics become:

* How many lines of code do I need to describe some component?
* How powerful are the abstractions the language allows me to build?
* To what extent does the language enable code reuse?

For unsupported constructs, like negative edge triggered things and asynchronous resets
(both FIRRTL and Chisel now support asynchronous resets),
Chisel always provides an escape hatch kludge via Verilog blackboxes.
However, all of these are really just features looking for a developer.

## Hardware Compiler Answer

Chisel forms part of a Hardware Compiler Framework that looks very much like LLVM applied to hardware generation.
The Chisel-to-Verilog process forms part of a multi-stage compiler.
The "Chisel stage/front-end" compiles Chisel to a circuit intermediate representation called
FIRRTL (Flexible Intermediate Representation for RTL). "FIRRTL stage/mid-end" then optimizes
FIRRTL and applies user-custom transformations. Finally the "Verilog stage/back-end" emits Verilog based on the optimized FIRRTL.

While subtle, this compiler structure enables the following things:

* The front-end and back-ends are decoupled meaning that other front-ends and back-ends can be written.
A Verilog front-end exists through Yosys and other languages can directly target FIRRTL, e.g.,
Magma is it's own Chisel-like language in Python that can target FIRRTL.
New front-ends get all the benefits of existing mid-end optimizations and available back-ends.
New back-ends can also be written, e.g., a VHDL back-end just needs one motivated developer to write it.
* The introduction of a circuit IR enables automated specialization/transformation of circuits.
This has been exploited to transform circuits into FPGA optimized versions that run faster than
unoptimized versions (Midas and FireSim), enable hardware breakpoints/assertions (Dessert),
and to add run-time configurable fault injection capabilities (Chiffre).
Doing these optimizations directly on the Verilog would be insurmountably complex and brittle.

From this perspective, the question is closer to "What utility does C++ and LLVM provide over manually written assembly language?"

When discussing compilers, usually the question of Chisel (a hardware construction language)
vs. high level synthesis (HLS) comes up. The differentiating factor here is that Chisel is still,
fundamentally, a powerful language for describing circuits while HLS is a path for converting programs to circuits.

## Conclusion

Summarily, a better way of thinking about this question is, "What does Chisel enable beyond Verilog?"
That's a difficult question without a pithy answer and requires comparing the set of programming
paradigms both languages provide. Currently, and through my own experience,
the best way to tackle this is to do deep dives on mature Chisel code bases and try to use it yourself
(and withholding judgment for several months).
The difficulty with these approaches is that this takes time.
Additionally, the requisite skill sets for reading these code bases and making these types of judgments
are not commonly prevalent in hardware engineers. Hardware engineers are usually very proficient with C,
but may have never seen (in any depth) object oriented programming, functional programming,
or written complex projects exploiting modern software engineering principles.
This tends to set up biases where people are looking for reasons to dismiss Chisel (or similar languages) as opposed to reasons to use it.
