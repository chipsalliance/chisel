---
layout: docs
title:  "Introduction"
section: "chisel3"
---
This document is a tutorial introduction to _Chisel_ (Constructing
Hardware In a Scala Embedded Language).  Chisel is a hardware
construction language embedded in the high-level programming language
Scala.  At some point we will provide a proper reference manual, in
addition to more tutorial examples.  In the meantime, this document
along with a lot of trial and error should set you on your way to
using Chisel. _Chisel is really only a set of special class
definitions, predefined objects, and usage conventions within Scala,
so when you write Chisel you are actually writing a Scala
program that constructs a hardware graph._  However, for the tutorial we don't presume that you
understand how to program in Scala.  We will point out necessary Scala
features through the Chisel examples we give, and significant hardware
designs can be completed using only the material contained herein.
But as you gain experience and want to make your code simpler or more
reusable, you will find it important to leverage the underlying power
of the Scala language. We recommend you consult one of the excellent
Scala books to become more expert in Scala programming.

>Through the tutorial, we format commentary on our design choices as in
this paragraph.  You should be able to skip the commentary sections
and still fully understand how to use Chisel, but we hope you'll find
them interesting.

>We were motivated to develop a new hardware language by years of
struggle with existing hardware description languages in our research
projects and hardware design courses.  _Verilog_ and _VHDL_ were developed
as hardware _simulation_ languages, and only later did they become
a basis for hardware _synthesis_.  Much of the semantics of these
languages are not appropriate for hardware synthesis and, in fact,
many constructs are simply not synthesizable.  Other constructs are
non-intuitive in how they map to hardware implementations, or their
use can accidentally lead to highly inefficient hardware structures.
While it is possible to use a subset of these languages and still get
acceptable results, they nonetheless present a cluttered and confusing
specification model, particularly in an instructional setting.

>However, our strongest motivation for developing a new hardware
language is our desire to change the way that electronic system design
takes place.  We believe that it is important to not only teach
students how to design circuits, but also to teach them how to design
*circuit generators* ---programs that automatically generate
designs from a high-level set of design parameters and constraints.
Through circuit generators, we hope to leverage the hard work of
design experts and raise the level of design abstraction for everyone.
To express flexible and scalable circuit construction, circuit
generators must employ sophisticated programming techniques to make
decisions concerning how to best customize their output circuits
according to high-level parameter values and constraints.  While
Verilog and VHDL include some primitive constructs for programmatic
circuit generation, they lack the powerful facilities present in
modern programming languages, such as object-oriented programming,
type inference, support for functional programming, and reflection.

>Instead of building a new hardware design language from scratch, we
chose to embed hardware construction primitives within an existing
language.  We picked Scala not only because it includes the
programming features we feel are important for building circuit
generators, but because it was specifically developed as a base for
domain-specific languages.
