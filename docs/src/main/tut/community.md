---
layout: page
title:  "Community"
section: "Community"
position: 16
---

## Chisel Users Community

If you're a Chisel user and want to stay connected to the wider user community, any of the following are great avenues:

- Interact with other Chisel users in one of our Gitter chat rooms:
  - [Chisel](https://gitter.im/freechipsproject/chisel3)
  - [FIRRTL](https://gitter.im/freechipsproject/firrtl)
- Ask/Answer [Questions on Stack Overflow using the `[chisel]` tag](https://stackoverflow.com/questions/tagged/chisel)
- Ask questions and discuss ideas on the Chisel/FIRRTL Mailing Lists:
  - [Chisel Users](https://groups.google.com/forum/#!forum/chisel-users)
  - [Chisel Developers](https://groups.google.com/forum/#!forum/chisel-dev)
- Follow us on our [`@chisel_lang` Twitter Account](https://twitter.com/chisel_lang)
- Subscribe to our [`chisel-lang` YouTube Channel](https://www.youtube.com/c/chisel-lang)

## Chisel Developers Community

If you want to get *more involved* with the Chisel/FIRRTL ecosystem of projects, feel free to reach out to us on any of the mediums above. If you prefer to dive right in (or have bugs to report), a complete list of the associated Chisel/FIRRTL ecosystem of projects is below:

- [Chisel](https://github.com/freechipsproject/chisel3)
- [FIRRTL](https://github.com/freechipsproject/firrtl)
- [Chisel Testers/`chisel3.iotesters`](https://github.com/freechipsproject/chisel-testers)
- [Chisel Testers2/`chisel3.testers`](https://github.com/ucb-bar/chisel-testers)
- [Treadle](https://github.com/freechipsproject/treadle)
- [Diagrammer](https://github.com/freechipsproject/diagrammer)

## Contributors

Chisel, FIRRTL, and all related projects would not be possible without the contributions of our fantastic developer community.
The following people have contributed to the current release of the projects:

{% include_relative contributors.md %}

## Papers

While Chisel has come a long way since 2012, the original Chisel paper provides some background on motivations and an overview of the (now deprecated) Chisel 2 language:

- [Bachrach, Jonathan, et al. "Chisel: constructing hardware in a scala embedded language." DAC Design Automation Conference 2012. IEEE, 2012.](https://people.eecs.berkeley.edu/~jrb/papers/chisel-dac-2012-corrected.pdf)

The FIRRTL IR and FIRRTL compiler, introduced as part of Chisel 3, are discussed in both the following paper and specification[^1]:

- [Izraelevitz, Adam, et al. "Reusability is FIRRTL ground: Hardware construction languages, compiler frameworks, and transformations." Proceedings of the 36th International Conference on Computer-Aided Design. IEEE Press, 2017.](https://aspire.eecs.berkeley.edu/wp/wp-content/uploads/2017/11/Reusability-is-FIRRTL-Ground-Izraelevitz.pdf)
- [Li, Patrick S., Adam M. Izraelevitz, and Jonathan Bachrach. "Specification for the FIRRTL Language." EECS Department, University of California, Berkeley, Tech. Rep. UCB/EECS-2016-9 (2016).](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-9.pdf)

Finally, Chisel's functional programming and bit-width inference ideas were inspired by earlier work on a hardware description language called *Gel*:

- [Bachrach, Jonathan, Dany Qumsiyeh, and Mark Tobenkin. "Hardware scripting in gel." 2008 16th International Symposium on Field-Programmable Custom Computing Machines. IEEE, 2008.](http://people.eecs.berkeley.edu/~jrb/papers/gel-fccm-2008.pdf)

## Attribution

If you use Chisel in your research, consider citing:

```bib
@inproceedings{bachrach:2012:chisel,
  author={J. {Bachrach} and H. {Vo} and B. {Richards} and Y. {Lee} and A. {Waterman} and R {Avižienis} and J. {Wawrzynek} and K. {Asanović}},
  booktitle={DAC Design Automation Conference 2012},
  title={Chisel: Constructing hardware in a Scala embedded language},
  year={2012},
  volume={},
  number={},
  pages={1212-1221},
  keywords={application specific integrated circuits;C++ language;field programmable gate arrays;hardware description languages;Chisel;Scala embedded language;hardware construction language;hardware design abstraction;functional programming;type inference;high-speed C++-based cycle-accurate software simulator;low-level Verilog;FPGA;standard ASIC flow;Hardware;Hardware design languages;Generators;Registers;Wires;Vectors;Finite impulse response filter;CAD},
  doi={10.1145/2228360.2228584},
  ISSN={0738-100X},
  month={June},}
```

If you use FIRRTL in your research consider citing:

```bib
@INPROCEEDINGS{8203780,
  author={A. Izraelevitz and J. Koenig and P. Li and R. Lin and A. Wang and A. Magyar and D. Kim and C. Schmidt and C. Markley and J. Lawson and J. Bachrach},
  booktitle={2017 IEEE/ACM International Conference on Computer-Aided Design (ICCAD)},
  title={Reusability is FIRRTL ground: Hardware construction languages, compiler frameworks,
  and transformations},
  year={2017},
  volume={},
  number={},
  pages={209-216},
  keywords={field programmable gate arrays;hardware description languages;program compilers;software reusability;hardware development practices;hardware libraries;open-source hardware intermediate representation;hardware compiler transformations;Hardware construction languages;retargetable compilers;software development;virtual Cambrian explosion;hardware compiler frameworks;parameterized libraries;FIRRTL;FPGA mappings;Chisel;Flexible Intermediate Representation for RTL;Reusability;Hardware;Libraries;Hardware design languages;Field programmable gate arrays;Tools;Open source software;RTL;Design;FPGA;ASIC;Hardware;Modeling;Reusability;Hardware Design Language;Hardware Construction Language;Intermediate Representation;Compiler;Transformations;Chisel;FIRRTL},
  doi={10.1109/ICCAD.2017.8203780},
  ISSN={1558-2434},
  month={Nov},}
```
```bib
@techreport{Li:EECS-2016-9,
    Author = {Li, Patrick S. and Izraelevitz, Adam M. and Bachrach, Jonathan},
    Title = {Specification for the FIRRTL Language},
    Institution = {EECS Department, University of California, Berkeley},
    Year = {2016},
    Month = {Feb},
    URL = {http://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-9.html},
    Number = {UCB/EECS-2016-9}
}
```

[^1]: This specification is provided for historical perspective. For the latest version of the FIRRTL specification you can use [this link](https://github.com/freechipsproject/firrtl/raw/master/spec/spec.pdf).
