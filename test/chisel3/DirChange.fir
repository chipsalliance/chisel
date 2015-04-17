;RUN: firrtl %s abcefghipjklmno c | tee %s.out | FileCheck %s
;CHECK: To Flo
circuit DirChange : 
  module DirChange : 
    input test1 : UInt(5)
    output test2 : UInt(5)
    input test3 : UInt(2)[10]
    output test4 : {test41 : UInt(5), test42 : UInt(5)}
    skip
;CHECK: Finished To Flo
