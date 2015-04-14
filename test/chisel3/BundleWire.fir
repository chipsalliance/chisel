circuit BundleWire : 
  module BundleWire : 
    output in : {input y : UInt(32), input x : UInt(32)}
    output outs : {output y : UInt(32), output x : UInt(32)}[4]
    
    wire coords : {output y : UInt(32), output x : UInt(32)}[4]
    coords.0 := in
    outs.0 := coords.0
    coords.1 := in
    outs.1 := coords.1
    coords.2 := in
    outs.2 := coords.2
    coords.3 := in
    outs.3 := coords.3