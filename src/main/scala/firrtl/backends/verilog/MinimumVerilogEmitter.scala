package firrtl

import firrtl.stage.TransformManager

class MinimumVerilogEmitter extends VerilogEmitter with Emitter {

  override def prerequisites = firrtl.stage.Forms.AssertsRemoved ++
    firrtl.stage.Forms.LowFormMinimumOptimized

  override def transforms =
    new TransformManager(firrtl.stage.Forms.VerilogMinimumOptimized, prerequisites).flattenedTransformOrder

}
