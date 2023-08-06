set(LIBS
        CIRCTLLHD
        CIRCTComb
        CIRCTHW
        CIRCTLLHDToLLVM
        CIRCTLLHDSimEngine
        )

# llhd-sim fails to link on Windows with MSVC.
IF(CIRCT_LLHD_SIM_ENABLED)
  add_llvm_tool(llhd-sim
    llhd-sim.cpp)
  
  llvm_update_compile_flags(llhd-sim)
  target_link_libraries(llhd-sim PRIVATE ${LIBS})
endif()
