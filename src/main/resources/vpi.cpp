#include "vpi.h"

vpi_api_t vpi_api;

/*==========================================================================
                 User Functions
=============================================================================*/

PLI_INT32 init_clks_calltf(PLI_BYTE8 *user_data) {
  vpi_api.init_clks();
  return 0;
}

PLI_INT32 init_rsts_calltf(PLI_BYTE8 *user_data) {
  vpi_api.init_rsts();
  return 0;
}

PLI_INT32 init_ins_calltf(PLI_BYTE8 *user_data) {
  vpi_api.init_ins();
  return 0;
}

PLI_INT32 init_outs_calltf(PLI_BYTE8 *user_data) {
  vpi_api.init_outs();
  return 0;
}

PLI_INT32 init_sigs_calltf(PLI_BYTE8 *user_data) {
  vpi_api.init_sigs();
  return 0;
}

PLI_INT32 tick_calltf(PLI_BYTE8 *user_data) {
  vpi_api.tick();
  return 0;
}

PLI_INT32 tick_cb(p_cb_data cb_data) {
  vpi_api.tick();
  return 0;
}

/*==========================================================================
                 Registration Functions
=============================================================================*/
void init_clks_registration() {
  s_vpi_systf_data tf_data;
  tf_data.type      = vpiSysTask;
  tf_data.tfname    = (PLI_BYTE8*) "$init_clks";
  tf_data.sizetf    = NULL;
  tf_data.calltf    = init_clks_calltf;
  tf_data.compiletf = NULL;
  tf_data.user_data = NULL;
  vpi_register_systf(&tf_data);
  return;
}

void init_rsts_registration() {
  s_vpi_systf_data tf_data;
  tf_data.type      = vpiSysTask;
  tf_data.tfname    = (PLI_BYTE8*) "$init_rsts";
  tf_data.sizetf    = NULL;
  tf_data.calltf    = init_rsts_calltf;
  tf_data.compiletf = NULL;
  tf_data.user_data = NULL;
  vpi_register_systf(&tf_data);
  return;
}

void init_ins_registration() {
  s_vpi_systf_data tf_data;
  tf_data.type      = vpiSysTask;
  tf_data.tfname    = (PLI_BYTE8*) "$init_ins";
  tf_data.sizetf    = NULL;
  tf_data.calltf    = init_ins_calltf;
  tf_data.compiletf = NULL;
  tf_data.user_data = NULL;
  vpi_register_systf(&tf_data);
  return;
}

void init_outs_registration() {
  s_vpi_systf_data tf_data;
  tf_data.type      = vpiSysTask;
  tf_data.tfname    = (PLI_BYTE8*) "$init_outs";
  tf_data.sizetf    = NULL;
  tf_data.calltf    = init_outs_calltf;
  tf_data.compiletf = NULL;
  tf_data.user_data = NULL;
  vpi_register_systf(&tf_data);
  return;
}

void init_sigs_registration() {
  s_vpi_systf_data tf_data;
  tf_data.type      = vpiSysTask;
  tf_data.tfname    = (PLI_BYTE8*) "$init_sigs";
  tf_data.sizetf    = NULL;
  tf_data.calltf    = init_sigs_calltf;
  tf_data.compiletf = NULL;
  tf_data.user_data = NULL;
  vpi_register_systf(&tf_data);
  return;
}

void tick_registration() {
  s_vpi_systf_data tf_data;
  tf_data.type      = vpiSysTask;
  tf_data.tfname    = (PLI_BYTE8*) "$tick";
  tf_data.sizetf    = NULL;
  tf_data.calltf    = tick_calltf;
  tf_data.compiletf = NULL;
  tf_data.user_data = NULL;
  vpi_register_systf(&tf_data);
  return;
}

/*==========================================================================
                 Start-up Array
=============================================================================*/
void (*vlog_startup_routines[]) () = {
  init_clks_registration,
  init_rsts_registration,
  init_ins_registration,
  init_outs_registration,
  init_sigs_registration,
  tick_registration,
  0
};
