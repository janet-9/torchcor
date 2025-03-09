// ----------------------------------------------------------------------------
// openCARP is an open cardiac electrophysiology simulator.
//
// Copyright (C) 2020 openCARP project
//
// This program is licensed under the openCARP Academic Public License (APL)
// v1.0: You can use and redistribute it and/or modify it in non-commercial
// academic environments under the terms of APL as published by the openCARP
// project v1.0, or (at your option) any later version. Commercial use requires
// a commercial license (info@opencarp.org).
//
// This program is distributed without any warranty; see the openCARP APL for
// more details.
//
// You should have received a copy of the openCARP APL along with this program
// and can find it online: http://www.opencarp.org/license
// ----------------------------------------------------------------------------


/*
*

*
*/
        
//// HEADER GUARD ///////////////////////////
// If automatically generated, keep above
// comment as first line in file.
#ifndef __MMS_H__
#define __MMS_H__
//// HEADER GUARD ///////////////////////////
// DO NOT EDIT THIS SOURCE CODE FILE
// ANY CHANGES TO THIS FILE WILL BE OVERWRITTEN!!!!

#include "ION_IF.h"

#if !(defined(MMS_CPU_GENERATED)    || defined(MMS_MLIR_CPU_GENERATED)    || defined(MMS_MLIR_ROCM_GENERATED)    || defined(MMS_MLIR_CUDA_GENERATED))
#ifdef MLIR_CPU_GENERATED
#define MMS_MLIR_CPU_GENERATED
#endif

#ifdef MLIR_ROCM_GENERATED
#define MMS_MLIR_ROCM_GENERATED
#endif

#ifdef MLIR_CUDA_GENERATED
#define MMS_MLIR_CUDA_GENERATED
#endif
#endif

#ifdef CPU_GENERATED
#define MMS_CPU_GENERATED
#endif

namespace limpet {

#define mMS_REQDAT Vm_DATA_FLAG
#define mMS_MODDAT Iion_DATA_FLAG

struct mMS_Params {
    GlobalData_t V_gate;
    GlobalData_t V_max;
    GlobalData_t V_min;
    GlobalData_t a_crit;
    GlobalData_t tau_close;
    GlobalData_t tau_in;
    GlobalData_t tau_open;
    GlobalData_t tau_out;

};

struct mMS_state {
    GlobalData_t V_gate;
    GlobalData_t V_max;
    GlobalData_t V_min;
    GlobalData_t a_crit;
    GlobalData_t h;
    GlobalData_t tau_close;
    GlobalData_t tau_in;
    GlobalData_t tau_open;
    GlobalData_t tau_out;

};

class mMSIonType : public IonType {
public:
    using IonIfDerived = IonIf<mMSIonType>;
    using params_type = mMS_Params;
    using state_type = mMS_state;

    mMSIonType(bool plugin);

    size_t params_size() const override;

    size_t dlo_vector_size() const override;

    uint32_t reqdat() const override;

    uint32_t moddat() const override;

    void initialize_params(IonIfBase& imp_base) const override;

    void construct_tables(IonIfBase& imp_base) const override;

    void destroy(IonIfBase& imp_base) const override;

    void initialize_sv(IonIfBase& imp_base, GlobalData_t** data) const override;

    Target select_target(Target target) const override;

    void compute(Target target, int start, int end, IonIfBase& imp_base, GlobalData_t** data) const override;

    bool has_trace() const override;

    void trace(IonIfBase& imp_base, int node, FILE* file, GlobalData_t** data) const override;

    void tune(IonIfBase& imp_base, const char* im_par) const override;

    int read_svs(IonIfBase& imp_base, FILE* file) const override;

    int write_svs(IonIfBase& imp_base, FILE* file, int node) const override;

    SVgetfcn get_sv_offset(const char* svname, int* off, int* sz) const override;

    int get_sv_list(char*** list) const override;

    int get_sv_type(const char* svname, int* type, char** type_name) const override;

    void print_params() const override;

    void print_metadata() const override;

    IonIfBase* make_ion_if(Target target, int num_node, const std::vector<std::reference_wrapper<IonType>>& plugins) const override;

    void destroy_ion_if(IonIfBase *imp) const override;
};

// This needs to be extern C in order to be linked correctly with the MLIR code
extern "C" {

//void compute_mMS(int, int, IonIfBase&, GlobalData_t**);
#ifdef MMS_CPU_GENERATED
void compute_mMS_cpu(int, int, IonIfBase&, GlobalData_t**);
#endif
#ifdef MMS_MLIR_CPU_GENERATED
void compute_mMS_mlir_cpu(int, int, IonIfBase&, GlobalData_t**);
#endif
#ifdef MMS_MLIR_ROCM_GENERATED
void compute_mMS_mlir_gpu_rocm(int, int, IonIfBase&, GlobalData_t**);
#endif
#ifdef MMS_MLIR_CUDA_GENERATED
void compute_mMS_mlir_gpu_cuda(int, int, IonIfBase&, GlobalData_t**);
#endif

}
}  // namespace limpet

//// HEADER GUARD ///////////////////////////
#endif
//// HEADER GUARD ///////////////////////////
