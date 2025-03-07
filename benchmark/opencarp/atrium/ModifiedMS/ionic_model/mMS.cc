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
        

// DO NOT EDIT THIS SOURCE CODE FILE
// ANY CHANGES TO THIS FILE WILL BE OVERWRITTEN!!!!

#include "mMS.h"

#ifdef _OPENMP
#include <omp.h>
#endif



namespace limpet {

using ::opencarp::f_open;
using ::opencarp::FILE_SPEC;

mMSIonType::mMSIonType(bool plugin) : IonType(std::move(std::string("mMS")), plugin) {}

size_t mMSIonType::params_size() const {
  return sizeof(struct mMS_Params);
}

size_t mMSIonType::dlo_vector_size() const {

  return 1;
}

uint32_t mMSIonType::reqdat() const {
  return mMS_REQDAT;
}

uint32_t mMSIonType::moddat() const {
  return mMS_MODDAT;
}

void mMSIonType::destroy(IonIfBase& imp_base) const {
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  imp.destroy_luts();
  // rarely need to do anything else
}

Target mMSIonType::select_target(Target target) const {
  switch (target) {
    case Target::AUTO:
#   ifdef MMS_MLIR_CUDA_GENERATED
      return Target::MLIR_CUDA;
#   elif defined(MMS_MLIR_ROCM_GENERATED)
      return Target::MLIR_ROCM;
#   elif defined(MMS_MLIR_CPU_GENERATED)
      return Target::MLIR_CPU;
#   elif defined(MMS_CPU_GENERATED)
      return Target::CPU;
#   else
      return Target::UNKNOWN;
#   endif
#   ifdef MMS_MLIR_CUDA_GENERATED
    case Target::MLIR_CUDA:
      return Target::MLIR_CUDA;
#   endif
#   ifdef MMS_MLIR_ROCM_GENERATED
    case Target::MLIR_ROCM:
      return Target::MLIR_ROCM;
#   endif
#   ifdef MMS_MLIR_CPU_GENERATED
    case Target::MLIR_CPU:
      return Target::MLIR_CPU;
#   endif
#   ifdef MMS_CPU_GENERATED
    case Target::CPU:
      return Target::CPU;
#   endif
    default:
      return Target::UNKNOWN;
  }
}

void mMSIonType::compute(Target target, int start, int end, IonIfBase& imp_base, GlobalData_t** data) const {
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  switch(target) {
    case Target::AUTO:
#   ifdef MMS_MLIR_CUDA_GENERATED
      compute_mMS_mlir_gpu_cuda(start, end, imp, data);
#   elif defined(MMS_MLIR_ROCM_GENERATED)
      compute_mMS_mlir_gpu_rocm(start, end, imp, data);
#   elif defined(MMS_MLIR_CPU_GENERATED)
      compute_mMS_mlir_cpu(start, end, imp, data);
#   elif defined(MMS_CPU_GENERATED)
      compute_mMS_cpu(start, end, imp, data);
#   else
#     error "Could not generate method mMSIonType::compute."
#   endif
      break;
#   ifdef MMS_MLIR_CUDA_GENERATED
    case Target::MLIR_CUDA:
      compute_mMS_mlir_gpu_cuda(start, end, imp, data);
      break;
#   endif
#   ifdef MMS_MLIR_ROCM_GENERATED
    case Target::MLIR_ROCM:
      compute_mMS_mlir_gpu_rocm(start, end, imp, data);
      break;
#   endif
#   ifdef MMS_MLIR_CPU_GENERATED
    case Target::MLIR_CPU:
      compute_mMS_mlir_cpu(start, end, imp, data);
      break;
#   endif
#   ifdef MMS_CPU_GENERATED
    case Target::CPU:
      compute_mMS_cpu(start, end, imp, data);
      break;
#   endif
    default:
      throw std::runtime_error(std::string("Could not compute with the given target ") + get_string_from_target(target) + ".");
      break;
  }
}

// Define all constants
#define h_init (GlobalData_t)(1.0)



void mMSIonType::initialize_params(IonIfBase& imp_base) const
{
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  cell_geom* region = &imp.cgeom();
  mMS_Params *p = imp.params();

  // Compute the regional constants
  {
  }
  // Compute the regional initialization
  {
    p->V_gate = 0.13;
    p->V_max = 1.0;
    p->V_min = 0.0;
    p->a_crit = 0.13;
    p->tau_close = 150.0;
    p->tau_in = 0.3;
    p->tau_open = 120.0;
    p->tau_out = 5.0;
  }

}


// Define the parameters for the lookup tables
enum Tables {

  N_TABS
};

// Define the indices into the lookup tables.

    enum Rosenbrock {
    

      N_ROSEN
    };


void mMSIonType::construct_tables(IonIfBase& imp_base) const
{
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  GlobalData_t dt = imp.get_dt() * 1e0;
  cell_geom* region = &imp.cgeom();
  mMS_Params *p = imp.params();

  imp.tables().resize(N_TABS);

  // Define the constants that depend on the parameters.

}



void mMSIonType::initialize_sv(IonIfBase& imp_base, GlobalData_t **impdata ) const
{
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  GlobalData_t dt = imp.get_dt() * 1e0;
  cell_geom *region = &imp.cgeom();
  mMS_Params *p = imp.params();

  mMS_state *sv_base = (mMS_state *)imp.sv_tab().data();
  GlobalData_t t = 0;

  IonIfDerived* IF = &imp;
  // Define the constants that depend on the parameters.
  //Prepare all the public arrays.
  GlobalData_t *Iion_ext = impdata[Iion];
  GlobalData_t *V_ext = impdata[Vm];
  //Prepare all the private functions.

  //set the initial values
  for(int __i=0; __i < imp.get_num_node(); __i+=1 ){
    mMS_state *sv = sv_base+__i / 1;
    //Initialize the external vars to their current values
    GlobalData_t Iion = Iion_ext[__i];
    GlobalData_t V = V_ext[__i];
    //Change the units of external variables as appropriate.
    
    
    sv->V_gate = p->V_gate;
    sv->V_max = p->V_max;
    sv->V_min = p->V_min;
    sv->a_crit = p->a_crit;
    sv->tau_close = p->tau_close;
    sv->tau_in = p->tau_in;
    sv->tau_open = p->tau_open;
    sv->tau_out = p->tau_out;
    // Initialize the rest of the nodal variables
    double Uamp = (sv->V_max-(sv->V_min));
    double V_init = sv->V_min;
    sv->h = h_init;
    V = V_init;
    double J_in = ((((sv->h*((V-(sv->V_min))/Uamp))*(((V-(sv->V_min))/Uamp)-(sv->a_crit)))*((sv->V_max-(V))/Uamp))/sv->tau_in);
    double J_out = ((-(1.-(sv->h)))*(((V-(sv->V_min))/Uamp)/sv->tau_out));
    Iion = ((-Uamp)*(J_in+J_out));
    //Change the units of external variables as appropriate.
    
    
    //Save all external vars
    Iion_ext[__i] = Iion;
    V_ext[__i] = V;
  }

}

/** compute the  current
 *
 * param start   index of first node
 * param end     index of last node
 * param IF      IMP
 * param plgdata external data needed by IMP
 */
#ifdef MMS_CPU_GENERATED
extern "C" {
void compute_mMS_cpu(int start, int end, IonIfBase& imp_base, GlobalData_t **impdata )
{
  mMSIonType::IonIfDerived& imp = static_cast<mMSIonType::IonIfDerived&>(imp_base);
  GlobalData_t dt = imp.get_dt()*1e0;
  cell_geom *region = &imp.cgeom();
  mMS_Params *p  = imp.params();
  mMS_state *sv_base = (mMS_state *)imp.sv_tab().data();

  GlobalData_t t = imp.get_tstp().cnt*dt;

  mMSIonType::IonIfDerived* IF = &imp;

  // Define the constants that depend on the parameters.
  //Prepare all the public arrays.
  GlobalData_t *Iion_ext = impdata[Iion];
  GlobalData_t *V_ext = impdata[Vm];
  //Prepare all the private functions.

#pragma omp parallel for schedule(static)
  for (int __i=(start / 1) * 1; __i<end; __i+=1) {
    mMS_state *sv = sv_base+__i / 1;
                    
    //Initialize the external vars to their current values
    GlobalData_t Iion = Iion_ext[__i];
    GlobalData_t V = V_ext[__i];
    //Change the units of external variables as appropriate.
    
    
    //Compute lookup tables for things that have already been defined.
    
    
    //Compute storevars and external modvars
    GlobalData_t Uamp = (sv->V_max-(sv->V_min));
    GlobalData_t J_in = ((((sv->h*((V-(sv->V_min))/Uamp))*(((V-(sv->V_min))/Uamp)-(sv->a_crit)))*((sv->V_max-(V))/Uamp))/sv->tau_in);
    GlobalData_t J_out = ((-(1.-(sv->h)))*(((V-(sv->V_min))/Uamp)/sv->tau_out));
    Iion = ((-Uamp)*(J_in+J_out));
    
    
    //Complete Forward Euler Update
    GlobalData_t U = ((V-(sv->V_min))/Uamp);
    GlobalData_t diff_h = ((U<sv->V_gate) ? ((1.-(sv->h))/sv->tau_open) : ((-sv->h)/sv->tau_close));
    GlobalData_t h_new = sv->h+diff_h*dt;
    
    
    //Complete Rush Larsen Update
    
    
    //Complete RK2 Update
    
    
    //Complete RK4 Update
    
    
    //Complete Sundnes Update
    
    
    //Complete Markov Backward Euler method
    
    
    //Complete Rosenbrock Update
    
    
    //Complete Cvode Update
    
    
    //Finish the update
    Iion = Iion;
    sv->h = h_new;
    //Change the units of external variables as appropriate.
    
    
    //Save all external vars
    Iion_ext[__i] = Iion;
    V_ext[__i] = V;

  }

            }
}
#endif // MMS_CPU_GENERATED

bool mMSIonType::has_trace() const {
    return false;
}

void mMSIonType::trace(IonIfBase& imp_base, int node, FILE* file, GlobalData_t** data) const {}
IonIfBase* mMSIonType::make_ion_if(Target target, int num_node, const std::vector<std::reference_wrapper<IonType>>& plugins) const {
        // Place the allocated IonIf in managed memory if a GPU target exists for this model
        // otherwise, place it in main RAM
    IonIfDerived* ptr;
    if (this->select_target(Target::MLIR_ROCM) == Target::MLIR_ROCM) {
        ptr = allocate_on_target<IonIfDerived>(Target::MLIR_ROCM, 1, true);
    }
    else if (this->select_target(Target::MLIR_CUDA) == Target::MLIR_CUDA) {
        ptr = allocate_on_target<IonIfDerived>(Target::MLIR_CUDA, 1, true);
    }
    else {
        ptr = allocate_on_target<IonIfDerived>(Target::MLIR_CPU, 1, true);
    }
    // Using placement new to place the object in the correct memory
    return new(ptr) IonIfDerived(*this, this->select_target(target),
    num_node, plugins);
}

void mMSIonType::destroy_ion_if(IonIfBase *imp) const {
    // Call destructor and deallocate manually because the object might
    // be located on GPU (delete won't work in this case)
    imp->~IonIfBase();
    IonIfDerived* ptr = static_cast<IonIfDerived *>(imp);
    if (this->select_target(Target::MLIR_ROCM) == Target::MLIR_ROCM) {
        deallocate_on_target<IonIfDerived>(Target::MLIR_ROCM, ptr);
    }
    else if (this->select_target(Target::MLIR_CUDA) == Target::MLIR_CUDA) {
        deallocate_on_target<IonIfDerived>(Target::MLIR_CUDA, ptr);
    }
    else {
        deallocate_on_target<IonIfDerived>(Target::MLIR_CPU, ptr);
    }
}

}  // namespace limpet
        