
#ifdef _AIX
#include <stdlib.h>
#endif
#include <string.h>
#include <dlfcn.h>

#include "ION_IF.h"
#include "MULTI_ION_IF.h"

namespace limpet {
using ::opencarp::dupstr;
extern opencarp::FILE_SPEC _nc_logf;
}  // namespace limpet


////////////////////////////////////////////////

#include "mMS.h"



namespace limpet {

void mMSIonType::tune(IonIfBase& iif_base, const char* im_par) const {
  IonIfDerived& iif = static_cast<IonIfDerived&>(iif_base);
  mMS_Params *p;							// pointer to parameter structure
  char parameter[1024], mod[1024];

  p = iif.params();

  // make sure flags is a valid member and flags specified are valid
  char *npar, *par, *parptr;
  par=tokstr_r(npar = dupstr(im_par), ",", &parptr);
  while( par ) {
    if( !strncmp(par, "flags=", 6 ) ) {


      log_msg( _nc_logf, 5, 0, "Unrecognized parameter: flags\n" );
      exit(1);


    }
    par = tokstr_r( NULL, ",", &parptr );
  }
  free(npar);

  // now process the regular parameters
  par=tokstr_r(npar = dupstr(im_par), ",", &parptr);
  while( par ) {
    process_param_mod( par, parameter, mod );
    if (0) ;

    else if( !strcmp( "V_gate", parameter ) )

      CHANGE_PARAM( mMS, p, V_gate, mod );

    else if( !strcmp( "V_max", parameter ) )

      CHANGE_PARAM( mMS, p, V_max, mod );

    else if( !strcmp( "V_min", parameter ) )

      CHANGE_PARAM( mMS, p, V_min, mod );

    else if( !strcmp( "a_crit", parameter ) )

      CHANGE_PARAM( mMS, p, a_crit, mod );

    else if( !strcmp( "tau_close", parameter ) )

      CHANGE_PARAM( mMS, p, tau_close, mod );

    else if( !strcmp( "tau_in", parameter ) )

      CHANGE_PARAM( mMS, p, tau_in, mod );

    else if( !strcmp( "tau_open", parameter ) )

      CHANGE_PARAM( mMS, p, tau_open, mod );

    else if( !strcmp( "tau_out", parameter ) )

      CHANGE_PARAM( mMS, p, tau_out, mod );


    else if( !strcmp( "flags", parameter ) )
      ;
    else {
      log_msg( _nc_logf, 5, 0,"Unrecognized parameter: %s.\n",parameter);
      log_msg( _nc_logf, 5, 0,"Run bench --imp=YourModel --imp-info to get a list of all parameters.\n",parameter);
      exit(1);
    }
    par = tokstr_r( NULL, ",", &parptr );
  }
  free( npar );
}




/** output all parameters which may be tuned for all IMPs
 */
void mMSIonType::print_params() const {
  IonIfDerived IF{*this, Target::AUTO, 0, {}};
  IF.initialize_params();
  printf("Name: mMS\n" );
  printf("\tParameters:\n" );

  printf( "\t%32s\t%g\n","V_gate", IF.params()->V_gate );

  printf( "\t%32s\t%g\n","V_max", IF.params()->V_max );

  printf( "\t%32s\t%g\n","V_min", IF.params()->V_min );

  printf( "\t%32s\t%g\n","a_crit", IF.params()->a_crit );

  printf( "\t%32s\t%g\n","tau_close", IF.params()->tau_close );

  printf( "\t%32s\t%g\n","tau_in", IF.params()->tau_in );

  printf( "\t%32s\t%g\n","tau_open", IF.params()->tau_open );

  printf( "\t%32s\t%g\n","tau_out", IF.params()->tau_out );


}



int mMSIonType::write_svs(IonIfBase& IF_base, FILE *out, int node) const {
  IonIfDerived& IF = static_cast<IonIfDerived&>(IF_base);
  fprintf( out, "%s\n", IF.get_type().get_name().c_str() );

  int inner_id = node % this->dlo_vector_size();

  mMS_state *sv = IF.sv_tab().data()+(node / (this->dlo_vector_size()));

  fprintf( out, "%-20g# V_gate\n", sv->V_gate );

  fprintf( out, "%-20g# V_max\n", sv->V_max );

  fprintf( out, "%-20g# V_min\n", sv->V_min );

  fprintf( out, "%-20g# a_crit\n", sv->a_crit );

  fprintf( out, "%-20g# h\n", sv->h );

  fprintf( out, "%-20g# tau_close\n", sv->tau_close );

  fprintf( out, "%-20g# tau_in\n", sv->tau_in );

  fprintf( out, "%-20g# tau_open\n", sv->tau_open );

  fprintf( out, "%-20g# tau_out\n", sv->tau_out );


  fprintf( out, "\n");
  return 0;
}



int mMSIonType::read_svs(IonIfBase& IF_base, FILE *in) const {
  IonIfDerived& IF = static_cast<IonIfDerived&>(IF_base);
  const int  BUFSIZE=256;
  char       impname[256], buf[BUFSIZE];
  const char *gdt_sc = sizeof(GlobalData_t)==sizeof(float)?"%f":"%lf";
  int flg = 0;
  
  // skip possible empty lines
  do {
    if( fgets(buf,BUFSIZE,in)==NULL ) {
      log_msg( _nc_logf, 4, 0, "no state information for IMP: %s\n", IF.get_type().get_name().c_str() );
      return 2;
    }
  } while( *buf=='\n' );
  sscanf( buf, "%s", impname );

  if( strcmp( impname, IF.get_type().get_name().c_str() ) ) {
    log_msg( _nc_logf, 5, 0, "IMPs do not match region (%s vs %s). Skipping this statefile.\n",impname, IF.get_type().get_name().c_str());
    return 2;
  }


  if(!IF.get_num_node())  {
    flg = 1;
  }
  mMS_state *sv = IF.sv_tab().data();

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->V_gate );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->V_max );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->V_min );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->a_crit );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->h );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->tau_close );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->tau_in );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->tau_open );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->tau_out );


  return flg;
}



SVgetfcn mMSIonType::get_sv_offset(const char *svname, int *off, int *sz) const {
  SVgetfcn retall = (SVgetfcn)(1);


        mMS_state *sv;

        if( !strcmp(svname,"ALL_SV") )  {

          *off  = 0;

          *sz   = sizeof(mMS_state);

          return retall;

        }

        if( !strcmp(svname,"V_gate") )  {

          *off  = offsetof(mMS_state,V_gate);

          *sz   = sizeof  (sv->V_gate) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"V_max") )  {

          *off  = offsetof(mMS_state,V_max);

          *sz   = sizeof  (sv->V_max) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"V_min") )  {

          *off  = offsetof(mMS_state,V_min);

          *sz   = sizeof  (sv->V_min) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"a_crit") )  {

          *off  = offsetof(mMS_state,a_crit);

          *sz   = sizeof  (sv->a_crit) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"h") )  {

          *off  = offsetof(mMS_state,h);

          *sz   = sizeof  (sv->h) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"tau_close") )  {

          *off  = offsetof(mMS_state,tau_close);

          *sz   = sizeof  (sv->tau_close) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"tau_in") )  {

          *off  = offsetof(mMS_state,tau_in);

          *sz   = sizeof  (sv->tau_in) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"tau_open") )  {

          *off  = offsetof(mMS_state,tau_open);

          *sz   = sizeof  (sv->tau_open) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"tau_out") )  {

          *off  = offsetof(mMS_state,tau_out);

          *sz   = sizeof  (sv->tau_out) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }


  return NULL;
}



int mMSIonType::get_sv_list(char*** list) const {

  *list = (char**)malloc( sizeof(char*)*9 );

  (*list)[0] = dupstr("V_gate");

  (*list)[1] = dupstr("V_max");

  (*list)[2] = dupstr("V_min");

  (*list)[3] = dupstr("a_crit");

  (*list)[4] = dupstr("h");

  (*list)[5] = dupstr("tau_close");

  (*list)[6] = dupstr("tau_in");

  (*list)[7] = dupstr("tau_open");

  (*list)[8] = dupstr("tau_out");

  return 9;


}



#define BOGUSTYPE -1
int mMSIonType::get_sv_type(const char *svname, int *type, char **Typename) const
{
  *type = BOGUSTYPE;
  if (0) ;

  else if( !strcmp(svname,"V_gate") )
 *type = 7;

  else if( !strcmp(svname,"V_max") )
 *type = 7;

  else if( !strcmp(svname,"V_min") )
 *type = 7;

  else if( !strcmp(svname,"a_crit") )
 *type = 7;

  else if( !strcmp(svname,"h") )
 *type = 7;

  else if( !strcmp(svname,"tau_close") )
 *type = 7;

  else if( !strcmp(svname,"tau_in") )
 *type = 7;

  else if( !strcmp(svname,"tau_open") )
 *type = 7;

  else if( !strcmp(svname,"tau_out") )
 *type = 7;


  else return 0;
  *Typename = get_typename(*type);
  return 1;
}




void mMSIonType::print_metadata() const {
  printf("Metadata:\n");
  printf("\t\n");
  printf("\t\n");
  printf("\t\n");
  printf("\t\n");
  printf("\t\n");
  printf("\t\n");
}

}  // namespace limpet

        

extern "C" {
    limpet::mMSIonType* __new_IonType(bool plugin) {
              return new limpet::mMSIonType(plugin);
    }
}
    
