/*
 * GradientCheckProbe.cpp
 *
 *  Created on: May 6, 2016
 *      Author: Sheng Lundquist
 */

#include "GradientCheckProbe.hpp"

namespace PV {

/**
 * @probeName
 * @hc
 */
GradientCheckProbe::GradientCheckProbe(const char * probeName, HyPerCol * hc) {
   init_base();
}

GradientCheckProbe::GradientCheckProbe()
   : LayerProbe()
{
   init_base();
   // Derived classes should call initGradientCheckProbe
}

GradientCheckProbe::~GradientCheckProbe()
{
   int rank = getParent()->columnId();
}

int GradientCheckProbe::init_base() {
   gradLayer = NULL;
   estLayer = NULL;
   gtLayer = NULL;
   gradLayerName = NULL;
   estLayerName = NULL;
   gtLayerName = NULL;
   costFunction = NULL;
   epsilon = 1e-4;
   prevIdx = -1;
   prevVal = 0;

   return PV_SUCCESS;
}

int GradientCheckProbe::communicateInitInfo() {
   LayerProbe::communicateInitInfo();

   //Get necessary layers from col
   gradLayer = parent->getLayerFromName(gradLayerName);
   if (gradLayer==NULL) {
      if (parent->columnId()==0) {
         pvError() << getKeyword() << " \"" << name << "\" error: gradLayerName " << gradLayerName << " is not a layer in the HyPerCol.\n";
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   
   //targetLayer (forward layer) and gradLayer must have the same restricted size
   pvAssert(gradLayer->getNumNeurons() == targetLayer->getNumNeurons());

   estLayer = parent->getLayerFromName(estLayerName);
   if (estLayer ==NULL) {
      if (parent->columnId()==0) {
         pvError() << getKeyword() << " \"" << name << "\" error: estLayerName " << estLayerName << " is not a layer in the HyPerCol.\n";
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   gtLayer = parent->getLayerFromName(gtLayerName);
   if (gtLayer ==NULL) {
      if (parent->columnId()==0) {
         pvError() << getKeyword() << " \"" << name << "\" error: gtLayerName " << gtLayerName << " is not a layer in the HyPerCol.\n";
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   pvAssert(gtLayer->getNumNeurons() == estLayer->getNumNeurons());

   if(parent->getFinalStep() - parent->getInitialStep() > targetLayer->getNumNeurons() + 1){
      std::cout << "Maximum number of steps for GradientCheckConn is " << targetLayer->getNumNeurons() << "\n";
      exit(-1);
   }

}

int GradientCheckProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_gradLayerName(ioFlag);
   ioParam_gtLayerName(ioFlag);
   ioParam_estLayerName(ioFlag);
   ioParam_costFunction(ioFlag);

   return status;
}

void GradientCheckProbe::ioParam_gradLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "gradLayerName", &gradLayerName);
}

void GradientCheckProbe::ioParam_estLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "estLayerName", &estLayerName);
}

void GradientCheckProbe::ioParam_gtLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "gtLayerName", &gtLayerName);
}

void GradientCheckProbe::ioParam_costFunction(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "costFunction", &costFunction);
   if(strcmp(costFunction, "sqerr") != 0 && 
      strcmp(costFunction, "logerr") != 0){
      std::cout << "costFunction " << costFunction << " not known, options are \"sqerr\" and \"logerr\" \n";
      exit(-1);
   }
}

/**
 * @time
 * @l
 */
int GradientCheckProbe::outputState(double timed)
{
#ifdef PV_USE_MPI
   InterColComm * icComm = parent->icCommunicator();
   MPI_Comm comm = icComm->globalCommunicator();
   int rank = icComm->globalCommRank();
#endif // PV_USE_MPI
   std::cout << name << " probeOutputState on timestep " << timed << "\n";

   return PV_SUCCESS;
}

BaseObject * createGradientCheckProbe(char const * name, HyPerCol * hc) {
   return hc ? new GradientCheckProbe(name, hc) : NULL;
}

}
