/*
 * BackwardsBatchNorm.cpp
 * BackwardsBatchNorm is a cloneVLayer, grabs activity from orig layer and rescales it
 * Implements Batch Normlization paper, Ioffe et. al. 2015
 */

#ifndef BACKWARDSBATCHNORM_HPP_
#define BACKWARDSBATCHNORM_HPP_

#include <layers/CloneVLayer.hpp>
#include "BatchNorm.hpp"

#include <utils/PVLog.hpp>
#include <utils/PVAssert.hpp>
#include <utils/PVAlloc.hpp>

namespace PV {

// CloneLayer can be used to implement Sigmoid junctions between spiking neurons
class BackwardsBatchNorm: public PV::CloneVLayer {
   public:
      BackwardsBatchNorm(const char * name, PV::HyPerCol * hc);
      virtual ~BackwardsBatchNorm();
      virtual int communicateInitInfo();
      virtual int allocateV();
      virtual int updateState(double timef, double dt);
      virtual int setActivity();
      virtual int allocateDataStructures();

   protected:
      BackwardsBatchNorm();
      int initialize(const char * name, PV::HyPerCol * hc);
      int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
      void ioParam_forwardLayerName(enum ParamsIOFlag ioFlag);
      int clearDelta();

   private:
      int initialize_base();

   protected:
      char * forwardLayerName;
      BatchNorm* forwardLayer; //The orig batchnorm layer

      //Buffers for calculating partial derivatives
      float* deltaVar;
      float* deltaMean;
      float* deltaVarShift;
      float* deltaMeanShift;
};

PV::BaseObject * createBackwardsBatchNorm(char const * name, PV::HyPerCol * hc);

} /* namespace PV */

#endif /* CLONELAYER_HPP_ */
