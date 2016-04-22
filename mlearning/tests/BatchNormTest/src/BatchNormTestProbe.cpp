/*
 * BatchNormTestProbe.cpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#include "BatchNormTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <layers/BatchNorm.hpp>
#include <string.h>
#include <assert.h>

namespace PVMLearning {

BatchNormTestProbe::BatchNormTestProbe(const char * probeName, PV::HyPerCol * hc)
: StatsProbe()
{
   initBatchNormTestProbe(probeName, hc);
}

int BatchNormTestProbe::initBatchNormTestProbe_base() { return PV_SUCCESS; }

int BatchNormTestProbe::initBatchNormTestProbe(const char * probeName, PV::HyPerCol * hc)
{
   return initStatsProbe(probeName, hc);
}

//void BatchNormTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
//   requireType(BufActivity);
//}

int BatchNormTestProbe::communicateInitInfo() {
   int status = StatsProbe::communicateInitInfo();
   assert(getTargetLayer());
   BatchNorm * targetBatchNorm = dynamic_cast<BatchNorm *>(getTargetLayer());
   if (targetBatchNorm==NULL) {
      if (getParent()->columnId()==0) {
         fprintf(stderr, "BatchNormTestProbe Error: targetLayer \"%s\" is not a BatchNorm.\n", this->getTargetName());
      }
      MPI_Barrier(getParent()->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return status;
}

int BatchNormTestProbe::outputState(double timed)
{
   int status = StatsProbe::outputState(timed);

   if (timed==getParent()->getStartTime()) { return PV_SUCCESS; }
   float tolerance = 2.0e-5f;

   BatchNorm * targetBatchNorm = dynamic_cast<BatchNorm *>(getTargetLayer());
   assert(targetBatchNorm);

   int numNeurons = targetBatchNorm->getNumNeurons();
   assert(numNeurons == targetBatchNorm->getOriginalLayer()->getNumNeurons());
   PVLayerLoc const * rescaleLoc = targetBatchNorm->getLayerLoc();
   PVHalo const * rescaleHalo = &rescaleLoc->halo;
   int nf = rescaleLoc->nf;
   int ny = rescaleLoc->ny;
   int nx = rescaleLoc->nx;
   int nbatch = parent->getNBatch();

   float correctThresh = 1e-4;

   //Kept here in case we want to measure colinearity 
   //PV::HyPerLayer * originalLayer = targetBatchNorm->getOriginalLayer();
   //PVLayerLoc const * origLoc = originalLayer->getLayerLoc();
   //PVHalo const * origHalo = &origLoc->halo;
   //assert(origLoc->nf == nf);
   //assert(origLoc->nx == nx);
   //assert(origLoc->ny == ny);
   float normVal = parent->getNBatchGlobal() * rescaleLoc->nyGlobal * rescaleLoc->nxGlobal;

   for (int fi = 0; fi < nf; fi++){
      float featureSum = 0;
      for(int b = 0; b < nbatch; b++){
         pvadata_t const * rescaledData = targetBatchNorm->getLayerData() + b * targetBatchNorm->getNumExtended();
         //pvadata_t const * originalData = originalLayer->getLayerData() + b * originalLayer->getNumExtended();
          for(int iY = 0; iY < ny; iY++){ 
             for(int iX = 0; iX < nx; iX++){ 
                int rescaleKext = kIndex(iX, iY, fi, nx+rescaleHalo->lt+rescaleHalo->rt, ny+rescaleHalo->dn+rescaleHalo->up, nf);
                float rescaleVal = rescaledData[rescaleKext];
                featureSum += rescaleVal;
             }
          }
      }
      //MPI reduce the sum using the global communicator, as we're summing over batches as well
#ifdef PV_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &featureSum, 1, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->globalCommunicator());
#endif // PV_USE_MPI

      float featureMean = featureSum/normVal;

      float featureSumSq = 0;
      for(int b = 0; b < nbatch; b++){
         pvadata_t const * rescaledData = targetBatchNorm->getLayerData() + b * targetBatchNorm->getNumExtended();
         //pvadata_t const * originalData = originalLayer->getLayerData() + b * originalLayer->getNumExtended();
          for(int iY = 0; iY < ny; iY++){ 
             for(int iX = 0; iX < nx; iX++){ 
                int rescaleKext = kIndex(iX, iY, fi, nx+rescaleHalo->lt+rescaleHalo->rt, ny+rescaleHalo->dn+rescaleHalo->up, nf);
                float rescaleVal = rescaledData[rescaleKext];
                featureSumSq += powf(rescaleVal - featureMean, 2);
             }
          }
      }
#ifdef PV_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &featureSumSq, 1, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->globalCommunicator());
#endif // PV_USE_MPI
      float featureVar = featureSumSq/normVal;

      //Since we're not learning a meanshift/varshift, mean should be 0, var should be 1
      if (fabs(featureMean - 0) > correctThresh){
         std::cout << "Feature " << fi << " contains values not of correct mean. Actual mean: " << featureMean << "\n";
         status = PV_FAILURE;
      }
      if (fabs(featureVar - 1) > correctThresh){
         std::cout << "Feature " << fi << " contains values not of correct var. Actual var: " << featureVar << "\n";
         status = PV_FAILURE;
      }
   }
   if (status == PV_FAILURE) {
      exit(EXIT_FAILURE);
   }
   return status;
}

PV::BaseObject * createBatchNormTestProbe(char const * name, PV::HyPerCol * hc) { 
   return hc ? new BatchNormTestProbe(name, hc) : NULL;
}

} /* namespace PV */
