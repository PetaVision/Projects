/*
 * BatchNorm.cpp
 */

#include "BatchNorm.hpp"
#include <stdio.h>

#include <include/default_params.h>

namespace PV {
BatchNorm::BatchNorm() {
    initialize_base();
}

BatchNorm::BatchNorm(const char * name, PV::HyPerCol * hc) {
    initialize_base();
    initialize(name, hc);
}

BatchNorm::~BatchNorm()
{
}

int BatchNorm::initialize_base() {
    originalLayer = NULL;
    batchMean = NULL;
    batchVar = NULL;
    batchMeanShift = NULL;
    batchVarShift = NULL;
    epsilon = 1e-6;
    return PV_SUCCESS;
}

int BatchNorm::initialize(const char * name, PV::HyPerCol * hc) {
    //int num_channels = sourceLayer->getNumChannels();
    int status_init = PV::CloneVLayer::initialize(name, hc);

    return status_init;
}

int BatchNorm::communicateInitInfo() {
    int status = PV::CloneVLayer::communicateInitInfo();
    return status;
}

//Rescale layer does not use the V buffer, so absolutely fine to clone off of an null V layer
int BatchNorm::allocateV() {
    //Do nothing
    return PV_SUCCESS;
}

int BatchNorm::allocateDataStructures() {
    int status = PV::CloneVLayer::allocateDataStructures();
    //If CloneVLayer is returning postpone because it's original layer hasn't allocated,
    //we pass postpone to caller
    if(status == PV_POSTPONE){
        return status;
    }
    //All buffers are of length nf
    const PVLayerLoc * loc = getLayerLoc();
    int nf = loc->nf; 
    batchMean = (float*) calloc(nf, sizeof(float));
    batchVar = (float*) calloc(nf, sizeof(float));
    batchMeanShift = (float*) calloc(nf, sizeof(float));
    batchVarShift = (float*) calloc(nf, sizeof(float));
    //Initialize varshift as 1
    for(int i = 0; i < nf; i++){
        batchVarShift[i] = 1;
    }

    return status;
}

int BatchNorm::ioParamsFillGroup(enum ParamsIOFlag ioFlag){
    //readOriginalLayerName(params);  // done in CloneVLayer
    PV::CloneVLayer::ioParamsFillGroup(ioFlag);
    return PV_SUCCESS;
}

int BatchNorm::setActivity() {
    pvdata_t * activity = clayer->activity->data;
    memset(activity, 0, sizeof(pvdata_t) * clayer->numExtendedAllBatches);
    return 0;
}

int BatchNorm::updateState(double timef, double dt) {
    int status = PV_SUCCESS;

    int numNeurons = originalLayer->getNumNeurons();
    pvdata_t * A = clayer->activity->data;
    const pvdata_t * originalA = originalLayer->getCLayer()->activity->data;
    const PVLayerLoc * loc = getLayerLoc();
    const PVLayerLoc * locOriginal = originalLayer->getLayerLoc();
    int nbatch = loc->nbatch; 

    //Make sure all sizes match
    //assert(locOriginal->nb == loc->nb);
    assert(locOriginal->nx == loc->nx);
    assert(locOriginal->ny == loc->ny);
    assert(locOriginal->nf == loc->nf);

    int nx = loc->nx;
    int ny = loc->ny;
    int nf = loc->nf;
    PVHalo const * halo = &loc->halo;
    PVHalo const * haloOrig = &locOriginal->halo;

    float normVal = parent->getNBatchGlobal() * loc->nyGlobal * loc->nxGlobal;

    //Can't thread here as we're using MPI_Allreduce
    for(int iF = 0; iF < nf; iF++){
        //Find mean
        float featureSum = 0;
        //Parallelizing over batches and Y, while reducing featureSum
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for collapse(2) reduction( + : featureSum)
#endif
        for(int b = 0; b < nbatch; b++){
            for(int iY = 0; iY < ny; iY++){ 
                const pvdata_t* originalABatch = originalA + b * originalLayer->getNumExtended();
                for(int iX = 0; iX < nx; iX++){ 
                    int kextOrig = kIndex(iX, iY, iF, nx+haloOrig->lt+haloOrig->rt, ny+haloOrig->dn+haloOrig->up, nf);
                    float origVal = originalABatch[kextOrig];
                    featureSum += origVal;
                }
            }
        }
        //MPI reduce the sum using the global communicator, as we're summing over batches as well
#ifdef PV_USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &featureSum, 1, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->globalCommunicator());
#endif // PV_USE_MPI
        batchMean[iF] = featureSum/normVal;

        //Find var
        float featureSumSq = 0;
        //Parallelizing over batches and Y, while reducing featureSumSq
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for collapse(2) reduction( + : featureSumSq)
#endif
        for(int b = 0; b < nbatch; b++){
            for(int iY = 0; iY < ny; iY++){ 
                const pvdata_t* originalABatch = originalA + b * originalLayer->getNumExtended();
                for(int iX = 0; iX < nx; iX++){ 
                    int kextOrig = kIndex(iX, iY, iF, nx+haloOrig->lt+haloOrig->rt, ny+haloOrig->dn+haloOrig->up, nf);
                    float origVal = originalABatch[kextOrig];
                    featureSumSq += powf(origVal-batchMean[iF], 2);
                }
            }
        }
        //MPI reduce the sum using the global communicator, as we're summing over batches as well
#ifdef PV_USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &featureSumSq, 1, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->globalCommunicator());
#endif // PV_USE_MPI
        //Fill batchMean and batchVar
        batchVar[iF] = featureSumSq/normVal;

        //Normalize and shift by mean and var
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for collapse(2)
#endif
        for(int b = 0; b < nbatch; b++){
            for(int iY = 0; iY < ny; iY++){ 
                const pvdata_t* originalABatch = originalA + b * originalLayer->getNumExtended();
                pvdata_t* ABatch = A + b * getNumExtended();
                //Loop through x and y
                for(int iX = 0; iX < nx; iX++){ 
                    int kextOrig = kIndex(iX, iY, iF, nx+haloOrig->lt+haloOrig->rt, ny+haloOrig->dn+haloOrig->up, nf);
                    int kext = kIndex(iX, iY, iF, nx+halo->lt+halo->rt, ny+halo->dn+halo->up, nf);
                    //Normalize
                    float normalizedVal = ((originalABatch[kextOrig]-batchMean[iF])/sqrtf(batchVar[iF] + epsilon));
                    //Shift
                    ABatch[kext] = normalizedVal * batchVarShift[iF] + batchMeanShift[iF];
                }
            }
        }
    }
    return status;
}

PV::BaseObject * createBatchNorm(char const * name, PV::HyPerCol * hc) { 
    return hc ? new BatchNorm(name, hc) : NULL;
}

} // end namespace PV

