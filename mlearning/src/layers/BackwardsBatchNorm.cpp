/*
 * BackwardsBatchNorm.cpp
 */

#include "BackwardsBatchNorm.hpp"
#include <stdio.h>

#include <include/default_params.h>

namespace PV {
BackwardsBatchNorm::BackwardsBatchNorm() {
    initialize_base();
}

BackwardsBatchNorm::BackwardsBatchNorm(const char * name, PV::HyPerCol * hc) {
    initialize_base();
    initialize(name, hc);
}

BackwardsBatchNorm::~BackwardsBatchNorm()
{
}

int BackwardsBatchNorm::initialize_base() {
    originalLayer = NULL; //Original layer is next_gradident
    forwardLayer = NULL;
    return PV_SUCCESS;
}

int BackwardsBatchNorm::initialize(const char * name, PV::HyPerCol * hc) {
    //int num_channels = sourceLayer->getNumChannels();
    int status_init = PV::CloneVLayer::initialize(name, hc);

    return status_init;
}

int BackwardsBatchNorm::communicateInitInfo() {
    int status = PV::CloneVLayer::communicateInitInfo();

    //Get layer from col
    PV::HyPerLayer* tmpLayer = parent->getLayerFromName(forwardLayerName);
    //TODO check if there is a better way to handle error messages
    if (tmpLayer==NULL) {
        if (parent->columnId()==0) {
            pvError() << getKeyword() << " \"" << name << "\" error: forwardLayer " << forwardLayerName << " is not a layer in the HyPerCol.\n";
        }
        MPI_Barrier(parent->icCommunicator()->communicator());
        exit(EXIT_FAILURE);
    }

    //Cast to BatchNorm
    forwardLayer = dynamic_cast<BatchNorm*>(tmpLayer);
    if (forwardLayer==NULL) {
        if (parent->columnId()==0) {
            pvError() << getKeyword() << " \"" << name << "\" error: forwardLayer " << forwardLayerName << " is not a BatchLayer.\n";
        }
        MPI_Barrier(parent->icCommunicator()->communicator());
        exit(EXIT_FAILURE);
    }

    const PVLayerLoc * fLoc = forwardLayer->getLayerLoc();
    const PVLayerLoc * thisLoc = getLayerLoc();
    assert(fLoc != NULL && thisLoc != NULL);
    if (fLoc->nxGlobal != thisLoc->nxGlobal || fLoc->nyGlobal != thisLoc->nyGlobal || fLoc->nf != thisLoc->nf) {
        if (parent->columnId()==0) {
            pvError() << getKeyword() << " \"" << name << "\" error: " << 
              "forwardLayer " << forwardLayerName << " does not have the same dimensions.\n" <<
              "forward (nx=" << fLoc->nxGlobal << 
              ", ny=" << fLoc->nyGlobal << 
              ", nf=" << fLoc->nf << ") versus " <<
              "(nx=" << thisLoc->nxGlobal <<
              ", ny=" << fLoc->nyGlobal <<
              ", nf=" << fLoc->nf << ")\n";
        }
        MPI_Barrier(parent->icCommunicator()->communicator());
        exit(EXIT_FAILURE);
    }
    assert(fLoc->nx==thisLoc->nx && fLoc->ny==thisLoc->ny);

    return status;
}

//Rescale layer does not use the V buffer, so absolutely fine to clone off of an null V layer
int BackwardsBatchNorm::allocateV() {
    //Do nothing
    return PV_SUCCESS;
}

int BackwardsBatchNorm::allocateDataStructures() {
    int status = PV::CloneVLayer::allocateDataStructures();
    //If CloneVLayer is returning postpone because it's original layer hasn't allocated,
    //we pass postpone to caller
    if(status == PV_POSTPONE){
        return status;
    }
    assert(status == PV_SUCCESS);
    //We need a few temp buffers, preallocated here
    int nf = getLayerLoc()->nf;
    deltaVar  = (float*) pvMalloc(nf * sizeof(float));
    deltaMean = (float*) pvMalloc(nf * sizeof(float));
    deltaVarShift = (float*) pvMalloc(nf * sizeof(float));
    deltaMeanShift = (float*) pvMalloc(nf * sizeof(float));

    clearDelta();

    return status;
}

int BackwardsBatchNorm::ioParamsFillGroup(enum ParamsIOFlag ioFlag){
    //readOriginalLayerName(params);  // done in CloneVLayer
    PV::CloneVLayer::ioParamsFillGroup(ioFlag);
    ioParam_forwardLayerName(ioFlag);
    return PV_SUCCESS;
}

void BackwardsBatchNorm::ioParam_forwardLayerName(enum ParamsIOFlag ioFlag) {
    parent->ioParamStringRequired(ioFlag, name, "forwardLayerName", &forwardLayerName);
}

int BackwardsBatchNorm::setActivity() {
    pvdata_t * activity = clayer->activity->data;
    memset(activity, 0, sizeof(pvdata_t) * clayer->numExtendedAllBatches);
    return 0;
}

int BackwardsBatchNorm::clearDelta(){
    int nf = getLayerLoc()->nf;
    for(int i = 0; i < nf; i++){
        deltaVar[i] = 0;
        deltaMean[i] = 0;
        deltaVarShift[i] = 0;
        deltaMeanShift[i] = 0;
    }
    return PV_SUCCESS;
}

int BackwardsBatchNorm::updateState(double timef, double dt) {
    int status = PV_SUCCESS;

    //We are filling this activity buffer
    pvdata_t * thisA = clayer->activity->data;
    assert(thisA);

    //We need the normalized input vals, orig input vals, and the input gradients
    const pvdata_t * inputGradA = originalLayer->getCLayer()->activity->data;
    const pvdata_t * forwardA = forwardLayer->getCLayer()->activity->data;
    const pvdata_t * origInputA = forwardLayer->getOriginalLayer()->getCLayer()->activity->data;
    assert(inputGradA && forwardA && origInputA);

    //Get locs for all buffers
    const PVLayerLoc * thisLoc = getLayerLoc();
    const PVLayerLoc * inputGradLoc = originalLayer->getLayerLoc();
    const PVLayerLoc * forwardLoc = forwardLayer->getLayerLoc();
    const PVLayerLoc * origInputLoc = forwardLayer->getOriginalLayer()->getLayerLoc();

    int nbatch = thisLoc->nbatch; 
    
    //All nx, ny, and nf should be the same
    int nx = thisLoc->nx;
    int ny = thisLoc->ny;
    int nf = thisLoc->nf;

    //Get buffer margins here
    int xThisMargin = thisLoc->halo.lt + thisLoc->halo.rt;
    int yThisMargin = thisLoc->halo.up + thisLoc->halo.dn;
    int xInputGradMargin = inputGradLoc->halo.lt + inputGradLoc->halo.rt;
    int yInputGradMargin = inputGradLoc->halo.up + inputGradLoc->halo.dn;
    int xForwardMargin = forwardLoc->halo.lt + forwardLoc->halo.rt;
    int yForwardMargin = forwardLoc->halo.up + forwardLoc->halo.dn;
    int xOrigInputMargin = origInputLoc->halo.lt + origInputLoc->halo.rt;
    int yOrigInputMargin = origInputLoc->halo.up + origInputLoc->halo.dn;
    
    //We also need various mean and var buffers from the forward layer
    const float* batchMean = forwardLayer->getBatchMean();
    const float* batchVar = forwardLayer->getBatchVar();
    float*       batchMeanShift = forwardLayer->getBatchMeanShift();
    float*       batchVarShift = forwardLayer->getBatchVarShift();
    float        epsilon = forwardLayer->getEpsilon();

    //Total number of neurons to divide by for each feature
    float normVal = parent->getNBatchGlobal() * thisLoc->nyGlobal * thisLoc->nxGlobal;

    //We're accumulating into delta buffers, so clear
    clearDelta();

    //Ioffe et. al. Batch Normalization

    //Calculate deltaVar
    //TODO parallize over threads
    for(int iF = 0; iF < nf; iF++){
        float secondTerm = -.5*(powf(batchVar[iF] + epsilon, -1.5));
        for(int b = 0; b < nbatch; b++){
            const pvdata_t* batchOrigInputA = origInputA + b * forwardLayer->getOriginalLayer()->getNumExtended();
            const pvdata_t* batchInputGradA = inputGradA + b * originalLayer->getNumExtended();
            for(int iY = 0; iY < ny; iY++){ 
                for(int iX = 0; iX < nx; iX++){ 
                    int kExtOrigInput = kIndex(iX, iY, iF, nx+xOrigInputMargin, ny+yOrigInputMargin, nf);
                    int kExtInputGrad = kIndex(iX, iY, iF, nx+xInputGradMargin, ny+yInputGradMargin, nf);
                    float deltaNorm = batchInputGradA[kExtInputGrad] * batchVarShift[iF];
                    deltaVar[iF] += deltaNorm * (batchOrigInputA[kExtOrigInput] - batchMean[iF]);
                }
            }
        }
        //Multiply deltaVar by secondTerm
        deltaVar[iF] = deltaVar[iF] * secondTerm;
    }

    //Reduce deltaVar
#ifdef PV_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, deltaVar, nf, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->globalCommunicator());
#endif // PV_USE_MPI

    //Calculate deltaMean
    //Calculate first term first
    //TODO parallize over threads
    for(int iF = 0; iF < nf; iF++){
        float multiplier = -1.0/(sqrtf(batchVar[iF]+epsilon));
        for(int b = 0; b < nbatch; b++){
            const pvdata_t* batchInputGradA = inputGradA + b * originalLayer->getNumExtended();
            for(int iY = 0; iY < ny; iY++){ 
                for(int iX = 0; iX < nx; iX++){ 
                    int kExtInputGrad = kIndex(iX, iY, iF, nx+xInputGradMargin, ny+yInputGradMargin, nf);
                    float deltaNorm = batchInputGradA[kExtInputGrad] * batchVarShift[iF];
                    deltaMean[iF] += deltaNorm * multiplier;
                }
            }
        }
    }
    //Reduce deltaMean across mpi
#ifdef PV_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, deltaMean, nf, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->globalCommunicator());
#endif // PV_USE_MPI

    //Calculate second term
    //TODO parallize over threads
    for(int iF = 0; iF < nf; iF++){
        float tmpMean = 0;
        for(int b = 0; b < nbatch; b++){
            const pvdata_t* batchOrigInputA = origInputA + b * forwardLayer->getOriginalLayer()->getNumExtended();
            for(int iY = 0; iY < ny; iY++){ 
                for(int iX = 0; iX < nx; iX++){ 
                    int kExtOrigInput = kIndex(iX, iY, iF, nx+xOrigInputMargin, ny+yOrigInputMargin, nf);
                    tmpMean += -2 * (batchOrigInputA[kExtOrigInput] - batchMean[iF]);
                }
            }
        }
        //Reduce tmpMean
#ifdef PV_USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &tmpMean, 1, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->globalCommunicator());
#endif // PV_USE_MPI
        tmpMean = tmpMean / normVal;
        //Add second term to first term
        deltaMean[iF] += deltaVar[iF] * tmpMean;
    }

    //No more sums, go with efficient loop
    //TODO Is the efficient loop better for optimization or do we put
    //features on the outer most loop for precalculation of constants over features?
    for(int b = 0; b < nbatch; b++){
        const pvdata_t* batchOrigInputA = origInputA + b * forwardLayer->getOriginalLayer()->getNumExtended();
        const pvdata_t* batchInputGradA = inputGradA + b * originalLayer->getNumExtended();
        pvdata_t* batchThisA = thisA + b * getNumExtended();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for collapse(3)
#endif
        for(int iY = 0; iY < ny; iY++){ 
            for(int iX = 0; iX < nx; iX++){ 
                for(int iF = 0; iF < nf; iF++){
                    int kExtOrigInput = kIndex(iX, iY, iF, nx+xOrigInputMargin, ny+yOrigInputMargin, nf);
                    int kExtInputGrad = kIndex(iX, iY, iF, nx+xInputGradMargin, ny+yInputGradMargin, nf);
                    int kExtThis = kIndex(iX, iY, iF, nx+xThisMargin, ny+yThisMargin, nf);
                    float deltaNorm = batchInputGradA[kExtInputGrad] * batchVarShift[iF];
                    float firstTerm = deltaNorm/sqrtf(batchVar[iF] + epsilon);
                    float secondTerm = deltaVar[iF] * (2*(batchOrigInputA[kExtOrigInput] - batchMean[iF])/normVal);
                    float thirdTerm = deltaMean[iF]/normVal;
                    batchThisA[kExtThis] = firstTerm + secondTerm + thirdTerm;
                }
            }
        }
    }

    //We calculate delta varShift and deltaMeanShift here
    //TODO parallize over threads
    //Since we're summing into delta*shift buffers, we have to sequentialize over features
    for(int iF = 0; iF < nf; iF++){
        for(int b = 0; b < nbatch; b++){
            const pvdata_t* batchForwardA = forwardA + b * forwardLayer->getNumExtended();
            const pvdata_t* batchInputGradA = inputGradA + b * originalLayer->getNumExtended();
            for(int iY = 0; iY < ny; iY++){ 
                for(int iX = 0; iX < nx; iX++){ 
                    int kExtInputGrad = kIndex(iX, iY, iF, nx+xInputGradMargin, ny+yInputGradMargin, nf);
                    int kExtForwardA = kIndex(iX, iY, iF, nx+xForwardMargin, ny + yForwardMargin, nf);
                    deltaVarShift[iF] += batchInputGradA[kExtInputGrad] * batchForwardA[kExtForwardA];
                    deltaMeanShift[iF] += batchInputGradA[kExtInputGrad];
                }
            }
        }
    }

    //Reduce delta*Shift across all mpi
#ifdef PV_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, deltaVarShift, nf, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->globalCommunicator());
    MPI_Allreduce(MPI_IN_PLACE, deltaMeanShift, nf, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->globalCommunicator());
#endif // PV_USE_MPI

    //TODO implement learning rule for meanShift and varShift

    return status;
}

PV::BaseObject * createBackwardsBatchNorm(char const * name, PV::HyPerCol * hc) { 
   return hc ? new BackwardsBatchNorm(name, hc) : NULL;
}

} // end namespace PV

