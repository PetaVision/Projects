/*
 * BatchNorm.cpp
 * BatchNorm is a cloneVLayer, grabs activity from orig layer and rescales it
 * Implements Batch Normlization paper, Ioffe et. al. 2015
 */

#ifndef BATCHNORM_HPP_
#define BATCHNORM_HPP_

#include <layers/CloneVLayer.hpp>

namespace PV {

// CloneLayer can be used to implement Sigmoid junctions between spiking neurons
class BatchNorm: public PV::CloneVLayer {
    public:
        BatchNorm(const char * name, PV::HyPerCol * hc);
        virtual ~BatchNorm();
        virtual int communicateInitInfo();
        virtual int allocateV();
        virtual int updateState(double timef, double dt);
        virtual int setActivity();
        virtual int allocateDataStructures();

        //Getter functions for batch buffers
        const float* getBatchMean(){return batchMean;}
        const float* getBatchVar(){return batchVar;}
        //BackwardsBatchNorm will update these buffers
        float* getBatchMeanShift(){return batchMeanShift;}
        float* getBatchVarShift(){return batchVarShift;}
        float getEpsilon(){return epsilon;}

    protected:
        BatchNorm();
        int initialize(const char * name, PV::HyPerCol * hc);
        int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
    private:
        int initialize_base();

    protected:
        float* batchMean;
        float* batchVar;
        float* batchMeanShift;
        float* batchVarShift;
        float epsilon;
};

PV::BaseObject * createBatchNorm(char const * name, PV::HyPerCol * hc);

} /* namespace PV */

#endif /* CLONELAYER_HPP_ */
