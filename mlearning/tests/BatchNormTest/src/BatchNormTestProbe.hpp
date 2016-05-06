/*
 * BatchNormTestProbe.hpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#ifndef BATCHNORMTESTPROBE_HPP_
#define BATCHNORMTESTPROBE_HPP_

#include <io/StatsProbe.hpp>

namespace PV{

class BatchNormTestProbe: public PV::StatsProbe {
public:
   BatchNormTestProbe(const char * probeName, PV::HyPerCol * hc);
   virtual int communicateInitInfo();
   virtual int outputState(double timed);

protected:
   int initBatchNormTestProbe(const char * probeName, PV::HyPerCol * hc);
   //void ioParam_buffer(enum ParamsIOFlag ioFlag);

private:
   int initBatchNormTestProbe_base();

}; // end class BatchNormTestProbe

PV::BaseObject * createBatchNormTestProbe(char const * name, PV::HyPerCol * hc);

}  // end namespace PV
#endif 
