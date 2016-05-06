/*
 * GradientCheckProbe.hpp
 *
 *  Created on: May 6, 2016
 *      Author: Sheng Lundquist
 */

#ifndef GRADIENTCHECKPROBE_HPP_
#define GRADIENTCHECKPROBE_HPP_

#include <io/LayerProbe.hpp>
#include <layers/HyPerLayer.hpp>
#include <utils/PVLog.hpp>
#include <utils/PVAssert.hpp>

namespace PV {

class GradientCheckProbe: public PV::LayerProbe {
public:
   GradientCheckProbe(const char * probeName, HyPerCol * hc);
   virtual ~GradientCheckProbe();
   virtual int communicateInitInfo();
   int outputState(double timef);


protected:
   /**
    * Implements calcValues() for StatsProbe to always fail (getValues and getValue methods should not be used).
    */
   virtual int calcValues(double timevalue) { return PV_FAILURE; }

   GradientCheckProbe();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_gradLayerName(enum ParamsIOFlag ioFlag);
   void ioParam_estLayerName(enum ParamsIOFlag ioFlag);
   void ioParam_gtLayerName(enum ParamsIOFlag ioFlag);
   void ioParam_costFunction(enum ParamsIOFlag ioFlag);
   HyPerLayer* gradLayer;
   HyPerLayer* estLayer;
   HyPerLayer* gtLayer;
   char* gradLayerName;
   char* estLayerName;
   char* gtLayerName;
   char* costFunction;
   int prevIdx;
   float prevVal;
   float epsilon;
   
   
// Member variables

private:
   int init_base();
}; // end class GradientCheckProbe

BaseObject * createGradientCheckProbe(char const * name, HyPerCol * hc);

}

#endif /* STATSPROBE_HPP_ */
