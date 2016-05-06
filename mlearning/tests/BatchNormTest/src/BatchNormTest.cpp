/*
 *  * main.cpp for BatchNormTest
 *   *
 *    */


#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include <columns/MLPRegisterKeywords.hpp>
#include "BatchNormTestProbe.hpp"

int main(int argc, char * argv[]) {
   PV_Init initObj(&argc, &argv, false/*allowUnrecognizedArguments*/);
   initObj.initialize();
   PVMLearning::MLPRegisterKeywords(&initObj);
   initObj.registerKeyword("BatchNormTestProbe", PV::createBatchNormTestProbe);
   int rank = initObj.getWorldRank();
   int status = buildandrun(&initObj, NULL, NULL);

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
