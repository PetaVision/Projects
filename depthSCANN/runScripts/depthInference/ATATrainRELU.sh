lua ~/depthSCANN/input/depthInference/ATA_train_RELU.lua > ~/depthSCANN/input/depthInference/generated/ATA_train_RELU.params;
#One GPU run
~/depthSCANN/Release/depthSCANN -p ~/depthSCANN/input/depthInference/generated/ATA_train_RELU.params -t 8

#MultiGPU run
#mpirun -np 4 --bind-to none ~/depthSCANN/Release/depthSCANN -p ~/depthSCANN/input/depthInference/generated/ATA_train_RELU.params -t 8
