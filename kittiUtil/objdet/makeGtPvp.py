import pdb
import numpy as np
import scipy.sparse as sp
from pvtools import pvpOpen
from os.path import isfile

baseDir = '/nh/compneuro/Data/KITTI/objdet/training/'
labelDir = baseDir + '/label_2/'

outDir = '/nh/compneuro/Data/KITTI/objdet/training/tmp/'

numData = 7481

maxImageSizeX = 1242
maxImageSizeY = 376

#xTarget = 512
#yTarget = 144

def getFeatureIdx(bbType):
    if bbType == 'Car':
        featureIdx = 0
    elif bbType == 'Van':
        featureIdx = 1
    elif bbType == 'Truck':
        featureIdx = 2
    elif bbType == 'Pedestrian':
        featureIdx = 3
    elif bbType == 'Person_sitting':
        featureIdx = 4
    elif bbType == 'Cyclist':
        featureIdx = 5
    elif bbType == 'Tram':
        featureIdx = 6
    elif bbType == 'Misc':
        featureIdx = 7
    elif bbType == 'DontCare':
        featureIdx = -1;
    else:
        assert(0)
    return featureIdx

def writeIdxList(objs, idx, xSizeOrig, ySizeOrig, xSizeTarget, ySizeTarget, gtStream, dncStream):

   scaleX = float(xSizeTarget)/xSizeOrig
   scaleY = float(ySizeTarget)/ySizeOrig

   outGT = np.zeros((ySizeTarget, xSizeTarget, 8)) #In petavision order, 8 classes
   dncImg = np.zeros((ySizeTarget, xSizeTarget, 1)) #DNC class seperate

   for obj in objs:
       (name, x1, y1, x2, y2) = obj
       featureIdx = getFeatureIdx(name)
       x1 = round(x1 * scaleX)
       x2 = round(x2 * scaleX)
       y1 = round(y1 * scaleY)
       y2 = round(y2 * scaleY)
       if(featureIdx == -1):
           dncImg[y1:y2, x1:x2, 0] = 1
       else:
           outGT[y1:y2, x1:x2, featureIdx] = 1

   #Convert to sparse matrix
   sparseGT = sp.csr_matrix(outGT.reshape(1, ySizeTarget*xSizeTarget*8))
   sparseDNC = sp.csr_matrix(dncImg.reshape(1, ySizeTarget*xSizeTarget))
   time = [idx]

   sparseGTData = {"values": sparseGT, "time": time}
   sparseDNCData = {"values": sparseDNC, "time": time}

   #Write to file
   gtStream.write(sparseGTData, shape=(ySizeTarget, xSizeTarget, 8))
   dncStream.write(sparseDNCData, shape=(ySizeTarget, xSizeTarget, 1))

def findFilenames(baseDir, imgIdx):
    #Goes left0, left-1, left-2, left-3, right0, right-1, right-2, right-3
    filenames = [baseDir + "/image_2/" + str(imgIdx).zfill(6) + ".png",
                 baseDir + "/prev_2/"  + str(imgIdx).zfill(6) + "_01.png",
                 baseDir + "/prev_2/"  + str(imgIdx).zfill(6) + "_02.png",
                 baseDir + "/prev_2/"  + str(imgIdx).zfill(6) + "_03.png",
                 baseDir + "/image_3/" + str(imgIdx).zfill(6) + ".png",
                 baseDir + "/prev_3/"  + str(imgIdx).zfill(6) + "_01.png",
                 baseDir + "/prev_3/"  + str(imgIdx).zfill(6) + "_02.png",
                 baseDir + "/prev_3/"  + str(imgIdx).zfill(6) + "_03.png",
                ]
    #Make sure all files exist
    allFilesExist = True
    for fn in filenames:
        if(not isfile(fn)):
            allFilesExist = False
            print "Not found: " + fn
            break
    if(allFilesExist):
        return filenames
    else:
        return None

def readLabels(labelDir, imgIdx):
    filename = labelDir + "/" + str(imgIdx).zfill(6) + ".txt"
    f = open(filename, 'r')
    objs = f.readlines()
    f.close()

    out = []
    for obj in objs:
        split = obj.split()
        out.append((split[0], float(split[4]), float(split[5]), float(split[6]), float(split[7])))
    return out

#Open pvp files
gtStream = pvpOpen(outDir + "kittiGT.pvp", 'w')
dncStream = pvpOpen(outDir + "dncData.pvp", 'w')

#List of filenames
left_zeroList = open(outDir + "kitti_objdet_left_t0.txt", 'w')
left_prev1List = open(outDir + "kitti_objdet_left_t-1.txt", 'w')
left_prev2List = open(outDir + "kitti_objdet_left_t-2.txt", 'w')
left_prev3List = open(outDir + "kitti_objdet_left_t-3.txt", 'w')
right_zeroList = open(outDir + "kitti_objdet_right_t0.txt", 'w')
right_prev1List = open(outDir + "kitti_objdet_right_t-1.txt", 'w')
right_prev2List = open(outDir + "kitti_objdet_right_t-2.txt", 'w')
right_prev3List = open(outDir + "kitti_objdet_right_t-3.txt", 'w')

for i in range(numData):
    print str(i) + " out of " + str(numData)
    #Find file
    filenames = findFilenames(baseDir, i)
    #Write out filenames
    if(not filenames):
        continue
    left_zeroList.write(filenames[0] + "\n")
    left_prev1List.write(filenames[1] + "\n")
    left_prev2List.write(filenames[2] + "\n")
    left_prev3List.write(filenames[3] + "\n")
    right_zeroList.write(filenames[4] + "\n")
    right_prev1List.write(filenames[5] + "\n")
    right_prev2List.write(filenames[6] + "\n")
    right_prev3List.write(filenames[7] + "\n")

    #Get labels for frame
    objs = readLabels(labelDir, i)
    #Write out idx list
    writeIdxList(objs, i, maxImageSizeX, maxImageSizeY, maxImageSizeX, maxImageSizeY, gtStream, dncStream)

#Close all files
left_zeroList.close()
left_prev1List.close()
left_prev2List.close()
left_prev3List.close()
right_zeroList.close()
right_prev1List.close()
right_prev2List.close()
right_prev3List.close()
gtStream.close()
dncStream.close()

