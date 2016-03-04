import os
import sys
import gc
import numpy as np
import scipy.sparse as sparse
import lmdb
import argparse

# IPython is for debugging
import IPython

parser = argparse.ArgumentParser(description="Create an LMDB file from an input pvp file.",
                                 usage="create_lmdb_from_pvp.py -p <pvp_file> -i <img_list_txt_file> -o <output_file>")

parser.add_argument("-p", "--pvp-file", type=str, required=True, help="input PVP file")
parser.add_argument("-i", "--image-list", type=str, required=True, help="text list of image files")
parser.add_argument("-o", "--output-file", type=str, required=True, help="output file name")
parser.add_argument("-pv", "--pv-path", type=str, required=True, help="path to PetaVision python folder")
parser.add_argument("-ca", "--caffe-path", type=str, required=True, help="path to Caffe python folder")
parser.add_argument("-s", "--image-label-pos", type=int, default=0, required=False, help="location of image label in image_list file")
parser.add_argument("-w", "--write-progress", type=int, default=0, required=False, help="interval to write out progress")
parser.add_argument("-v", "--validation-num", type=int, default=0, required=False, help="number of images to use for validation set (0 is training only)")
parser.add_argument("-a", "--validation-file", type=str, default="./validation_lmdb", required=False, help="validation set filename")
parser.add_argument("-c", "--cpu-bias", action="store_true", required=False, help="if set, program will try to save memory by pushing computation to file IO")
parser.add_argument("-l", "--label-output", action="store_true", required=False, help="set flag to create label text file")
parser.add_argument("-m", "--mode", type=str, default="TRAIN", required=False, help="mode should be TRAIN or TEST")

def cifarList2Vec(file_loc,label_pos):
    label_list = []
    with open(file_loc, 'r') as f:
        for line in f:
            line_arry = line.split('/')
            label_list.append(line_arry[label_pos])
    return np.array(label_list)

def write_lmdb(filename, map_size, progress_write, data, labels, index_list):
    with lmdb.open(filename, map_size=map_size) as env:
        with env.begin(write=True) as txn:   # txn is a Transaction object
            for i in index_list:
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = data.shape[1]
                datum.height = data.shape[2]
                datum.width = data.shape[3]
                #logActivities = np.copy(data[i])
                #logActivities[np.nonzero(logActivities)] = np.log(logActivities[np.nonzero(logActivities)])
                datum = caffe.io.array_to_datum(data[i], int(labels[i]))
                keystr = '{:04}'.format(i)
                txn.put(keystr.encode('ascii'), datum.SerializeToString())

def main(args):
    """
        Entry point.
    """

    # Each image has num_perturbations frames
    num_perturbations = 50
    # max_frames is the number of input images * the number of perturbations
    max_frames = 250000 #float('inf')

    assert(args.write_progress >= 0)
    assert(args.validation_num >= 0)

    if os.path.isdir(os.path.dirname(args.output_file)):
        if not os.path.exists(os.path.dirname(args.output_file)):
            os.makedirs(os.path.dirname(args.output_file))

    #TODO: make TRAIN efficient like TEST is
    if args.mode.upper() == "TRAIN":
        pvData = readpvpfile(args.pvp_file, args.write_progress, max_frames)
        num_imgs = pvData['values'].shape[0]
        nf = pvData['header']['nf']
        ny = pvData['header']['ny']
        nx = pvData['header']['nx']
        pvActivities = np.array(pvData['values'].todense()).reshape((num_imgs,nf,ny,nx)).astype('float')
    else:
        pvHeader = readpvpheader(args.pvp_file)

    if max_frames == float('inf'):
        labels = cifarList2Vec(args.image_list, args.image_label_pos).astype(np.int64)
    else:
        labels = cifarList2Vec(args.image_list, args.image_label_pos).astype(np.int64)[0:max_frames]

    if args.label_output:
        if os.path.isdir(os.path.dirname(args.output_file)):
            label_file = os.path.dirname(args.output_file)+'labels.txt'
        else:
            label_file = 'labels.txt'
        label_out_text = open(label_file,'w')
        for label in labels:
            label_out_text.write(str(label)+'\n')
        label_out_text.close()

    if args.mode.upper() == "TRAIN":
        assert(labels.shape[0] == pvActivities.shape[0])

        # Database size set to be 10x bigger than needed
        # as suggested in http://deepdish.io/2015/04/28/creating-lmdb-in-python/
        map_size = pvActivities[0:num_imgs-args.validation_num].nbytes * 10

        # Create initial set
        write_lmdb(args.output_file, map_size, args.write_progress, pvActivities, labels, range(num_imgs - args.validation_num))

        # Create validation set if required
        if (args.validation_num > 0):
            print "-------\nWriting validation set\n-------"
            map_size = pvActivities[num_imgs-args.validation_num:num_imgs].nbytes * 10
            write_lmdb(args.validation_file, map_size, args.write_progress, pvActivities, labels, range(num_imgs - args.validation_num, num_imgs))

    elif args.mode.upper() == "TEST":
        if not args.cpu_bias:
            pvData = readpvpfile(args.pvp_file, args.write_progress, max_frames)
        for num_avg in range(1, num_perturbations+1):
            num_images = int(max_frames/float(num_perturbations))
            nf = pvHeader['nf']
            ny = pvHeader['ny']
            nx = pvHeader['nx']
            out_mat = np.zeros((num_images, nf, ny, nx))
            for start_idx in range(num_avg):
                if args.cpu_bias:
                    pvData = readpvpfile(args.pvp_file, args.write_progress, max_frames, start_idx, num_perturbations)
                    out_mat += np.array(pvData['values'].todense()).reshape(num_images, nf, ny, nx)
                    del pvData
                else:
                    out_mat += np.array(pvData['values'].tocsc()[start_idx:max_frames:num_perturbations].todense()).reshape(num_images, nf, ny, nx)
                print "-->Loaded sample number "+str(start_idx)+" out of "+str(num_avg)+" for all "+str(pvData['values'].shape[0])+" frames."
                gc.collect()
            if num_avg > 0:
                out_mat /= float(num_avg)
            map_size = out_mat.nbytes * 10
            write_lmdb(args.output_file+'_samples_'+str(num_avg), map_size, args.write_progress, out_mat, labels, range(num_images))
            print "---->Wrote output for "+str(num_avg)+" perturbations."

    else:
        assert False, "Input mode must be TRAIN or TEST."

if __name__ == "__main__":
    args = parser.parse_args()

    # Add paths
    sys.path.insert(0, os.path.abspath(args.pv_path))
    sys.path.insert(0, os.path.abspath(args.caffe_path))
    
    # Import aux libraries
    from pvtools import *
    import caffe

    main(args)
