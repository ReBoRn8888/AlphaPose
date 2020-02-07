import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data

import cv2
import numpy as np
import pickle as pkl
from opt import opt
from summary import summary

from dataloader import ImageLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *
from SPPE.src.utils.eval import getPrediction, getMultiPeakPrediction
from matching import candidate_reselect as matching

import os
import sys
from tqdm import tqdm
import time
from fn import getTime

from pPose_nms import pose_nms, write_json

args = opt
args.dataset = 'coco'
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    inputpath = args.inputpath
    inputlist = args.inputlist
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    if len(inputlist):
        im_names = open(inputlist, 'r').readlines()
    elif len(inputpath) and inputpath != '/':
        for root, dirs, files in os.walk(inputpath):
            im_names = files
    else:
        raise IOError('Error: must contain either --indir/--list')

    tmp = []
    for k in range(10):
        # Load input images (resize + transpose)
        data_loader = ImageLoader(im_names, batchSize=args.detbatch, format='yolo').start()

        # Load detection loader
        print('Loading YOLO model..')
        sys.stdout.flush()
        det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start() # Get detection results for human
        det_processor = DetectionProcessor(det_loader).start() # Get human boxes based on detection results
        
        # Load pose model
        pose_dataset = Mscoco()
        if args.fast_inference:
            pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        # print('pose_model : \n{}'.format(summary(model=pose_model, input_size=(3, 320, 256))))
        pose_model.cuda()
        pose_model.eval()

        runtime_profile = {
            'dt': [], # time for human detection
            'pt': [], # time for pose estimation
            'pn': []  # time for writing video frames
        }

        # Init data writer
        writer = DataWriter(args.save_video).start()

        data_len = data_loader.length()
        print('data_len = {}'.format(data_len))
        im_names_desc = tqdm(range(data_len))

        batchSize = args.posebatch

        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                # Human detection
                (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
                # inps (tensor) [numBoxes, channel, height, width] : human boxes of 320*256 with padding
                # orig_img (ndarray) [height, width, channel] : original video frame (one frame)
                # im_name (str) : e.g., 1.jpg, 2.jpg...
                # boxes (tensor) [numBoxes, 4] : points for human boxes
                # scores (tensor) [numBoxes, 1] : score for each human box
                # pt1 : (tensor) [numBoxes, 2] : topleft point of rescaled box
                # pt2 : (tensor) [numBoxes, 2] : bottomright point of rescaled box
                print('==================================================')
                print('Human boxes = {}'.format(inps.shape))
                print('Image shape = {}'.format(orig_img.shape))
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                    continue

                ckpt_time, det_time = getTime(start_time)
                runtime_profile['dt'].append(det_time)

                # Pose Estimation
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
                    # for cnt in range(len(inps_j)):
                    #     cv2.imwrite('examples/res/vis/human_{}.jpg'.format(cnt), np.transpose(inps_j[cnt].cpu().numpy(), (2, 1, 0)))
                    hm_j = pose_model(inps_j)
                    hm.append(hm_j)
                hm = torch.cat(hm)
                print('hm.shape = {}'.format(hm.shape))
                ckpt_time, pose_time = getTime(ckpt_time)
                runtime_profile['pt'].append(pose_time)

                # Write video frames
                hm = hm.cpu()
                writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
                if opt.matching:
                    preds = getMultiPeakPrediction(
                        hm, pt1.numpy(), pt2.numpy(), opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
                    result = matching(boxes, scores.numpy(), preds)
                else:
                    preds_hm, preds_img, preds_scores = getPrediction(
                        hm, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
                    result = pose_nms(
                        boxes, scores, preds_img, preds_scores)

                print('Pose detected = {}'.format(len(result))) # the number of people detected in this img
                # print('result[0].keys() = {}'.format(result[0].keys()))
                # print('result[0].keypoints.shape = {}'.format(result[0]['keypoints'].shape)) # (x, y) for 17 keypoints
                # print('result[0].kp_score.shape = {}'.format(result[0]['kp_score'].shape)) # 17 keypoint scores
                # print('result[0].proposal_score.shape = {}'.format(result[0]['proposal_score'].shape)) # total score for this pose estimation
                # result.len = 5
                # result[0].keys() = dict_keys(['keypoints', 'kp_score', 'proposal_score'])
                # result[0].keypoints.shape = torch.Size([17, 2])
                # result[0].kp_score.shape = torch.Size([17, 1])
                # result[0].proposal_score.shape = torch.Size([1])
                ###############################################
                #  keypoint results for COCO (17 body parts)  #
                #             Add (#17: Neck) here            #
                ###############################################
                # kpName = {
                #   0 : "Nose",
                #   1 : "LEye",
                #   2 : "REye",
                #   3 : "LEar",
                #   4 : "REar",
                #   5 : "LShoulder",
                #   6 : "RShoulder",
                #   7 : "LElbow",
                #   8 : "RElbow",
                #   9 : "LWrist",
                #   10 : "RWrist",
                #   11 : "LHip",
                #   12 : "RHip",
                #   13 : "LKnee",
                #   14 : "Rknee",
                #   15 : "LAnkle",
                #   16 : "RAnkle",          
                #   17 : "Neck",            
                # }
                # print('result : {}'.format(result[0]))
                # save result to .pkl
                with open('result.pkl', 'wb') as f:
                    pkl.dump(result, f)

                ckpt_time, post_time = getTime(ckpt_time)
                runtime_profile['pn'].append(post_time)
            
            if args.profile:
                # TQDM
                # im_names_desc.set_description(
                # 'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
                #   dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                # )
                print('det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
                    dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn'])))
                wholeTime = getTime(start_time)[1]
                print('[det + pose + post]: {:.3f} ({:.2f} fps)'.format(wholeTime, 1/wholeTime))
                tmp.append(wholeTime)
    tmp = np.array(tmp)
    print('K fold: {:.3f} ({:.0f} fps)'.format(tmp.mean(), 1/tmp.mean()))

    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
    while(writer.running()):
        pass
    writer.stop()
    final_result = writer.results()
    write_json(final_result, args.outputpath)

