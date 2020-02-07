import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader import VideoLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

import ntpath
import os
import sys
from tqdm import tqdm
import time
from fn import getTime
import cv2

from pPose_nms import pose_nms, write_json

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

args = opt
args.dataset = 'coco'
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    videofile = args.video
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)
    
    if not len(videofile):
        raise IOError('Error: must contain --video')

    # Load input video (resize to [608*608] and transpose)
    data_loader = VideoLoader(videofile, batchSize=args.detbatch).start()
    (fourcc,fps,frameSize) = data_loader.videoinfo()
    print('Video info:\n\tfourcc = {}\n\tfps = {}\n\tframeSize = {}'.format(fourcc,fps,frameSize))

    # Load detection loader
    print('Loading YOLO model..')
    sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()# Get detection results for human
    det_processor = DetectionProcessor(det_loader).start() # Get human boxes (320*256) based on detection results
    
    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }
    duration = AverageMeter()

    # Data writer
    save_path = os.path.join(args.outputpath, ntpath.basename(videofile).split('.')[0]+'_pose.avi')
    writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()

    im_names_desc =  tqdm(range(data_loader.length()))
    batchSize = args.posebatch
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            # inps (tensor) [numBoxes, channel(3), height(320), width(256)] : human boxes of 320*256 with padding
            # orig_img (ndarray) [height, width, channel] : original video frame (one frame)
            # im_name (str) : e.g., 1.jpg, 2.jpg...
            # boxes (tensor) [numBoxes, 4] : points for human boxes
            # scores (tensor) [numBoxes, 1] : score for each human box
            # pt1 : (tensor) [numBoxes, 2] : topleft point of rescaled box
            # pt2 : (tensor) [numBoxes, 2] : bottomright point of rescaled box
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
            print('inps : {} {}'.format(type(inps), inps.shape))
            print('orig_img : {} {}'.format(type(orig_img), orig_img.shape))
            print('im_name : {} {}'.format(type(im_name), im_name))
            print('boxes : {} {}'.format(type(boxes), boxes.shape))
            print('scores : {} {}'.format(type(scores), scores.shape))
            print('pt1 : {} {}'.format(type(pt1), pt1.shape))
            print('pt2 : {} {}'.format(type(pt2), pt2.shape))
            if orig_img is None:
                break
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
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            # hm (tensor) [numHuman, 17, 80, 64] : human pose results by pose_model (17:num of keypoints)
            # print('hm.shape : {}'.format(hm.shape))
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)
            duration.update(getTime(start_time)[1], 1)

            hm = hm.cpu().data
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])

            ###############################
            #  Get keypoint data from hm  #
            ###############################
            if opt.matching:
                preds = getMultiPeakPrediction(
                    hm, pt1.numpy(), pt2.numpy(), opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
                result = matching(boxes, scores.numpy(), preds)
            else:
                preds_hm, preds_img, preds_scores = getPrediction(
                    hm, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
                result = pose_nms(
                    boxes, scores, preds_img, preds_scores)
            # result (list) : keypoint data * n
            # result[0] (dict) : keys() = ['keypoints', 'kp_score', 'proposal_score']
            # result[0].keypoints (tensor) [17, 2] : 17 keypoints
            # result[0].kp_score (tensor) [17, 1] : 17 scores for every keypoint
            # result[0].proposal_score (tensor) [1] : total pose score

            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)

        if args.profile:
            # TQDM
            # im_names_desc.set_description(
            # 'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
            #     dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
            # )
# -------------------------------------------------------------------------------------------------------
            # totalTime = np.mean(runtime_profile['dt']) + np.mean(runtime_profile['pt']) + np.mean(runtime_profile['pn'])
            # im_names_desc.set_description('Total time: {:.3f} ({:.0f} fps)'.format(totalTime, 1/totalTime))
# -------------------------------------------------------------------------------------------------------
            tt = duration.avg
            # tt = getTime(start_time)[1]
            im_names_desc.set_description(
                '[ det + pose ] time: {:.3f} ({}fps)'.format(tt, int(1.0 / tt))
            )
    # print(runtime_profile)
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
    while(writer.running()):
        pass
    writer.stop()
    final_result = writer.results()
    write_json(final_result, args.outputpath)

