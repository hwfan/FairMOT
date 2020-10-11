import os
import time
import json
from tqdm import tqdm
import numpy as np
import ipdb
import pickle
import motmetrics as mm
import argparse

def calc_iou(bbox1, bbox2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)
    
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    
    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))
    
    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w
    
    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return intersect / union
def main(model, thres):
    video_shape_dict = json.load(open('video_shape_dict.json','r'))
    vt_box_dict_gt = pickle.load(open('vt_box_dict_gt.pkl','rb'))
    fps_dict = json.load(open('fps_dict.json','r'))
    test_videos = pickle.load(open('test_choose_videos_20200909.pkl','rb'))
    res_file = '%s.txt' % model
    if os.path.exists(res_file):
        os.system('rm %s' % res_file)
    result_dir = model
    result_list = np.empty((0,5))

    collect_txt = []
    for result_txt_name in tqdm(os.listdir(result_dir)):
        if result_txt_name.endswith('.txt'):
            result_txt_path = os.path.join(result_dir, result_txt_name)
            collect_txt.append(result_txt_path)

    for result_txt_path in sorted(collect_txt):
        vt_box_dict_predicted = dict()
        video_name = os.path.splitext(os.path.split(result_txt_path)[1])[0]
        print(video_name)
        if video_name not in test_videos:
            continue
        fps = fps_dict[video_name]
        height, width, _ = video_shape_dict[video_name]
        id_mapping = dict()
        id_now = 1
        for line in tqdm(open(result_txt_path,'r')):
            frame_num, predicted_human_id, x, y, w, h, _, _, _, _ = line.strip().split(',')
            frame_num, predicted_human_id, x, y, w, h = int(frame_num), int(predicted_human_id), float(x), float(y), float(w), float(h)
            if predicted_human_id not in id_mapping:
                id_mapping[predicted_human_id] = id_now
                id_now += 1
            else:
                predicted_human_id = id_mapping[predicted_human_id]
            x = int(x / 1920 * width)
            y = int(y / 1080 * height)
            w = int(w / 1920 * width)
            h = int(h / 1080 * height)
            sec = frame_num // fps
            remained_frames = frame_num % fps
            if remained_frames == 0:
                if sec >= 2:
                    key = video_name+'.'+str(sec+900)
                    if key not in vt_box_dict_predicted:
                        vt_box_dict_predicted[key] = dict()
                    if predicted_human_id not in vt_box_dict_predicted[key]:
                        vt_box_dict_predicted[key][predicted_human_id] = [x, y, x+w, y+h]   
        print(id_now)
        acc = mm.MOTAccumulator(auto_id=True)
        for each_sec in tqdm(range(902, 1799)):
            key = video_name+'.'+str(each_sec)

            if key in vt_box_dict_gt:
                sub_dict_gt = vt_box_dict_gt[key]
                gt_human_ids = list(sub_dict_gt.keys())
                gt_human_bboxes = np.array(list(sub_dict_gt.values()))
            else:
                gt_human_ids = []
                gt_human_bboxes = np.empty((0,4),dtype=int)

            if key in vt_box_dict_predicted:
                sub_dict = vt_box_dict_predicted[key]
                pred_human_ids = list(sub_dict.keys())
                pred_human_bboxes = np.array(list(sub_dict.values()))
            else:
                pred_human_ids = []
                pred_human_bboxes = np.empty((0,4),dtype=int)
            # iou_matrix = 1-calc_iou(gt_human_bboxes, pred_human_bboxes)

            gt_human_bboxes[:, 2] -= gt_human_bboxes[:, 0]
            gt_human_bboxes[:, 3] -= gt_human_bboxes[:, 1]

            pred_human_bboxes[:, 2] -= pred_human_bboxes[:, 0]
            pred_human_bboxes[:, 3] -= pred_human_bboxes[:, 1]

            iou_mat = mm.distances.iou_matrix(gt_human_bboxes, pred_human_bboxes, max_iou=(0.5 if thres else 1.0))
            # ipdb.set_trace()
            acc.update(
                gt_human_ids,                     # Ground truth objects in this frame
                pred_human_ids,                  # Detector hypotheses in this frame
                iou_mat
            )
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1', 'idp', 'idr'], name='acc')
        strsummary = mm.io.render_summary(summary)
        
        print(strsummary)
        # ipdb.set_trace()
        with open(res_file, 'a') as f:
            f.write(video_name)
            f.write('\n\r')
            f.write(strsummary)
            f.write('\n\r')
    
        result_list = np.concatenate((result_list, summary.to_numpy()[:, 1:]), axis=0)
    total_res = result_list.mean(axis=0)
    print('-- total --')
    print(total_res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--thres', action='store_true')
    args = parser.parse_args()
    main(args.model, args.thres)