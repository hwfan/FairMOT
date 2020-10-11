from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
import pickle
from custom_multiprocessing import process_pool
import numpy as np
logger.setLevel(logging.INFO)

def split_range(num_parts, start_in, end_in):
    a = np.arange(start_in, end_in)
    res = np.array_split(a, num_parts)
    end = list(np.add.accumulate([len(x) for x in res]))
    start = [0] + end[:-1]
    ix = list(zip(start, end))
    return ix

def multiproc(args, gpu_list, data_length):
    cmd = ('CUDA_VISIBLE_DEVICES={gpu} python -u {binary} mot '
            '--custom-gpus 0 --paths-pkl {paths_pkl} --output-root {output_root} --load_model {load_model} '
            '--range {start} {end}' )
    # print(args.range)
    range_list = split_range(len(gpu_list), args.range[0], args.range[1])
    cmd_cwd_list = [(cmd.format(binary='demo.py', gpu=gpu, output_root=args.output_root, load_model=args.load_model, paths_pkl=args.paths_pkl, start=range_list[gpu_idx][0], end=range_list[gpu_idx][1]), '.') for gpu_idx, gpu in enumerate(gpu_list)]

    print('processes num: {:d}, data length: {:d}...'.format(len(cmd_cwd_list), data_length))

    pool = process_pool()
    pool.apply(cmd_cwd_list)
    pool.wait()

def demo(opt):
    paths = pickle.load(open(opt.paths_pkl, 'rb'))
    gpu_list = opt.custom_gpus.split(',')
    if opt.range[1] == -1:
        opt.range[1] = len(paths)
    tasks = paths[opt.range[0]:opt.range[1]]
    to_del_list = []
    for idx, path in enumerate(tasks):
        video_name = osp.splitext(osp.split(path)[1])[0]
        result_root = opt.output_root if opt.output_root != '' else '.'
        result_filename = os.path.join(result_root, '%s.txt' % video_name)
        if os.path.exists(result_filename):
            to_del_list.append(idx)
    deleted_tasks = [tasks[idx] for idx in range(len(tasks)) if idx not in to_del_list]
    opt.range[0] = 0
    opt.range[1] = len(deleted_tasks)
    if len(gpu_list) > 1:
        pickle.dump(deleted_tasks, open('cache.pkl','wb'))
        opt.paths_pkl = 'cache.pkl'
        multiproc(opt, gpu_list, len(deleted_tasks))
    else:
        for idx, path in enumerate(deleted_tasks):
            video_name = osp.splitext(osp.split(path)[1])[0]
            result_root = opt.output_root if opt.output_root != '' else '.'
            mkdir_if_missing(result_root)
            result_filename = os.path.join(result_root, '%s.txt' % video_name)
            if os.path.exists(result_filename):
                continue

            # logger.info('Starting tracking...')
            # dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
            dataloader = datasets.LoadVideo(path, opt.img_size)
            frame_rate = dataloader.frame_rate
            frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
            eval_seq(opt, dataloader, 'mot', result_filename,
                    save_dir=frame_dir, show_image=False, frame_rate=frame_rate, tasks_num=len(deleted_tasks), idx=idx)

            # if opt.output_format == 'video':
            #     output_video_path = osp.join(result_root, 'MOT16-03-results.mp4')
            #     cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
            #     os.system(cmd_str)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
