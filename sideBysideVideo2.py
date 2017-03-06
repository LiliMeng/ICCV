# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:25:42 2016

@author: jimmy

play video
"""

import numpy as np
import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--file1', type = str, required = True)
parser.add_argument('--file2', type = str, required = True)
parser.add_argument('--start_frame', type = int, required = True)
parser.add_argument('--end_frame', type = int, required = True)
parser.add_argument('--play_step', type = int, required = True)
parser.add_argument('--fps', type = float, required = True)
parser.add_argument('--gap', type = int, required = True)
parser.add_argument('--overlay_box_size', type = int, required = True)
parser.add_argument('--is_half_size', type = int, required = True)
parser.add_argument('--save_folder', type = str, required = True)


args = parser.parse_args()


names = ['/Users/jimmy/Data/DRP/WWoS_soccer_2014/Camera1.mov',
        '/Users/jimmy/Data/DRP/WWoS_soccer_2014/Camera2.mov',
        '/Users/jimmy/Data/DRP/WWoS_soccer_2014/Camera3.mov']

print('default video names:')
for name in names:
    print('%s\n', name)


file1 = args.file1
file2 = args.file2
start_frame = args.start_frame
end_frame = args.end_frame
play_step = args.play_step
fps = args.fps
gap = args.gap
overlay_box_size = args.overlay_box_size
is_half_size = args.is_half_size
save_folder = args.save_folder


"""
file1 = 'camera_selection_prof_test.csv'
file2 = 'c3d_pred.csv'
play_step = 4
fps = 60.0
gap = 10
"""

caps = []
for name in names:
    cap = cv2.VideoCapture(name)
    caps.append(cap)

selections1 = np.loadtxt(file1, dtype = int, delimiter = ',', skiprows = 1)
selections2 = np.loadtxt(file2, dtype = int, delimiter = ',', skiprows = 1)
    
length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))


print('FPS is %f, player step is %d\n' % (fps, play_step))

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

def millisecondsFromIndex(index, fps):
    return index * 1000.0/fps;

def getFrameByIndex(cap, index, fps = 60.0):
    t = millisecondsFromIndex(index, fps)
    cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, t)
    ret, frame = cap.read()
    if ret == True:
        return frame
    else:
        return None

def segment_to_per_frame(segment):
    [N, M] = segment.shape
    print N
    fns, labels = [], []
    for i in range(N):
        start_fn, end_fn, camera_id = segment[i][0], segment[i][1], segment[i][2]
        for j in range(start_fn-1, end_fn):
            fns.append(j)
            labels.append(camera_id)
        if start_fn == end_fn:
            fns.append(start_fn)
            labels.append(camera_id)
            
    assert(len(fns) == len(labels))
    print len(fns)
    N = len(fns)
    per_frame = np.zeros((N, 2), dtype = np.int64)
    per_frame[:,0] = np.array(fns)
    per_frame[:,1] = np.array(labels)
    return per_frame

def horizontal_concat_frames(frame1, frame2, gap):
    h, w, c = frame1.shape
    frame = np.zeros((h, w*2+gap, c), dtype = frame1.dtype)
    frame[:,range(0,w),:] = frame1
    frame[:,range(w+gap,2*w+gap),:] = frame2
    return frame

def overlay_box(frame, sz):
    h, w, c = frame.shape
    frame[range(0, sz),:, :] = 255
    frame[range(h-sz, h), :, :] = 255
    frame[:, range(0, sz), :] = 255
    frame[:, range(w-sz, w), :] = 255    
    return frame

fn_selections1 = segment_to_per_frame(selections1)     
fn_selections2 = segment_to_per_frame(selections2)

print len(fn_selections1)
print len(fn_selections2)

assert(len(fn_selections1) == len(fn_selections2))

[N, M] = fn_selections1.shape
# check frame number is the same
for i in range(N):
    assert(fn_selections1[i][0] == fn_selections2[i][0])

for i in range(0, N, play_step):
    fn, sel1, sel2 = fn_selections1[i][0], fn_selections1[i][1], fn_selections2[i][1]
    if fn not in range(start_frame, end_frame):
        continue
    frame1 = getFrameByIndex(caps[sel1], fn, fps)
    frame2 = getFrameByIndex(caps[sel2], fn, fps)
    if is_half_size:
        h, w, c = frame1.shape
        frame1 = cv2.resize(frame1, (w/2, h/2))
        frame2 = cv2.resize(frame2, (w/2, h/2))
    frame1 = overlay_box(frame1, overlay_box_size)
    frame2 = overlay_box(frame2, overlay_box_size)
    
    frame = horizontal_concat_frames(frame1, frame2, gap)    
    name = save_folder + '/%08d.jpg'%fn
    cv2.imwrite(name, frame)
    print('save to %s\n' % name)    

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
print('Done')

    
    
