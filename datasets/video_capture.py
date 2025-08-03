import cv2
import torch
import random
import numpy as np
from decord import VideoReader, cpu

class VideoCapture:

    @staticmethod
    def load_frames_from_video(video_path,
                               num_frames,
                               sample='rand'):
        cap = cv2.VideoCapture(video_path)
        assert (cap.isOpened()), video_path
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []

        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))


        if sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        else:
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

        frames = []
        for index in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()

            if not ret:
                n_tries = 5
                for _ in range(n_tries):
                    ret, frame = cap.read()
                    if ret:
                        break

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)
            else:
                print(vlen, index, video_path)
                raise ValueError

        while len(frames) < num_frames:
            frames.append(frames[-1].clone())
            
        frames = torch.stack(frames).float() / 255
        cap.release()
        return frames, frame_idxs


    @staticmethod
    def load_frames_from_video_old(video_path,
                               max_frames,
                               sample='rand'):
        vreader = VideoReader(video_path, ctx=cpu(0))
        fps = vreader.get_avg_fps()
        f_start = 0
        f_end = len(vreader) - 1
        num_frames = f_end - f_start + 1
        
        if num_frames > 0:
            sample_fps = 1
            t_stride = int(round(float(fps)/sample_fps))
            all_pos = list(range(f_start, f_end+1, t_stride))
            if len(all_pos) > max_frames:
                sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
            else:
                sample_pos = all_pos
            patch_images = [torch.from_numpy(f).permute(2, 0, 1) for f in vreader.get_batch(sample_pos).asnumpy()]
            patch_images = torch.stack(patch_images).float()
            video = np.zeros((max_frames, 3, patch_images.shape[2], patch_images.shape[3]), dtype=np.float64)
            video[:patch_images.shape[0],...] = patch_images
            video = torch.from_numpy(video).float() / 255
            return video, sample_pos
    
    @staticmethod
    def load_frames_from_video_keyframe(video_path,
                               max_frames, sample_pos,
                               sample='rand'):
        cap = cv2.VideoCapture(video_path)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration
        
        frames = []

        for sec in sample_pos:
            sec_base = int(sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base)
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)

        cap.release()

        while len(frames) < max_frames:
            frames.append(frames[-1].clone())
    
        frames = torch.stack(frames).float() / 255
        return frames, sample_pos
    

