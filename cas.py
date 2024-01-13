#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from tempfile import TemporaryDirectory
import logging
import cv2
import asyncio
from pathlib import Path
from wand.image import Image

log = logging.getLogger('distortioner')

ap = ArgumentParser(
    description = 'a script that makes memetic distorted videos'
)

class TicketedDict(dict):
    def __init__(self, *args, **kwargs):
        self._ni = 0
        self._log = logging.getLogger('TicketedDict')
        self.event = asyncio.Event()
        super().__init__(*args, **kwargs)

    def has_next(self):
        self._log.debug("has_next is %s for %i", self._ni in self, self._ni)
        return self._ni in self

    async def wait(self):
        self._log.debug("requested wait")
        while not self.has_next():
            await self.event.wait()
            self._log.debug("wait broke")
            self.event.clear()

    async def pop(self):
        self._log.debug("requested pop")
        await self.wait()
        ret = super().pop(self._ni)
        self._ni += 1
        return ret

    def notify(self, *args, **kwargs):
        self._log.debug("requested notify")
        return self.event.set(*args, **kwargs)

ap.add_argument('input')
ap.add_argument('output')
ap.add_argument(
    '--distort-percentage','--distort-pct',  '-d', 
    default = 60.0,
    type = float,
    help = 'Percentage of image distortion.'
)
ap.add_argument(
    '--distort-percentage-end','--distort-pct-end',  '-D', 
    default = None,
    type = float,
    help = 'If specified, distortion percentage will gradually change towards specified percentage.'
)
ap.add_argument(
    '--distort-end','--distort',  '-E', 
    default = None,
    type = float,
    help = 'If specified, distortion change will happen only up to specific video progress.'
)
ap.add_argument(
    '--vibrato-frequency','--vibrato-freq',  '-f',
    default = 5.0,
    type = float,
    help = 'Modulation frequency in Hertz. Range is 0.1 - 20000.0. Default value is 5.0 Hz.'
)
ap.add_argument(
    '--vibrato-modulation-depth','--vibrato-depth',  '-m',
    default = 0.5,
    type = float,
    help = 'Depth of modulation as a percentage. Range is 0.0 - 1.0. Default value is 0.5.'
)
ap.add_argument(
    '--debug',
    action = 'store_true',
    default = False,
    help = 'Print debugging messages.'
)
def process_image(source, destination, distort):
    log.debug("distorting: src:'%s', dst:'%s' ", source, destination)
    with Image(filename=source) as original:
        dst_width = int(original.width*(distort / 100.))
        dst_height = int(original.height*(distort / 100.))
        log.debug("dst:'%s' size:%ix%i", destination, dst_width, dst_height)
        with original.clone() as distorted, \
             open(destination, mode='wb') as out:
            distorted.liquid_rescale(dst_width, dst_height)
            distorted.resize(original.width, original.height)
            distorted.save(out)

async def process_frames(coro, queue_in, out_pile):
    while True:
        frame_data = await queue_in.get()
        nr = frame_data.pop('nr')
        log.debug("processing frame '%s'", frame_data)
        await asyncio.to_thread(coro, **frame_data)
        queue_in.task_done()
        out_pile[nr]=frame_data['destination']
        log.debug("put item to output pile, notifying")
        out_pile.notify()
        log.debug("notified")
        

async def read_frames(capture, frames_distorted, frames_original, queue, tasks, distort_start, distort_end=None, distort_end_frame=None):
    frames_read = 0
    if distort_end_frame is None:
        distort_end_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        read_ok, frame = capture.read()
        if not read_ok:
            break
        log.debug("reading frame %i", frames_read)
        if distort_end is not None:
            distort = distort_start \
                + (distort_end - distort_start) \
                * (min(frames_read, distort_end_frame) / distort_end_frame ) 
        frame_filename = f'frame_{str(frames_read).zfill(32)}.png'
        frame_original = str(Path(frames_original)/frame_filename)
        frame_distorted = str(Path(frames_distorted)/frame_filename)
        cv2.imwrite(frame_original, frame)
        log.debug("saving frame %i: filename: %s", frames_read, frame_filename)
        await queue.put({
            'source': frame_original,
            'destination': frame_distorted,
            'distort': distort,
            'nr': frames_read
        })
        frames_read += 1
    log.info("waiting for queue to empty")
    await queue.join()
    log.info("quitting")
    for worker in tasks:
        worker.cancel()

async def write_frames(output, pile):
    while True:
        log.debug("getting next item ...")
        frame_distorted = await pile.pop()
        log.info("writing frame '%s'", frame_distorted)
        newframe = cv2.imread(frame_distorted)
        output.write(newframe)
        log.debug("finished frame '%s'", frame_distorted)


async def distort_video(capture, output, distort_start, distort_end=None, distort_end_frame=None):
    distort = distort_start
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_pile = TicketedDict()
    with \
            TemporaryDirectory() as frames_distorted, \
            TemporaryDirectory() as frames_original:
        log.debug(frames_distorted)
        log.debug(frames_original)
        log.debug("creating queues")
        capture_queue = asyncio.Queue(20)
        output_queue = asyncio.PriorityQueue()
        workers = [ asyncio.create_task(process_frames(process_image, capture_queue, output_pile)) for i in range(10) ]
        workers += [ asyncio.create_task(write_frames(output, output_pile)) ]
        generator = asyncio.create_task(read_frames(
            capture, frames_distorted, frames_original, capture_queue, workers, distort_start, distort_end, distort_end_frame
        ))
        await asyncio.gather(generator, *workers, return_exceptions=True)

    log.debug('done with distorting video frames')

def main():
    args = ap.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    log.debug(args)
    if args.distort_percentage <= 0.0:
        raise ValueError("--distort_percentage must be positive number")

    capture = cv2.VideoCapture(args.input)
    fps = capture.get(cv2.CAP_PROP_FPS)
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = args.distort_end or int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("video: name:'%s', fps:%i, frames:%i, size:%ix%i", args.input, fps, frames, video_width, video_height)
    output = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (video_width, video_height))
    asyncio.run(distort_video(capture, output, args.distort_percentage, args.distort_percentage_end, frames-1))
    capture.release()
    output.release()


if __name__ == '__main__':
    main()

