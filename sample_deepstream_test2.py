#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
sys.path.append('../')
import platform
import configparser

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.platform_info import PlatformInfo
from common.bus_call import bus_call

import pyds
import json

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_BATCH_TIMEOUT_USEC = 33000

LOG_FILE_PATH = "/home/jzching/testdeepstream/detection_log.jsonl"

import math

# Helper: calculate IoU between two boxes
def get_iou(box1, box2):
    x1 = max(box1['left'], box2['left'])
    y1 = max(box1['top'], box2['top'])
    x2 = min(box1['left'] + box1['width'], box2['left'] + box2['width'])
    y2 = min(box1['top'] + box1['height'], box2['top'] + box2['height'])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def is_inside(vehicle_bbox, child_bbox):
    cx = child_bbox['left'] + child_bbox['width'] / 2
    cy = child_bbox['top'] + child_bbox['height'] / 2
    return (vehicle_bbox['left'] <= cx <= vehicle_bbox['left'] + vehicle_bbox['width']) and \
           (vehicle_bbox['top'] <= cy <= vehicle_bbox['top'] + vehicle_bbox['height'])

def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    results = []

    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # collect detections
        l_obj = frame_meta.obj_meta_list
        vehicles = []
        brand_dets = []
        plate_dets = []

        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # fetch the label (already a Python str for detectors)
            label = obj_meta.obj_label or ""
            bbox = {
                "left":  obj_meta.rect_params.left,
                "top":   obj_meta.rect_params.top,
                "width": obj_meta.rect_params.width,
                "height":obj_meta.rect_params.height
            }

            # DEBUG: see every detection come through
            print(f"[DEBUG] Label:{label}, CompID:{obj_meta.unique_component_id}, ClassID:{obj_meta.class_id}, BBox:{bbox}")
            # PGIE vehic le boxes
            if obj_meta.class_id in (2,3,5,7):
                vehicles.append({
                    "object_id":    obj_meta.object_id,
                    "class_id":     obj_meta.class_id,
                    "label":        label,
                    "bbox":         bbox,
                    "brand":        None,
                    "license_plate":None
                })
            # SGIE car ^`^qplate detector (unique-id=2)
            elif obj_meta.unique_component_id == 2:
                plate_dets.append({
                    "label":      label,
                    "confidence": obj_meta.confidence,
                    "bbox":       bbox
                })
            # SGIE brand detector (unique-id=3)
            elif obj_meta.unique_component_id == 3:
                brand_dets.append({
                    "label":      label,
                    "confidence": obj_meta.confidence,
                    "bbox":       bbox
                })

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # spatially link plates & brands into vehicles
        IOU_THRESH = 0.1
        for v in vehicles:
            for b in brand_dets:
                iou = get_iou(v["bbox"], b["bbox"])
                inside = is_inside(v["bbox"], b["bbox"])
                print(f"[MATCH] V{v['object_id']}  ^f^t Brand {b['label']} IoU={iou:.2f} CenterIn={inside}")
                if iou > IOU_THRESH or inside:
                    v["brand"] = {
                        "name": b["label"],
                        "confidence": b["confidence"]
                    }

            for p in plate_dets:
                iou = get_iou(v["bbox"], p["bbox"])
                inside = is_inside(v["bbox"], p["bbox"])
                print(f"[MATCH] V{v['object_id']}  ^f^t Plate {p['label']} IoU={iou:.2f} CenterIn={inside}")
                if iou > IOU_THRESH or inside:
                    v["license_plate"] = {
                        "text": p["label"],
                        "confidence": p["confidence"],
                        "bbox": p["bbox"]
                    }
        if vehicles:
            # record this frame
            results.append({
                "frame_number": frame_meta.frame_num,
                "objects":      vehicles
            })

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    if results:
        # Append to JSONL
        try:
            with open(LOG_FILE_PATH, "a") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            print(f"Error writing to file: {e}")

    return Gst.PadProbeReturn.OK

def main(args):
    # Check input arguments
    if(len(args) < 2):
        sys.stderr.write("usage: %s <h264_elementary_stream>\n" % args[0])
        sys.exit(1)

    platform_info = PlatformInfo()
    Gst.init(None)

    # Create gstreamer elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    input_uri = args[1]
    use_uri_source = input_uri.startswith("rtsp://") or input_uri.startswith("http://") or input_uri.startswith("https://")
    # Create source based on URI or file
    if use_uri_source:
        print("Creating uridecodebin for RTSP/URI input")
        source = Gst.ElementFactory.make("uridecodebin", "uri-source")
        if not source:
            sys.stderr.write("Unable to create uridecodebin\n")
        source.set_property("uri", input_uri)
    else:
        print("Creating filesrc + h264parse + decoder for local file")
        source = Gst.ElementFactory.make("filesrc", "file-source")
        h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")

        if not source or not h264parser or not decoder:
            sys.stderr.write(" Unable to create file pipeline components\n")
        source.set_property("location", input_uri)

    # Now add the elements to the pipeline only once
    pipeline.add(source)
    if not use_uri_source:
        pipeline.add(h264parser)
        pipeline.add(decoder)

    # Create other elements
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    sgie1 = Gst.ElementFactory.make("nvinfer", "secondary1-nvinference-engine")
    sgie2 = Gst.ElementFactory.make("nvinfer", "secondary2-nvinference-engine")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    # Set width, height, and batch properties for the streammux element
    streammux.set_property('width', 1920)  # Set the width to match your video input
    streammux.set_property('height', 1080)  # Set the height to match your video input
    streammux.set_property('batch-size', 1)  # Number of frames per batch
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property('live-source', 1)

    #Set properties of pgie and sgie
    pgie.set_property('config-file-path', "/home/jzching/testdeepstream/config_infer_primary_yolo11.txt")
    sgie1.set_property('config-file-path', "/home/jzching/testdeepstream/config_infer_yolo.txt")
    sgie2.set_property('config-file-path', "/home/jzching/testdeepstream/config_infer_yolo_brand.txt")

    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read('dstest2_tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)

    # Create sink
    if platform_info.is_integrated_gpu():
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
    else:
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

    # Add the remaining elements
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie1)
    pipeline.add(sgie2)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    # Linking elements
    print("Linking elements in the Pipeline \n")
    sinkpad = streammux.request_pad_simple("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")

    if use_uri_source:
        def cb_newpad(decodebin, pad, data):
            print("In cb_newpad\n")
            caps = pad.get_current_caps()
            string = caps.to_string()
            print(f"Pad added with caps: {string}")
            if pad.get_direction() != Gst.PadDirection.SRC:
                return
            pad.link(sinkpad)

        source.connect("pad-added", cb_newpad, None)
    else:
        source.link(h264parser)
        h264parser.link(decoder)

        srcpad = decoder.get_static_pad("src")
        if not srcpad:
            sys.stderr.write(" Unable to get source pad of decoder \n")
        srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie1)
    sgie1.link(sgie2)
    sgie2.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # Create the event loop and run the pipeline
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    osdsinkpad = nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except:
        pass

    # Cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))