import gi
import sys
import pyds
import cv2
import numpy as np
import os
from gi.repository import Gst, GLib

gi.require_version('Gst', '1.0')
Gst.init(None)

# Output folder for cropped plates
output_folder = "/home/jzching/testdeepstream/output/"
os.makedirs(output_folder, exist_ok=True)

def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return pyds.Gst.PadProbeReturn.OK

    # Get batch metadata
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    frame_number = 0

    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        frame_number += 1
        l_obj = frame_meta.obj_meta_list

        while l_obj:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            
            # Print the class_id to check what value it has
            print(f"Object Class ID: {obj_meta.class_id}")

            # Assuming class ID 2 is the license plate
            if obj_meta.class_id == 0:
                top = int(obj_meta.rect_params.top)
                left = int(obj_meta.rect_params.left)
                width = int(obj_meta.rect_params.width)
                height = int(obj_meta.rect_params.height)

                frame_image = extract_frame_image(gst_buffer, frame_meta.source_id)
                plate_crop = frame_image[top:top+height, left:left+width]
                plate_filename = f"{output_folder}/plate_{frame_number}_{obj_meta.object_id}.jpg"
                cv2.imwrite(plate_filename, plate_crop)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return pyds.Gst.PadProbeReturn.OK


def extract_frame_image(gst_buffer, batch_id):
    # Extract image from frame buffer (DeepStream uses GPU memory, need to copy to host memory)
    surface = pyds.get_nvds_buf_surface(hash(gst_buffer), batch_id)
    frame_rgba = np.array(surface, copy=True, dtype=np.uint8)
    frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
    return frame_bgr

def attach_probe_to_osd(pipeline):
    osd = pipeline.get_by_name("nvosd")
    if not osd:
        print("Error: 'nvosd' element not found in pipeline.")
        return
    osd_sink_pad = osd.get_static_pad("sink")
    if osd_sink_pad:
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
        print("Probe attached to OSD sink pad.")

def main():
    config_path = "/home/jzching/testdeepstream/sample_deepstream_app_config.txt"

    # Use subprocess to run deepstream-app with the config
    import subprocess
    subprocess.run(["deepstream-app", "-c", config_path])

if __name__ == '__main__':
    main()

#python3 /home/jzching/testdeepstream/extract_plate.py (run in deepstream-yolo directory in docker)

