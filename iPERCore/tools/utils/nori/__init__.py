import cv2
import os
import nori2 as nori

def build_single_video_nori(frames_path, info_dict, nori_path):
    nori_list = []
    with nori.open(nori_path, "w") as nw:
        for fn in frames_path:
            img = cv2.imread(fn)
            data_id = nw.put(cv2.imencode(".png", img)[1].tostring(), filename=os.path.basename(fn))
            nori_list.append(data_id)
    info_dict['nori_seq'] = nori_list
    return info_dict

