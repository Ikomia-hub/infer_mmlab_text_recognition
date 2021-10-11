import numpy as np
textrecog_models = {
            'CRNN': {
                'config': 'crnn/crnn_academic_dataset.py',
                'ckpt': 'crnn/crnn_academic-a723a1c5.pth'
            },
            'SAR': {
                'config': 'sar/sar_r31_parallel_decoder_academic.py',
                'ckpt': 'sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth'
            },
            'NRTR_1/16-1/8': {
                'config': 'nrtr/nrtr_r31_1by16_1by8_academic.py',
                'ckpt': 'nrtr/nrtr_r31_academic_20210406-954db95e.pth'
            },
            'NRTR_1/8-1/4': {
                'config': 'nrtr/nrtr_r31_1by8_1by4_academic.py',
                'ckpt':
                'nrtr/nrtr_r31_1by8_1by4_academic_20210406-ce16e7cc.pth'
            },
            'RobustScanner': {
                'config': 'robust_scanner/robustscanner_r31_academic.py',
                'ckpt': 'robustscanner/robustscanner_r31_academic-5f05874f.pth'
            },
            'SATRN': {
                'config': 'satrn/satrn_academic.py',
                'ckpt': 'satrn/satrn_academic_20210809-59c8c92d.pth'
            },
            'SATRN_sm': {
                'config': 'satrn/satrn_small.py',
                'ckpt': 'satrn/satrn_small_20210811-2badf6fc.pth'
            },
            'SEG': {
                'config': 'seg/seg_r31_1by16_fpnocr_academic.py',
                'ckpt': 'seg/seg_r31_1by16_fpnocr_academic-72235b11.pth'
            },
            'CRNN_TPS': {
                'config': 'tps/crnn_tps_academic_dataset.py',
                'ckpt': 'tps/crnn_tps_academic_dataset_20210510-d221a905.pth'
            }
        }
def polygon2bbox(pts):
    x = np.min(pts[:,0])
    y = np.min(pts[:,1])
    w = np.max(pts[:,0]) - x
    h = np.max(pts[:,1]) - y
    return [int(x),int(y),int(w),int(h)]

def bbox2polygon(box):
    x,y,w,h = box
    # starting with left bottom point then anti clockwise
    return [x,y+h,x+w,y+h,x+w,y,x,y]