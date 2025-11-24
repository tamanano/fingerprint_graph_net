# fingerprint_graph_net

[![arXiv](https://img.shields.io/badge/arXiv-2409.08782-b31b1b)](https://arxiv.org/abs/2409.08782)

**This repo is the official implementation of:**

IJCB 2025: [Improving Contactless Fingerprint Recognition with Robust 3D Feature Extraction and Graph Embedding](https://arxiv.org/abs/2409.08782)

## Notice

Now we release our 3D Feature Extraction Network testing code, our Graph Embedding Network and the 3D minutiae on UWA, you can download the datasets and preprocess the to 640*480 and run our codes.

## Data and model

Download UWA data and trained model.

```latex
链接: https://pan.baidu.com/s/19BNxtOqtctkO-G3C69tqZQ 提取码: hvf5
```
or
```latex
https://drive.google.com/drive/folders/1odq_QQZSbPau45RkM2zogw5K57sNopHq?usp=drive_link
```
Unzip and use `generate_pairs_identify_uwa()`  in `data_process.py`  to get the testing txt. Need to change the path in this function.

### Testing

```latex
python test_uwa.py
```

Then use `test_uwa_det.mat` to compute EER or draw DET curves.
