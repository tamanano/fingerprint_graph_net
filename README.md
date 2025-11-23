# fingerprint_graph_net

![arXiv](https://img.shields.io/badge/arXiv-2409.08782-b31b1b)

**This repo is the official implementation of:**

IJCB 2025: [Improving Contactless Fingerprint Recognition with Robust 3D Feature Extraction and Graph Embedding](https://arxiv.org/abs/2409.08782)

## Notice

Now we release our Graph Embedding Network and the 3D minutiae on UWA, the Feature Extraction Network and other  data able to be released will be released soon.

## Data

Download UWA data 

```latex
链接: https://pan.baidu.com/s/1rXaXvzPTax856wkNAcWpeQ 提取码: a5kj 
```

Unzip and use `generate_pairs_identify_uwa()`  in `data_process.py`  to get the testing txt. Need to change the path in this function.

### Testing

```latex
python test_uwa.py
```

Then use `/mnt/sdb/jyw_data/dgcnn/test_uwa_det.mat` to compute EER or draw DET curves.