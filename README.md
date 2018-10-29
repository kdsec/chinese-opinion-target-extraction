# Chinese Opinion Target Extraction
Pytorch implement of "Character-based BiLSTM-CRF Incorporating POS and Dictionaries for Chinese Opinion Target Extraction"

### Dependency

Python version >= 3.6

```
torch >= 0.4.0
thulac
tqdm
keras
numpy
numba
```

### Usage

1. Install dependency
2. Download dataset from [this repo](https://github.com/lsvih/chinese-customer-review), move files into `./dataset` folder, then unzip `dictionary.zip`.
3. Train model: `python3 main.py --type=train --dataset=baidu`
4. Test model: `python3 main.py --type=test --dataset=baidu`

> Note: It would cost about 10~20 minutes for pre-processing.

### Results

|   | Baidu | Mafengwo | Dianping |
| --- | --- | --- | --- |
| P | 85.791 | 83.273 | 83.753 |
| R | 82.531 | 89.989 | 85.672 |
| F1 | 84.130 | 86.501 | 84.702 |
