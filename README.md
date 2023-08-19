# cougar

A python C-extension for rolling window aggregations. Try to support more methods than `bottleneck` and run faster than `pandas`. Currently this is only a weekend project, feel free to contribute.


## Installation

```bash
pip install cougar
```

## Usage

```python
>>> import numpy as np
>>> import cougar as cg


>>> arr = np.random.randn(8)
>>> arr
array([-2.31572505, -1.78462521,  0.17355123,  0.77365821,  0.81431295,
        1.56188616,  0.74933881,  0.06184727])
>>> cg.rolling_mean(arr, 3)
array([        nan,         nan, -1.30893301, -0.27913859,  0.58717413,
        1.04995244,  1.04184597,  0.79102408])
>>> cg.rolling_rank(arr, 5, min_count=1)
array([ 0. ,  1. ,  1. ,  1. ,  1. ,  1. , -0.5, -1. ])
```

## Supported Methods

| Method   | Time Complexity |
| -------- | --------------- |
| sum      | $O(1)$          |
| mean     | $O(1)$          |
| std      | $O(1)$          |
| var      | $O(1)$          |
| skew     | $O(1)$          |
| kurt     | $O(1)$          |
| min      | $O(1)$          |
| max      | $O(1)$          |
| argmin   | $O(1)$          |
| argmax   | $O(1)$          |
| median   | $O(\log n)$     |
| rank     | $O(\log n)$     |
| quantile | $O(\log n)$     |

## Benchmark

```
python benchmark.py
```

```
            Cougar vs. Bottleneck             
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Method       ┃ Cougar       ┃ Bottleneck   ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ sum          │ 2.99         │ 1.00         │
│ mean         │ 3.08         │ 1.00         │
│ std          │ 2.10         │ 1.00         │
│ var          │ 2.15         │ 1.00         │
│ max          │ 2.84         │ 1.00         │
│ min          │ 1.23         │ 1.00         │
│ argmax       │ 1.40         │ 1.00         │
│ argmin       │ 1.16         │ 1.00         │
│ median       │ 1.33         │ 1.00         │
│ rank         │ 1.00         │ 15.02        │
└──────────────┴──────────────┴──────────────┘
```