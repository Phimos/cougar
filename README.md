# cougar

A python C-extension for rolling window aggregations. Try to support more methods than `bottleneck` and run faster than `pandas`. Currently this is only a weekend project, feel free to contribute.


## Installation

```bash
pip install cougar
```

## Usage

```python
import numpy as np
import cougar as cg

arr = np.random.rand(1_000_000)
cg.rolling_mean(arr, 1_000)
```

## Supported Methods

| Method   | Time Complexity |
| -------- | --------------- |
| sum      | O(1)            |
| mean     | O(1)            |
| min      | O(1)            |
| max      | O(1)            |
| argmin   | O(1)            |
| argmax   | O(1)            |
| rank     | O(log n)        |
| quantile | O(log n)        |

## Benchmark

todo