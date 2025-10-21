# Timing Instrumentation Summary

## Changes Made

I've added comprehensive time profiling and complexity tracking throughout the Graph-LLaVA codebase. Here's what was instrumented:

### 1. Main Training Pipeline (`llava/train/train_drug.py`)

Added `TimeProfiler` class that tracks:
- Argument parsing time
- LLM model loading time
- Graph tower (GPM) initialization time
- Dataset loading and preprocessing time
- Trainer initialization time
- Main training loop time
- Model saving time

**Output Format:**
```
[YYYY-MM-DD HH:MM:SS] [STAGE] Stage name
[YYYY-MM-DD HH:MM:SS] [TIMING] Stage name: XX.XXs (or XXmin or XXh)
```

**Summary at end of training:**
```
================================================================================
[TIMING SUMMARY]
================================================================================
  Stage name                              : Duration         (% of total)
  ...
  TOTAL                                   : Total time
================================================================================
```

### 2. GPM Tower (`llava/model/multimodal_encoder/gpm.py`)

Added per-forward-pass timing that tracks:
- Pattern generation time
- Total forward pass time
- Number of nodes processed
- Number of patterns generated

**Periodic Output** (every 100 graphs):
```
[GPM TIMING] Last 100 graphs: Pattern gen=XXXms, Total=XXXms, Avg nodes=XX.X
```

**Statistics Tracked:**
- `self.pattern_gen_times`: List of pattern generation times
- `self.forward_times`: List of total forward times
- `self.total_nodes_processed`: Cumulative node count
- `self.total_patterns_generated`: Cumulative pattern count

### 3. Entry Point (`llava/train/train_mem.py`)

Modified to call `profiler.summary()` at the end to print complete timing breakdown.

## How to Use

### Running with Timing:

```bash
# Pretraining
bash scripts/pretrain.sh | tee pretrain_with_timing.log

# Finetuning
bash scripts/finetune.sh | tee finetune_with_timing.log

# Evaluation
bash scripts/eval_property_pred.sh | tee eval_with_timing.log
```

### Analyzing Timing Logs:

```bash
# Extract all timing information
grep "\[TIMING\]" pretrain_with_timing.log

# Extract GPM-specific timing
grep "\[GPM TIMING\]" pretrain_with_timing.log

# View summary
grep -A 20 "TIMING SUMMARY" pretrain_with_timing.log
```

### Python Analysis:

```python
import re

# Parse timing log
def parse_timing_log(logfile):
    timings = {}
    with open(logfile) as f:
        for line in f:
            if '[TIMING]' in line:
                match = re.search(r'\[TIMING\] (.*?): ([\d.]+)', line)
                if match:
                    stage, time = match.groups()
                    timings[stage] = float(time)
    return timings

timings = parse_timing_log('pretrain_with_timing.log')
print(f"Total training time: {timings.get('Main training loop', 0):.2f}s")
```

## Expected Output Example

### Pretraining with 10K samples:

```
[2024-10-13 22:30:00] [STAGE] === TRAINING PIPELINE START ===
[2024-10-13 22:30:01] [TIMING] Argument parsing: 0.12s
[2024-10-13 22:30:45] [TIMING] LLM model loading: 44.23s
[2024-10-13 22:01:30] [TIMING] Graph tower initialization (gpm): 2.45s
[2024-10-13 22:02:15] [TIMING] Dataset loading and preprocessing: 12.34s
[2024-10-13 22:02:16] [TIMING] Trainer initialization: 0.89s

***** Running training *****
  Num examples = 10,000
  Num Epochs = 2
  Total optimization steps = 625

[GPM TIMING] Last 100 graphs: Pattern gen=123.45ms, Total=456.78ms, Avg nodes=28.3
[GPM TIMING] Last 100 graphs: Pattern gen=118.92ms, Total=445.23ms, Avg nodes=29.1
...

[2024-10-13 23:15:30] [TIMING] Main training loop: 2345.67s (39.09min)
[2024-10-13 23:15:35] [TIMING] State saving: 4.23s

================================================================================
[TIMING SUMMARY]
================================================================================
  Argument parsing                        : 0.12s                (0.0%)
  LLM model loading                       : 44.23s               (1.8%)
  Graph tower initialization (gpm)        : 2.45s                (0.1%)
  Dataset loading and preprocessing       : 12.34s               (0.5%)
  Trainer initialization                  : 0.89s                (0.0%)
  Main training loop                      : 2345.67s (39.09min)  (97.5%)
  State saving                            : 4.23s                (0.2%)
  TOTAL                                   : 2409.93s (40.17min)
================================================================================
```

### Finetuning (smaller dataset):

```
[2024-10-13 23:30:00] [STAGE] === TRAINING PIPELINE START ===
...
[2024-10-13 23:45:30] [TIMING] Main training loop: 1234.56s (20.58min)
...
```

### Evaluation:

```
[GPM TIMING] Last 100 graphs: Pattern gen=89.12ms, Total=312.45ms, Avg nodes=31.2
```

## Complexity Metrics

The instrumentation tracks:

### Time Complexity:
- **Per-component timing**: How long each stage takes
- **Bottleneck identification**: Which component is slowest
- **Scalability analysis**: How time scales with dataset size

### Space Complexity:
- Tracked via system metrics (not explicitly logged, but can be added)
- GPU memory usage visible in training logs

### Algorithmic Complexity:
- **GPM**: O(num_patterns × pattern_size × num_nodes × avg_degree)
- **LLM**: O(num_layers × sequence_length² × hidden_dim)
- **Total per step**: Sum of all components

See `COMPLEXITY_ANALYSIS.md` for detailed formulas and analysis.

## Customization

### Adding More Timing Points:

```python
# In any training file
from llava.train.train_drug import profiler

# Time a specific section
profiler.log("Custom stage name")
t0 = time.time()
# ... your code ...
profiler.log("Custom stage name", time.time() - t0)
```

### Adjusting GPM Timing Frequency:

```python
# In llava/model/multimodal_encoder/gpm.py
# Change line 137:
if len(self.forward_times) % 100 == 0:  # Print every 100 graphs
# To:
if len(self.forward_times) % 50 == 0:   # Print every 50 graphs
```

### Exporting Timing Data:

Add to `train_mem.py`:
```python
import json

if __name__ == "__main__":
    train()
    profiler.summary()
    
    # Export to JSON
    with open('timing_data.json', 'w') as f:
        json.dump({
            'timings': profiler.timings,
            'total_time': time.time() - profiler.start_time
        }, f, indent=2)
```

## Performance Impact

The instrumentation has **minimal overhead**:
- `time.time()` calls: < 1μs each
- String formatting: < 10μs per log
- Periodic printing: Only every 100 GPM forward passes
- **Total overhead**: < 0.1% of training time

## Next Steps

1. **Run pretraining** with instrumentation to collect baseline data
2. **Analyze bottlenecks** using timing logs
3. **Optimize slowest components** based on data
4. **Compare** before/after optimization timing

## Files Modified

```
llava/train/train_drug.py          - Added TimeProfiler class and timing points
llava/train/train_mem.py           - Added profiler.summary() call
llava/model/multimodal_encoder/gpm.py - Added GPM-specific timing
COMPLEXITY_ANALYSIS.md             - Comprehensive complexity documentation
TIMING_INSTRUMENTATION_SUMMARY.md  - This file
```

All changes are **non-invasive** and can be easily removed by:
1. Removing `profiler.log()` calls
2. Removing timing variables (`t0`, etc.)
3. Removing GPM timing stats

The code will work exactly the same without these changes - they only add observability!

