import re
import numpy as np


def parse_metrics(text):
    result = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        f1_pos1 = line.index('f1=')
        f1_pos2 = line.index('f1=', f1_pos1 + 1)
        f1_value = float(line[f1_pos2 + 3:])
        result.append(f1_value)

    test_f1_10 = np.mean(result)

    last_line = text.split("\n")[-1]
    if not last_line.strip():
        last_line = text.split("\n")[-2]
    print(last_line)
    m = re.search(r"Train: accuracy=(\d+\.\d+),.* f1=(\d+\.\d+),.*"
                  r"Test: accuracy=(\d+\.\d+),.* f1=(\d+\.\d+)",
                  last_line)
    train_accuracy, train_f1, test_accuracy, test_f1 = map(
        float,
        [m.group(i) for i in range(1, 5)])

    metric_fmt = (f"| {test_f1:.3f} | {test_f1_10:.3f} | {test_accuracy:.3f} "
                  f"| {train_f1:.3f} | {train_accuracy:.3f} |")
    return metric_fmt
