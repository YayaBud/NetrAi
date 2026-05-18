import os
import csv
import sys
import atexit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def strip_compile_prefix(state_dict):
    """Remove '_orig_mod.' prefix from torch.compile()'d state dicts."""
    PREFIX = "_orig_mod."
    if any(k.startswith(PREFIX) for k in state_dict.keys()):
        return {(k[len(PREFIX):] if k.startswith(PREFIX) else k): v
                for k, v in state_dict.items()}
    return state_dict


def repair_csv_header(csv_path, expected_header):
    """Rewrite a CSV header if an older schema is already on disk."""
    if not os.path.exists(csv_path):
        return

    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
    except Exception as e:
        print(f"  CSV header check skipped for {os.path.basename(csv_path)}: {e}")
        return

    if not rows:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(expected_header)
        return

    current_header = rows[0]
    if current_header == expected_header:
        return

    print(f"  Repairing {os.path.basename(csv_path)} header -> current schema")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(expected_header)
        writer.writerows(rows[1:])


def append_csv_row(csv_path, row, label):
    """Append a CSV row without letting a disk/write failure stop training."""
    try:
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        return True
    except Exception as e:
        print(f"  WARNING: failed to write {label} to {os.path.basename(csv_path)}: {e}")
        return False


def load_loss_history(csv_path):
    """Load train/val/lcw history from an old or new loss.csv schema."""
    rows = []
    if not os.path.exists(csv_path):
        return rows

    try:
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for raw_row in reader:
                if not raw_row:
                    continue
                try:
                    epoch = int(raw_row[0])
                    train_loss = float(raw_row[1]) if len(raw_row) > 1 and raw_row[1] != "" else None
                    val_loss = float(raw_row[2]) if len(raw_row) > 2 and raw_row[2] != "" else None
                    snr = float(raw_row[3]) if len(raw_row) > 3 and raw_row[3] != "" else None
                    ms = float(raw_row[4]) if len(raw_row) > 4 and raw_row[4] != "" else None
                    val_snr = float(raw_row[5]) if len(raw_row) > 5 and raw_row[5] != "" else None
                    val_ms = float(raw_row[6]) if len(raw_row) > 6 and raw_row[6] != "" else None
                    lr = float(raw_row[7]) if len(raw_row) > 7 and raw_row[7] != "" else None
                    lcw = float(raw_row[8]) if len(raw_row) > 8 and raw_row[8] != "" else None
                    seg_weight = float(raw_row[9]) if len(raw_row) > 9 and raw_row[9] != "" else None
                except Exception:
                    continue

                rows.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "snr": snr,
                    "ms": ms,
                    "val_snr": val_snr,
                    "val_ms": val_ms,
                    "lr": lr,
                    "lcw": lcw,
                    "seg_weight": seg_weight,
                })
    except Exception as e:
        print(f"  WARNING: could not load {os.path.basename(csv_path)} for LCW history: {e}")

    return rows


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def setup_terminal_logging(log_path):
    try:
        log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    except Exception as e:
        print(f"  WARNING: could not open terminal log {log_path}: {e}")
        return None

    sys.stdout = _TeeStream(sys.stdout, log_file)
    sys.stderr = _TeeStream(sys.stderr, log_file)
    atexit.register(log_file.close)
    print(f"  Terminal log enabled: {log_path}")
    return log_file


def save_lcw_curve(checkpoint_dir, lcw_x, lcw_y):
    if not lcw_x or not lcw_y:
        return

    plt.figure(figsize=(10, 4))
    plt.plot(lcw_x, lcw_y, color="tab:orange", linewidth=1.4, label="LCW")
    plt.scatter(lcw_x[::max(len(lcw_x)//200, 1)], lcw_y[::max(len(lcw_y)//200, 1)],
                s=8, color="tab:red", alpha=0.35)
    plt.xlabel("Epoch progress")
    plt.ylabel("LCW")
    plt.title("LCW live curve")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "lcw_curve.png"), dpi=96, bbox_inches="tight")
    plt.close()
