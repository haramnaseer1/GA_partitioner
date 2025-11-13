#!/usr/bin/env python3
from __future__ import annotations
import os, ast, json, argparse, sys

PREFIXES = ("updated_schedule : ", "updated_schedule:")


def find_last_updated_schedule_line(log_path: str) -> str | None:
    """Read file once and return the last line containing the prefix."""
    if not os.path.exists(log_path):
        print(f"[make-solution] Log not found: {log_path}", file=sys.stderr)
        return None
    last = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if any(p in line for p in PREFIXES):
                last = line.rstrip("\n")
    return last


def extract_payload(line: str) -> str:
    """Get the dictionary text from an updated_schedule log line."""
    for p in PREFIXES:
        if p in line:
            payload = line.split(p, 1)[1].strip()
            break
    else:
        raise ValueError("No known prefix found in line")
    # remove quotes if it's a quoted string
    if (payload.startswith('"') and payload.endswith('"')) or (
        payload.startswith("'") and payload.endswith("'")
    ):
        payload = payload[1:-1]
    return payload


def parse_schedule_dict(text: str) -> dict:
    """Safely parse the literal dictionary string into a Python dict."""
    return ast.literal_eval(text)


def to_rows(schedule: dict) -> list[dict]:
    """Flatten GA-style nested schedule dict into a list of simple JSON objects."""
    rows = []
    for _, (tasks, partition_mk) in schedule.items():
        for task_id, entry in tasks.items():
            if len(entry) < 4:
                continue
            node_id, start, end, deps = entry
            rows.append(
                {
                    "node_id": int(node_id),
                    "task_id": int(task_id),
                    "start_time": float(start),
                    "end_time": float(end),
                    "dependencies": [
                        {
                            "task_id": int(d[0]),
                            "path_id": str(d[1]),
                            "message_size": float(d[2]),
                        }
                        for d in deps
                    ],
                }
            )
    rows.sort(key=lambda r: (r["node_id"], r["start_time"], r["task_id"]))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Create solution/<input_basename>_ga.json from the last updated_schedule in GA log."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input application model (e.g., Application/T2.json)",
    )
    parser.add_argument(
        "--log",
        default=os.path.join("Logs", "global_ga.log"),
        help="Path to GA log file (default: Logs/global_ga.log)",
    )
    parser.add_argument(
        "--outdir",
        default="solution",
        help="Folder for the generated _ga.json file (default: solution)",
    )
    args = parser.parse_args()

    # Resolve paths
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(repo_root, args.input)
    log_path = os.path.join(repo_root, args.log)
    outdir = os.path.join(repo_root, args.outdir)

    # Validate input file exists
    if not os.path.exists(input_path):
        print(f"[make-solution] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Build output name
    base = os.path.basename(input_path)
    name, ext = os.path.splitext(base)
    if not ext:
        ext = ".json"
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{name}_ga{ext}")

    # Find last updated_schedule
    line = find_last_updated_schedule_line(log_path)
    if not line:
        print(f"[make-solution] No 'updated_schedule' found in {log_path}", file=sys.stderr)
        sys.exit(2)

    # Parse the line
    try:
        payload = extract_payload(line)
        schedule = parse_schedule_dict(payload)
    except Exception as e:
        print(f"[make-solution] Parse error: {e}", file=sys.stderr)
        tail = line[-200:] if len(line) > 200 else line
        print(f"[make-solution] Offending line tail: {tail}", file=sys.stderr)
        sys.exit(3)

    # Flatten and save
    rows = to_rows(schedule)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"[make-solution] ✅ Wrote {out_path}  (rows={len(rows)})")


if __name__ == "__main__":
    main()
