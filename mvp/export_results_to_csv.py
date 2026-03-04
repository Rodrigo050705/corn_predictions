import argparse
import csv
import json
import sqlite3
from pathlib import Path


def export_sqlite_to_csv(
    db_path: Path,
    output_csv: Path,
    flatten_probs: bool = False,
    table: str = "inferences",
):
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    cur = conn.execute(f"SELECT * FROM {table};")
    rows = cur.fetchall()

    if not rows:
        print("No rows found.")
        return

    # Get base columns
    base_columns = rows[0].keys()

    # If flattening probability JSON into columns
    prob_keys = set()
    if flatten_probs:
        for row in rows:
            if row["probs_json"]:
                try:
                    probs = json.loads(row["probs_json"])
                    prob_keys.update(probs.keys())
                except Exception:
                    pass
        prob_keys = sorted(prob_keys)

    # Final header
    header = list(base_columns)
    if flatten_probs:
        header += [f"prob_{k}" for k in prob_keys]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for row in rows:
            row_dict = dict(row)

            if flatten_probs:
                try:
                    probs = json.loads(row_dict.get("probs_json", "{}"))
                except Exception:
                    probs = {}

                for k in prob_keys:
                    row_dict[f"prob_{k}"] = probs.get(k)

            writer.writerow(row_dict)

    conn.close()
    print(f"Exported {len(rows)} rows to {output_csv.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="results.sqlite")
    parser.add_argument("--output_csv", type=str, default="results.csv")
    parser.add_argument(
        "--flatten_probs",
        action="store_true",
        help="Expand probs_json into separate columns",
    )
    args = parser.parse_args()

    export_sqlite_to_csv(
        db_path=Path(args.db_path),
        output_csv=Path(args.output_csv),
        flatten_probs=args.flatten_probs,
    )


if __name__ == "__main__":
    main()