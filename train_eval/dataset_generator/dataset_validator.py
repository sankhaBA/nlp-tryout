import argparse
import csv
from collections import Counter
from pathlib import Path


def read_rows(csv_path: Path) -> list[tuple[str, str]]:
	rows: list[tuple[str, str]] = []
	with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
		reader = csv.DictReader(csv_file)
		required_columns = {"input", "target"}
		if not required_columns.issubset(reader.fieldnames or set()):
			raise ValueError(
				f"CSV must contain columns {sorted(required_columns)}, "
				f"found: {reader.fieldnames}"
			)

		for row in reader:
			rows.append((row["input"].strip(), row["target"].strip()))

	return rows


def summarize_duplicates(rows: list[tuple[str, str]]) -> tuple[int, int, dict[tuple[str, str], int]]:
	counts = Counter(rows)
	duplicated_items = {item: count for item, count in counts.items() if count > 1}
	duplicate_lines = len(duplicated_items)
	duplicate_instances = sum(count - 1 for count in duplicated_items.values())
	return duplicate_lines, duplicate_instances, duplicated_items


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Find duplicates in indoor navigation dataset and show duplication count per line."
	)
	parser.add_argument(
		"--file",
		type=Path,
		default=Path(__file__).with_name("indoor_navigation_dataset.csv"),
		help="Path to CSV file (default: indoor_navigation_dataset.csv next to this script)",
	)
	args = parser.parse_args()

	csv_path = args.file.resolve()
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	rows = read_rows(csv_path)
	duplicate_lines, duplicate_instances, duplicated_items = summarize_duplicates(rows)

	print(f"File: {csv_path}")
	print(f"Total data rows: {len(rows)}")
	print(f"Unique rows: {len(set(rows))}")
	print(f"Rows with duplicates: {duplicate_lines}")
	print(f"Total duplicate instances: {duplicate_instances}")

	if not duplicated_items:
		print("\nNo duplicates found.")
		return

	print("\nDuplicate rows and counts:")
	for index, ((input_text, target_text), count) in enumerate(
		sorted(duplicated_items.items(), key=lambda item: (-item[1], item[0][0], item[0][1])), start=1
	):
		print(f"{index}. count={count}")
		print(f"   input : {input_text}")
		print(f"   target: {target_text}")


if __name__ == "__main__":
	main()
