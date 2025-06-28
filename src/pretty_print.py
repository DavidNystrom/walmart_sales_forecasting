# src/pretty_print.py

import argparse
import pandas as pd
from rich.console import Console
from rich.table import Table

def pretty_print_csv(path: str, max_rows: int = None):
    """
    Load a CSV at `path` into a pandas DataFrame and print it as a rich table.
    If max_rows is specified, only the first max_rows are shown.
    """
    df = pd.read_csv(path)
    if max_rows:
        df = df.head(max_rows)

    console = Console()
    table = Table(show_header=True, header_style="bold cyan")
    # Add columns
    for col in df.columns:
        table.add_column(col)

    # Add rows
    for row in df.itertuples(index=False, name=None):
        table.add_row(*[str(cell) for cell in row])

    console.print(table)

def main():
    parser = argparse.ArgumentParser(
        description="Pretty-print any CSV file as a formatted table."
    )
    parser.add_argument(
        "file",
        help="Path to the CSV file to print."
    )
    parser.add_argument(
        "--max-rows", "-n",
        type=int,
        default=None,
        help="Maximum number of rows to display. Defaults to all rows."
    )
    args = parser.parse_args()
    pretty_print_csv(args.file, args.max_rows)

if __name__ == "__main__":
    main()
