import timeit
from typing import Union

import bottleneck as bn
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

import cougar as cg

methods = ["sum", "mean", "std", "max", "min", "argmax", "argmin", "median", "rank"]


def run_benchmark(shape: Union[int, tuple] = 1_000_000, dtype: str = "float64"):
    array = np.random.random(shape).astype(dtype)

    table = Table(title="Cougar vs. Bottleneck", show_header=True, header_style="bold")
    table.add_column("Method", width=12)
    table.add_column("Cougar", width=12)
    table.add_column("Bottleneck", width=12)

    for method in methods:
        results = {}

        nloops, time = timeit.Timer(
            f"cg.rolling_{method}(array, 1000)",
            globals={"array": array, "cg": cg},
        ).autorange()
        results["cougar"] = time / nloops

        nloops, time = timeit.Timer(
            f"bn.move_{method}(array, 1000)",
            globals={"array": array, "bn": bn},
        ).autorange()
        results["bottleneck"] = time / nloops

        fastest = min(results, key=results.get)
        results = {k: v / results[fastest] for k, v in results.items()}
        results = {
            k: f"{v:.2f}" if k != fastest else f"[bold green]{v:.2f}[/bold green]"
            for k, v in results.items()
        }

        table.add_row(
            method,
            results["cougar"],
            results["bottleneck"],
        )

    console = Console()
    console.print(table)


run_benchmark()
