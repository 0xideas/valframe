# ValFrame: Schema-Validated DataFrames for Robust Data Pipelines

ValFrame is a Python library for creating self-validating DataFrame types using Pandera schemas, supporting both in-memory and out-of-core (folder-based) data.

The core motivation is to leverage Python's type system to guarantee data validity at runtime. By creating specific, validated `ValFrame` types, you can write functions that are guaranteed (by using the @beartype decorator) to receive data with the correct shape and characteristics, preventing downstream errors and making your data pipelines more robust and reliable.

---

## Quick Start

Install ValFrame from PyPI:

```bash
pip install valframe
```

Define a schema and create a validated DataFrame type:

```python
import pandas as pd
import pandera.pandas as pa
from valframe import create_valframe_type

# Define a schema for your data
UserSchema = pa.DataFrameSchema({
    "user_id": pa.Column(int, pa.Check.ge(0)),
    "name": pa.Column(str)
})

# Create a validated DataFrame type
UserDataFrame = create_valframe_type("UserDataFrame", UserSchema, library="pandas")

# This succeeds
valid_df = UserDataFrame(pd.DataFrame({"user_id": [1, 2], "name": ["Alice", "Bob"]}))

# This will raise a pandera.errors.SchemaError
invalid_df = UserDataFrame(pd.DataFrame({"user_id": [-1, 0], "name": ["Carl", "Eve"]}))
```

---

## Features

* **Schema-First Validation**: Build DataFrame types directly from Pandera schemas.
* **In-Memory Validation**: Create DataFrame objects that validate their contents upon instantiation.
* **Folder-Based Virtual Frames**: Treat a directory of data files as a single, indexable DataFrame without loading the entire dataset into memory.
* **Pandas & Polars Support**: Works seamlessly with both major DataFrame libraries.
* **Lazy Validation**: Defer validation on folder-based frames until data is accessed for faster initialization.
* **Type System Integration**: Designed to work with type checkers like `beartype` to provide strong runtime guarantees about data contracts.

---

## Supported Formats

ValFrame's folder-based mode supports reading from the following file formats:
* `csv`
* `parquet`

---

## Relative Positioning

ValFrame occupies a unique niche by providing a balance of high data integrity and moderate processing efficiency.

* Unlike `pydantic-pandas`, it uses vectorized validation via **Pandera**, making it significantly more performant on large datasets, especially with Polars.
* Compared to high-scale tools like **Polars (lazy mode)** or **Dask**, ValFrame's integrity guarantee is inherent and automatic, whereas in lazy frameworks, validation is a manual step that must be explicitly added to the computation graph.
* While orchestration frameworks like **Dagster** provide pipeline-level integrity, ValFrame offers a lightweight, low-complexity solution perfect for "medium data" problems—datasets too large for memory but too simple to require a full data engineering framework.

---

## Installation

Install the package directly from PyPI:
```bash
pip install valframe
```

### Dependencies
* Python 3.10+
* `pandera[polars]`
* `pandas`
* `polars`
* `beartype`
* `numpy`

---

## In-Depth Example: Data Integrity with `beartype`

This example demonstrates how to combine `valframe` and `beartype` to create a function that is guaranteed to receive valid data, preventing runtime errors.

```python
import pandas as pd
import pandera.pandas as pa
from beartype import beartype
from valframe import create_valframe_type

# 1. Define a strict schema for transaction data
TransactionSchema = pa.DataFrameSchema(
    {
        "transaction_id": pa.Column(str, pa.Check.str_startswith("txn_")),
        "amount_usd": pa.Column(float, pa.Check.gt(0)),
        "seller_id": pa.Column(int, pa.Check.ge(1000)),
    },
    strict=True,  # Disallow any columns not defined in the schema
    ordered=True, # Enforce column order
)

# 2. Create a specific, validated DataFrame type for this schema
TransactionDataFrame = create_valframe_type(
    "TransactionDataFrame", TransactionSchema, library="pandas"
)

# 3. Use @beartype to enforce that our function ONLY accepts this type
@beartype
def process_payouts(transactions: TransactionDataFrame) -> float:
    """
    Calculates the total payout amount from a validated DataFrame of transactions.

    Because of the @beartype decorator and the TransactionDataFrame type,
    we are 100% certain that the `transactions` argument is a pandas DataFrame
    and that its contents conform to the TransactionSchema.
    """
    print("Payout processing started on valid data...")
    total_payout = transactions["amount_usd"].sum()
    return total_payout

# --- Main execution ---
if __name__ == "__main__":
    # a) Create a valid DataFrame
    valid_data = pd.DataFrame({
        "transaction_id": ["txn_123", "txn_456"],
        "amount_usd": [150.50, 75.00],
        "seller_id": [1001, 1024],
    })

    # Instantiate our validated type. This succeeds.
    validated_transactions = TransactionDataFrame(valid_data)
    total = process_payouts(validated_transactions)
    print(f"Total payout is: ${total:.2f}") # Output: Total payout is: $225.50

    print("-" * 20)

    # b) Create an invalid DataFrame
    invalid_data = pd.DataFrame({
        "transaction_id": ["txn_789", "inv_000"], # "inv_000" is invalid
        "amount_usd": [99.99, 50.00],
        "seller_id": [1050, 999], # 999 is invalid
    })

    try:
        # This line will fail immediately upon instantiation,
        # preventing the invalid data from ever reaching our function.
        invalid_transactions = TransactionDataFrame(invalid_data)
        process_payouts(invalid_transactions)
    except pa.errors.SchemaError as e:
        print("Failed to create DataFrame due to validation errors:")
        print(e)
```

## License

Apache 2.0
