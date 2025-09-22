import os
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pandera.polars as pal
import polars as pl
from beartype import beartype

SUPPORTED_FILE_FORMATS = ["csv", "parquet"]
SUPPORTED_LIBRARIES = ["polars", "pandas"]

FILE_FORMATS_TO_READ_METHOD = {
    "polars": {"csv": pl.read_csv, "parquet": pl.read_parquet},
    "pandas": {"csv": pd.read_csv, "parquet": pd.read_parquet},
}


@beartype
def create_valframe_type(
    name: str,
    schema: Union[pa.DataFrameSchema, pal.DataFrameSchema],
    library: str = "polars",
    folder: bool = False,
    nested_level: int = 0,
    input_file_formats: Optional[list[str]] = None,
    read_kwargs: Optional[dict[str, Any]] = None,
    max_errors: Optional[int] = 10,
    lazy_validation: bool = False,
):
    assert (
        library in SUPPORTED_LIBRARIES
    ), f"the supported libraries are {SUPPORTED_LIBRARIES}"
    assert (
        folder or nested_level == 0
    ), "for a file ValFrame, nested_level needs to be set to 0"
    assert (
        folder or input_file_formats is None
    ), "for a file ValFrame, input_file_formats needs to be None"
    assert folder == (
        input_file_formats is not None
    ), "for a folder ValFrame, input_file_formats cannot be None"
    assert input_file_formats is None or np.all(
        [format == format.lower() for format in input_file_formats]
    ), "input_file_formats must be all lower case"
    assert input_file_formats is None or np.all(
        [format in SUPPORTED_FILE_FORMATS for format in input_file_formats]
    ), f"only these file formats are supported: {SUPPORTED_FILE_FORMATS}"

    assert (
        folder or read_kwargs is None
    ), "for a file ValFrame, read_kwargs needs to be None"

    if folder:

        def __init__(self, path: str):  # type: ignore
            assert path.startswith("..") is False

            self.path = path
            self.schema = schema
            self.library = library
            self.nested_level = nested_level
            self.input_file_formats = input_file_formats
            self.read_kwargs = read_kwargs
            self.lazy_validation = lazy_validation

            self.invalid_file_paths, self.error_messages = [], []
            self.file_path_to_shape = {}
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_format = file.split(".")[-1].lower()
                    if (
                        self.input_file_formats
                        and file_format in self.input_file_formats
                    ):
                        file_path = os.path.join(root, file)
                        file_nested_level = (
                            len(os.path.relpath(file_path, root).split(os.sep)) - 1
                        )
                        if file_nested_level == nested_level:
                            try:
                                data = FILE_FORMATS_TO_READ_METHOD[library][
                                    file_format
                                ](
                                    file_path,
                                    **(read_kwargs if read_kwargs is not None else {}),
                                )
                            except Exception as e:
                                self.invalid_file_paths.append(file_path)
                                error_message = (
                                    f"reading data failed for {file_path}: {e}"
                                )
                                self.error_messages.append(error_message)
                                continue  # Skip to the next file

                            # Now, handle validation based on the lazy_validation flag
                            if self.lazy_validation:
                                # In lazy mode, we assume the file is valid for now and just store its shape.
                                # Validation will happen on access in __getitem__.
                                self.file_path_to_shape[file_path] = data.shape
                            else:
                                # In eager mode, validate immediately.
                                try:
                                    schema.validate(data)
                                    self.file_path_to_shape[file_path] = data.shape
                                except Exception as e:
                                    self.invalid_file_paths.append(file_path)
                                    error_message = f"valframe schema validation failed for {file_path}: {e}"
                                    self.error_messages.append(error_message)

                    if (
                        max_errors is not None
                        and len(self.error_messages) >= max_errors
                    ):
                        raise Exception("\n".join(self.error_messages))

            cumulative, row_index_file_tuples = 0, []
            for file_path, shape in self.file_path_to_shape.items():
                cumulative += shape[0]
                row_index_file_tuples.append((file_path, cumulative))
            self.file_paths, cumulative_rows = zip(*row_index_file_tuples)
            self.cumulative_rows = np.array(cumulative_rows)
            print("\n".join(self.error_messages))

        def __getitem__(self, key):
            # 1. Handle Empty Dataset
            if not self.file_paths:
                raise ValueError(
                    "No valid data files were found to create the dataset."
                )

            # 2. Input and Column Key Validation
            if not (isinstance(key, tuple) and len(key) == 2):
                raise TypeError(
                    "Indexing must be a tuple of length 2, e.g., `[rows, cols]`."
                )

            row_key, col_key = key
            schema_cols = set(self.schema.columns.keys())

            if isinstance(col_key, str):
                if col_key not in schema_cols:
                    raise KeyError(f"Column '{col_key}' not found in schema.")
            elif isinstance(col_key, list):
                if not set(col_key).issubset(schema_cols):
                    missing = set(col_key) - schema_cols
                    raise KeyError(f"Columns {missing} not found in schema.")

            total_rows = self.cumulative_rows[-1]

            # 3. Handle Integer Indexing
            if isinstance(row_key, int):
                if not (0 <= row_key < total_rows):
                    raise IndexError(
                        f"Row index {row_key} is out of bounds for {total_rows} total rows."
                    )

                file_index = np.searchsorted(
                    self.cumulative_rows, row_key, side="right"
                )
                file_path = self.file_paths[file_index]
                file_format = file_path.split(".")[-1].lower()

                data = FILE_FORMATS_TO_READ_METHOD[self.library][file_format](
                    file_path,
                    **(self.read_kwargs if self.read_kwargs is not None else {}),
                )
                if self.lazy_validation:
                    self.schema.validate(data)

                rows_in_previous_files = (
                    self.cumulative_rows[file_index - 1] if file_index > 0 else 0
                )
                relative_index = row_key - rows_in_previous_files

                if self.library == "polars":
                    return data[int(relative_index), col_key]
                elif self.library == "pandas":
                    # By selecting with [[...]], we ensure the result is a DataFrame,
                    # making its behavior consistent with the slice case.
                    return data.iloc[[relative_index]].loc[:, col_key]

            # 4. Handle Slice Indexing
            elif isinstance(row_key, slice):
                start = row_key.start if row_key.start is not None else 0
                stop = row_key.stop if row_key.stop is not None else total_rows
                step = row_key.step if row_key.step is not None else 1

                # The `stop=0` edge case is handled here, preventing `stop - 1` from becoming negative.
                if start >= stop:
                    empty_df_reader = FILE_FORMATS_TO_READ_METHOD[self.library][
                        self.file_paths[0].split(".")[-1].lower()
                    ]
                    if self.library == "pandas":
                        return empty_df_reader(self.file_paths[0], nrows=0).loc[
                            :, col_key
                        ]
                    else:
                        return empty_df_reader(self.file_paths[0], n_rows=0)[:, col_key]

                first_file_index = np.searchsorted(
                    self.cumulative_rows, start, side="right"
                )
                last_file_index = np.searchsorted(
                    self.cumulative_rows, stop - 1, side="right"
                )
                files_to_read = self.file_paths[first_file_index : last_file_index + 1]

                # This check is technically redundant if `start < stop` but is good for safety.
                if not files_to_read:
                    raise IndexError("Slice is out of bounds.")

                data_parts = [
                    FILE_FORMATS_TO_READ_METHOD[self.library][
                        fp.split(".")[-1].lower()
                    ](fp, **(self.read_kwargs if self.read_kwargs is not None else {}))
                    for fp in files_to_read
                ]

                if self.lazy_validation:
                    for part in data_parts:
                        self.schema.validate(part)

                if self.library == "polars":
                    data = pl.concat(data_parts, how="vertical")
                elif self.library == "pandas":
                    data = pd.concat(data_parts, axis=0, ignore_index=True)

                rows_before_concat = (
                    self.cumulative_rows[first_file_index - 1]
                    if first_file_index > 0
                    else 0
                )
                relative_start = start - rows_before_concat
                relative_stop = stop - rows_before_concat

                if self.library == "polars":
                    return data[relative_start:relative_stop:step, col_key]  # type: ignore
                elif self.library == "pandas":
                    return data.iloc[relative_start:relative_stop:step].loc[:, col_key]  # type: ignore

            else:
                raise TypeError(
                    f"Row key must be an integer or a slice, not {type(row_key)}."
                )

        ValFrame = type(
            name.capitalize(), (), {"__init__": __init__, "__getitem__": __getitem__}
        )
        return ValFrame

    elif library == "pandas":

        def __init__(self, data: pd.DataFrame):
            schema.validate(data)
            pd.DataFrame.__init__(self, data)

        ValFrame = type(
            name.capitalize(),
            (pd.DataFrame,),
            {
                "__init__": __init__,
            },
        )

        return ValFrame

    elif library == "polars":

        def __init__(self, data: pl.DataFrame):
            schema.validate(data)
            pl.DataFrame.__init__(self, data)

        ValFrame = type(
            name.capitalize(),
            (pl.DataFrame,),
            {
                "__init__": __init__,
            },
        )

        return ValFrame
