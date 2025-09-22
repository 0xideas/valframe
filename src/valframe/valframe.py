import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import pandera.pandas as pa
import polars as pl
from beartype import beartype

SUPPORTED_FILE_FORMATS = ["str", "parquet"]
SUPPORTED_LIBRARIES = ["polars", "pandas"]

FILE_FORMATS_TO_READ_METHOD = {
    "polars": {"csv": pl.read_csv, "parquet": pl.read_parquet},
    "pandas": {"csv": pd.read_csv, "parquet": pd.read_parquet},
}


@beartype
def create_valframe_type(
    name: str,
    schema: pa.DataFrameSchema,
    library: str = "polars",
    folder: bool = False,
    nested_level: int = 0,
    input_file_formats: Optional[list[str]] = None,
    read_kwargs: Optional[dict[str, Any]] = None,
    max_errors: Optional[int] = 10,
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

        def __init__(self, name: str, path: str):  # type: ignore
            assert path.startswith("..") is False

            self.path = path
            self.schema = schema
            self.library = library
            self.nested_level = nested_level
            self.input_file_formats = input_file_formats
            self.read_kwargs = read_kwargs

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
                                ](file_path, **read_kwargs)
                                schema.validate(data)
                                self.file_path_to_shape[file_path] = data.shape
                            except Exception as e:
                                self.invalid_file_paths.append(file_path)
                                error_message = f"reading data or valframe schema validation failed for {file_path}: {e}"
                                self.error_messages.append(error_message)

                    if (
                        max_errors is not None
                        and len(self.error_messages) >= max_errors
                    ):
                        raise Exception("\n".join(self.error_messages))

            cumulative, row_index_file_tuples = 0, []
            for file_path, shape in self.file_path_to_shape.items():
                cumulative += shape[0]
                row_index_file_tuples.append((cumulative, file_path))
            self.file_paths, cumulative_rows = zip(*row_index_file_tuples)
            self.cumulative_rows = np.array(cumulative_rows)

        def __getitem__(self, key):
            assert (
                isinstance(key, tuple) and len(key) == 2
            ), "key must be tuple of length 2"
            row_key, col_key = key

            if isinstance(row_key, slice):
                start, step, stop = row_key.start, row_key.step, row_key.stop
                first_file_index = np.argmax(self.cumulative_rows > start)
                last_file_index = np.argmax(self.cumulative_rows > stop)

                data = [
                    FILE_FORMATS_TO_READ_METHOD[library][
                        file_path.split(".")[-1].lower()
                    ](file_path, **self.read_kwargs)
                    for file_path in self.file_paths[first_file_index:last_file_index]
                ]
                if library == "polars":
                    data = pl.concat(data, how="vertical")
                elif library == "pandas":
                    data = pd.concat(data, axis=0)

                rel_start_index = start - self.cumulative_rows[first_file_index]
                rel_end_index = stop - self.cumulative_rows[first_file_index]
                if library == "polars":
                    return data[rel_start_index:rel_end_index:step, col_key]  # type: ignore
                elif library == "pandas":
                    return data.iloc[rel_start_index:rel_end_index:step, col_key]  # type: ignore
            elif isinstance(row_key, int):
                file_index = np.argmax(self.cumulative_rows > row_key)
                file_path = self.file_paths[file_index]
                file_format = file_path.split(".")[-1].lower()
                data = FILE_FORMATS_TO_READ_METHOD[library][file_format](
                    file_path, **self.read_kwargs
                )

                rel_index = row_key - self.cumulative_rows[file_index - 1]

                if library == "polars":
                    return data[rel_index, col_key]
                elif library == "pandas":
                    return data.iloc[rel_index, col_key]

        ValFrame = type(
            name.capitalize(), (), {"__init__": __init__, "__getitem__": __getitem__}
        )
        return ValFrame

    elif library == "pandas":

        def __init__(self, data: pd.DataFrame):
            schema.validate(data)
            super(self).__init__(data)

        ValFrame = type(
            name.capitalize(),
            (pd.DataFrame),
            {
                "__init__": __init__,
            },
        )

        return ValFrame

    elif library == "polars":

        def __init__(self, data: pl.DataFrame):
            schema.validate(data)
            super(self).__init__(data)

        ValFrame = type(
            name.capitalize(),
            (pl.DataFrame),
            {
                "__init__": __init__,
            },
        )

        return ValFrame
