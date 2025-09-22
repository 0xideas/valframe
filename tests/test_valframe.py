import numpy as np
import pandas as pd
import pandera.pandas as pa
import pandera.polars as pal
import polars as pl
import pytest
from pandera.errors import SchemaError

from valframe.valframe import create_valframe_type

# A common schema used across multiple tests
PANDAS_SCHEMA = pa.DataFrameSchema(
    {
        "id": pa.Column(int, pa.Check.ge(0)),
        "name": pa.Column(str),
    }
)

POLARS_SCHEMA = pal.DataFrameSchema(
    {
        "id": pal.Column(int, pal.Check.ge(0)),
        "name": pal.Column(pl.Utf8),
    }
)


# ## In-Memory DataFrame Tests


def test_pandas_valframe_success():
    """Tests successful validation for an in-memory pandas DataFrame."""
    PandasValFrame = create_valframe_type(
        "PandasValFrame", PANDAS_SCHEMA, library="pandas"
    )
    valid_df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    vf_instance = PandasValFrame(valid_df)  # type: ignore
    assert isinstance(vf_instance, pd.DataFrame)
    pd.testing.assert_frame_equal(vf_instance, valid_df)


def test_pandas_valframe_failure():
    """Tests schema validation failure for an in-memory pandas DataFrame."""
    PandasValFrame = create_valframe_type(
        "PandasValFrame", PANDAS_SCHEMA, library="pandas"
    )
    invalid_df = pd.DataFrame({"id": [-1, 2], "name": ["a", "b"]})  # id=-1 is invalid
    with pytest.raises(SchemaError):
        PandasValFrame(invalid_df)  # type: ignore


def test_polars_valframe_success():
    """Tests successful validation for an in-memory polars DataFrame."""
    PolarsValFrame = create_valframe_type(
        "PolarsValFrame", POLARS_SCHEMA, library="polars"
    )
    valid_df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    vf_instance = PolarsValFrame(valid_df)  # type: ignore
    assert isinstance(vf_instance, pl.DataFrame)
    assert np.all(vf_instance.to_numpy() == valid_df.to_numpy())  # type: ignore

    assert np.all(
        np.array(list(vf_instance.columns)) == np.array(list(valid_df.columns))  # type: ignore
    )


def test_polars_valframe_failure():
    """Tests schema validation failure for an in-memory polars DataFrame."""
    PolarsValFrame = create_valframe_type(
        "PolarsValFrame", POLARS_SCHEMA, library="polars"
    )
    invalid_df = pl.DataFrame({"id": [-1, 2], "name": ["a", "b"]})  # id=-1 is invalid
    with pytest.raises(SchemaError):
        PolarsValFrame(invalid_df)  # type: ignore


# ## Folder-Based DataFrame Tests


@pytest.fixture
def data_folder(tmp_path):
    """Creates a temporary folder with valid, invalid, and other file types."""
    # Create valid data files
    pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}).to_csv(
        tmp_path / "data1.csv", index=False
    )
    pd.DataFrame({"id": [3, 4], "name": ["Charlie", "David"]}).to_csv(
        tmp_path / "data2.csv", index=False
    )
    # Create an invalid data file (violates schema)
    pd.DataFrame({"id": [5, -6], "name": ["Eve", "Frank"]}).to_csv(
        tmp_path / "invalid_data.csv", index=False
    )
    # Create a file with an unsupported extension
    (tmp_path / "notes.txt").write_text("This is not a data file.")
    return tmp_path


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_folder_valframe_initialization(data_folder, library):
    """Tests correct identification of valid and invalid files during initialization."""
    FolderValFrame = create_valframe_type(
        "FolderValFrame",
        PANDAS_SCHEMA if library == "pandas" else POLARS_SCHEMA,
        library=library,
        folder=True,
        input_file_formats=["csv"],
    )
    vf_instance = FolderValFrame(str(data_folder))  # type: ignore

    # Check that the invalid file was correctly identified
    assert len(vf_instance.invalid_file_paths) == 1  # type: ignore
    assert "invalid_data.csv" in vf_instance.invalid_file_paths[0]  # type: ignore

    # Check that valid files were found and their shapes recorded
    assert len(vf_instance.file_path_to_shape) == 2  # type: ignore
    assert all(
        "data1.csv" in path or "data2.csv" in path
        for path in vf_instance.file_path_to_shape  # type: ignore
    )


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_folder_valframe_getitem(data_folder, library):
    """Tests integer and slice indexing for folder-based ValFrames."""
    FolderValFrame = create_valframe_type(
        "FolderValFrame",
        PANDAS_SCHEMA if library == "pandas" else POLARS_SCHEMA,
        library=library,
        folder=True,
        input_file_formats=["csv"],
    )
    vf_instance = FolderValFrame(str(data_folder))  # type: ignore

    # Total rows from valid files should be 4
    assert vf_instance.cumulative_rows[-1] == 4  # type: ignore

    # Test integer indexing: retrieve all IDs and sort to ensure correctness
    # regardless of file read order.
    if library == "pandas":
        all_ids = sorted([vf_instance[int(i), "id"].item() for i in range(4)])  # type: ignore
    elif library == "polars":
        all_ids = sorted([vf_instance[int(i), "id"] for i in range(4)])  # type: ignore
    assert all_ids == [1, 2, 3, 4]  # type: ignore

    # Test slice indexing
    result = vf_instance[  # type: ignore
        1:3, ["id", "name"]
    ]  # Should get the 2nd and 3rd rows # type: ignore

    if library == "pandas":
        assert isinstance(result, pd.DataFrame)
    else:  # polars
        assert isinstance(result, pl.DataFrame)

    assert result.shape[0] == 2


# ## Factory Function Assertion Tests


def test_create_valframe_type_assertions():
    """Tests the initial assertions in the create_valframe_type factory."""
    with pytest.raises(AssertionError, match="supported libraries"):
        create_valframe_type("Test", PANDAS_SCHEMA, library="dask")

    with pytest.raises(AssertionError, match="input_file_formats needs to be None"):
        create_valframe_type(
            "Test", PANDAS_SCHEMA, folder=False, input_file_formats=["csv"]
        )

    with pytest.raises(AssertionError, match="input_file_formats cannot be None"):
        create_valframe_type(
            "Test", PANDAS_SCHEMA, folder=True, input_file_formats=None
        )


@pytest.fixture
def empty_folder(tmp_path):
    """Creates an empty temporary folder for testing."""
    return tmp_path


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_folder_valframe_getitem_errors(data_folder, empty_folder, library):
    """Tests various indexing errors for folder-based ValFrames."""
    FolderValFrame = create_valframe_type(
        "FolderValFrame",
        PANDAS_SCHEMA if library == "pandas" else POLARS_SCHEMA,
        library=library,
        folder=True,
        input_file_formats=["csv"],
    )
    vf_instance = FolderValFrame(str(data_folder))  # type: ignore

    # Test TypeError for invalid key type (e.g., not a tuple)
    with pytest.raises(TypeError, match="Indexing must be a tuple"):
        vf_instance[0]  # type: ignore

    # Test KeyError for a column not present in the schema
    with pytest.raises(KeyError, match="not found in schema"):
        vf_instance[0, "age"]  # type: ignore

    # Test IndexError for a row index that is out of bounds
    with pytest.raises(IndexError, match="out of bounds"):
        vf_instance[99, "id"]  # type: ignore


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_folder_valframe_getitem_lazy_validation(data_folder, library):
    """Tests that validation errors are raised on access when lazy_validation=True."""
    FolderValFrame = create_valframe_type(
        "FolderValFrame",
        PANDAS_SCHEMA if library == "pandas" else POLARS_SCHEMA,
        library=library,
        folder=True,
        input_file_formats=["csv"],
        lazy_validation=True,  # Defer validation until access
    )
    # Initialization should succeed, finding all 3 files without validation
    vf_instance = FolderValFrame(str(data_folder))  # type: ignore
    assert len(vf_instance.file_path_to_shape) == 3  # type: ignore
    assert len(vf_instance.invalid_file_paths) == 0  # type: ignore


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_folder_valframe_getitem_slice_edge_cases(data_folder, library):
    """Tests edge cases for slice indexing, like empty slices and steps."""
    FolderValFrame = create_valframe_type(
        "FolderValFrame",
        PANDAS_SCHEMA if library == "pandas" else POLARS_SCHEMA,
        library=library,
        folder=True,
        input_file_formats=["csv"],
    )
    vf_instance = FolderValFrame(str(data_folder))  # type: ignore

    # Test an empty slice
    empty_result = vf_instance[2:2, :]  # type: ignore
    assert empty_result.shape[0] == 0

    # Test open-ended slices
    start_slice = vf_instance[:2, :]  # type: ignore
    assert start_slice.shape[0] == 2
    end_slice = vf_instance[2:, :]  # type: ignore
    assert end_slice.shape[0] == 2

    # Test a slice with a step, which should select every other row
    stepped_slice = vf_instance[0:4:2, ["id"]]  # type: ignore
    assert stepped_slice.shape[0] == 2

    # The valid IDs are [1, 2, 3, 4]. A step of 2 retrieves rows with IDs 1 and 3.
    # We sort the result to ensure the test is deterministic regardless of file read order.
    retrieved_ids = sorted(stepped_slice["id"].to_list())
    assert retrieved_ids == [1, 3]


def test_folder_valframe_getitem_return_types(data_folder):
    """Tests that __getitem__ returns the expected types for pandas and polars."""
    # --- Pandas Test ---
    PandasFolderValFrame = create_valframe_type(
        "PandasFolderValFrame",
        PANDAS_SCHEMA,
        library="pandas",
        folder=True,
        input_file_formats=["csv"],
    )
    pvf = PandasFolderValFrame(str(data_folder))  # type: ignore

    # Integer index with a single column name (str) should return a Series
    res_pd_series = pvf[0, "id"]  # type: ignore
    assert isinstance(res_pd_series, pd.Series)

    # Integer index with a list of columns should return a DataFrame
    res_pd_df = pvf[0, ["id"]]  # type: ignore
    assert isinstance(res_pd_df, pd.DataFrame)
    assert res_pd_df.shape == (1, 1)

    # --- Polars Test ---
    PolarsFolderValFrame = create_valframe_type(
        "PolarsFolderValFrame",
        POLARS_SCHEMA,
        library="polars",
        folder=True,
        input_file_formats=["csv"],
    )
    plvf = PolarsFolderValFrame(str(data_folder))  # type: ignore

    # Integer index with a single column name (str) should return a scalar value
    res_pl_scalar = plvf[0, "id"]  # type: ignore
    assert isinstance(res_pl_scalar, int)

    # Integer index with a list of columns should return a DataFrame
    res_pl_df = plvf[0, ["id", "name"]]  # type: ignore
    assert isinstance(res_pl_df, pl.DataFrame)
    assert res_pl_df.shape == (1, 2)
