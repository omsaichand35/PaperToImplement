import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from retrieval.search import (  # noqa: E402
    find_exact_symbol,
    search_documents
)


def test_exact_symbol():

    result = find_exact_symbol(
        "torch.nn.ConvTranspose1d"
    )

    assert result is not None

    assert (
        result["symbol"]
        ==
        "torch.nn.ConvTranspose1d"
    )


def test_parameter_query():

    results = search_documents(
        "output_padding"
    )

    assert len(results) > 0

    assert (
        results[0]["symbol"]
        ==
        "torch.nn.ConvTranspose1d"
    )


def test_lstm_query():

    results = search_documents(
        "batch_first hidden_size"
    )

    assert len(results) > 0

    assert (
        results[0]["symbol"]
        ==
        "torch.nn.LSTM"
    )


def test_batchnorm_query():

    results = search_documents(
        "running mean variance"
    )

    assert len(results) > 0

    assert (
        results[0]["symbol"]
        ==
        "torch.nn.BatchNorm1d"
    )
