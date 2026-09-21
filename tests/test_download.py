from src.data.download import COLUMNS, klines_to_dataframe

# One real-shaped Binance klines row (fields per the documented klines API):
# [open_time, open, high, low, close, volume, close_time, quote_asset_volume,
#  num_trades, taker_buy_base, taker_buy_quote, ignore]
SAMPLE_ROW = [
    1726902000000,
    "63000.00000000",
    "63500.00000000",
    "62900.00000000",
    "63250.00000000",
    "120.5",
    1726905599999,
    "7580000.0",
    5000,
    "60.2",
    "3800000.0",
    "0",
]


def test_klines_to_dataframe_parses_shape():
    df = klines_to_dataframe([SAMPLE_ROW])

    assert len(df) == 1
    assert "ignore" not in df.columns
    assert df["open"].iloc[0] == 63000.0
    assert df["close"].iloc[0] == 63250.0
    assert df["num_trades"].iloc[0] == 5000
    assert df.index.tz is not None


def test_klines_to_dataframe_sorts_by_time():
    row_later = SAMPLE_ROW.copy()
    row_earlier = SAMPLE_ROW.copy()
    row_earlier[0] = 1726902000000 - 3600_000
    row_earlier[6] = row_earlier[6] - 3600_000

    df = klines_to_dataframe([row_later, row_earlier])
    assert df.index.is_monotonic_increasing


def test_all_expected_columns_handled():
    df = klines_to_dataframe([SAMPLE_ROW])
    expected_remaining = set(COLUMNS) - {"ignore", "open_time"}
    assert expected_remaining.issubset(set(df.columns) | {"close_time"})
