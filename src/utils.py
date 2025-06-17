import sys
from pathlib import Path
import yfinance as yf


def download_dataset(ticker, period="2y", interval="1d"):
    """Download Yfinance dataset

    Args:
        ticker (str): Stock Code
        period (str, optional): duration of period in years. Defaults to "2y".
        interval (str, optional): interval between each value of the stock. Defaults to "1d".

    Returns:
        _type_: _description_
    """
    df = yf.download(ticker, period=period, interval=interval)
    return df


def resolve_root(levels_up: int = 1):
    """
    Adds the project root directory to sys.path for script imports.

    Args:
        levels_up (int): How many directory levels to go up from the current script.
                         Default is 1 (e.g., from 'scripts/' to project root).
    """
    current_path = Path(__file__).resolve()
    root_path = current_path.parents[levels_up]
    sys.path.append(str(root_path))
