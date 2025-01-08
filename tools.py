import pandas as pd
import math

def load_csv(path: str) -> pd.DataFrame:
    """
    Function to load a csv
    Parameters : path of the csv
    Return : pd.DataFrame containing the csv datass
    """
    try:
        csv = pd.read_csv(path)
    except Exception as e:
        print(f"loading csv error : {e}")
        return None
    return csv

def min(column: pd.Series):
    min = column[0]
    for value in column:
        if not pd.isna(value):
            if value < min:
                min = value
    return min

def max(column: pd.Series):
    max = column[0]
    for value in column:
        if not pd.isna(value):
            if value > max:
                max = value
    return max

def mean(column: pd.Series) -> float :
    """
    Function that calculate the mean of a variable
    Parameters : a pd.Series column containing datas
    Return : a float containing the calculated mean
    """
    sum = 0
    size = 0
    for value in column:
        if not pd.isna(value):
            sum += value
            size += 1
    return sum / size

def std(column: pd.Series) -> float :
    """
    Function that calculate the std of a variable
    Parameters : a pd.Series column containing datas
    Return : a float containing the calculated std
    """
    sum = 0
    size = 0
    for value in column:
        if not pd.isna(value):
            sum += (mean(column) - value) ** 2
            size += 1
    return math.sqrt(sum / size)


def percentile(column: pd.Series, percentile: int) -> int : 
    """
    Function that look for the percentile asked for a variable
    Parameters : 
        - a pd.Series column containing datas
        - the percentile asked for that column
    Return : index of the percentile asked
    """
    try:
        cleanData = column.dropna()
        sortedData = cleanData.sort_values()
        n = len(sortedData)
        index = (n - 1) * percentile
        if index.is_integer():
            return sortedData.iloc[int(index)]
        lowerIndex = int(index)
        upperIndex = lowerIndex + 1

        lowerValue = sortedData.iloc[lowerIndex]
        upperValue = sortedData.iloc[upperIndex]

        fraction = index - lowerIndex
        return lowerValue + fraction * (upperValue - lowerValue)

    except Exception as e:
        print(f"Error: {e}")
