#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""functions that evaluate DataFrames or Series or data points,
either alone or in pairwise comparison"""

def is_numlike(value):
    """Determines if the value is a number or if it a string that can be
    expressed as a number"""
    try:
        _ = float(value)
        return True
    except:
        return False
    
    
def is_intlike(value, epsilon=1e-8, strings_ok=True):
    """
    Determines if a value can be expressed as an integer, or something close enough
    to it to make no difference.if

    Arguments:
        :value: {int, float[, str]} the 'number' to be evaluated
        :epsilon: {float} the allowable distance between a float(binary representation)
                   and integer (decimal representation), e.g. 1.999999999999 ~= 2
        :strings_ok': {bool} if True, then '2.0' will return True, else False

    Returns:
        {bool}, whether value is evaluated as intlike
    """
    if pd.isnull(value):
        return False
    if isinstance(value, str):
        try:
            to_evaluate = float(value)
        except ValueError:
            return False
    else:
        to_evaluate = value
    import numpy as np
    if (np.issubdtype(type(np.array([to_evaluate])[0]), int) or
        abs(int(to_evaluate) - float(to_evaluate)) < epsilon):
        return True
    else:
        return False

def guess_datatype(series):
    """Guesses datatype of pandas series/column, will return float, int or str,
    or 'mixed' or series dtype"""
    # ^ should probably tighten this up
    def maybe_int(number):
        try:
            if number == int(number):
                return True
            else:
                return False
        except:
            return False
    if series.dtype == float:
        # check if all non-null values can be expressed as int
        if all(series.dropna().apply(maybe_int)):
            return int
        else:
            return float
    elif series.dtype == 'O':
        # check if any value is a string
        if any(df_[col].apply(lambda x: type(x) == str)):
            return str
        else:
            return 'mixed'
    else:
        return series.dtype


def is_monotonic(series, order_series=None, equal_allowed=False):
    """
    Determines whether the values in a pandas Series are monotonic, with its
    own or another Series's sort order

    Args:
        series: a pandas Series
        order_series: default=None
                      a pandas Series. If None, series will be unsorted.
                      Note that this means you can pass the same series to
                      ``series`` and ``order_series`` and you will only get
                      the same result as passing None to ``order_series`` if
                      ``series`` is already sorted in increasing order.
        equal_okay: whether to consider consecutive equal values to be
                    allowed in a monotonic series.

    Returns:
        1 if series is monotonic increasing
        -1 if series is monotonic decreasing
        0 if series is not monotonic
    """

    if order_series is not None:
        df = pd.concat([series, order_series])
        df.columns = ['s', 'o']
        df = df.sort_values('o')
        s = df['s']
    else:
        s = series
    direction = None  # 0 for unknown, +1 for increasing, -1 for decreasing
    for i, val in enumerate(s):
        if i > 1:
            if val > last_val:
                this_dir = 1
            elif val == last_val:
                this_dir = 0
            else:
                this_dir = -1
            if direction is None:  #no direction decided yet
                if this_dir != 0:
                    direction = this_dir
                elif not equal_allowed:  # otherwise remains None
                    direction = 0
            else:
                if equal_allowed:
                    if this_dir == 0 or this_dir == direction:
                        pass
                    else:
                        direction = 0
                else:
                    if this_dir == direction:
                        pass
                    else:
                        direction = 0
        if direction == 0:
            break
        last_val = val
    return direction


def equalorbothnull(a, b):
    """Returns True if a == b, including when both a and b are NaN"""
    if a == b:
        return True
    elif pd.isnull(a) and pd.isnull(b):
        return True
    else:
        return False




