#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""function that tracks changes in a transactional dataframe
dfdelta is a working version
dfdelta2 is a dev version with added features"""

def df_delta(df, sort_col, on_change_col='index', ignore_cols=None,
             omit_first=False):
    """
    Prints only changes in values in a dataframe from snapshotted row to
    snapshotted row, which may be every row if desired

    Args:
        df: dataframe to be analyzed
        sort_col: str, column by which to sort dataframe
        on_change_col: str; column name or 'index'
                                     changes since last snapshot will be printed
                                     whenever the value in this column changes from the
                                     previous row. If there is a column named 'index', the
                                     actual index will be used instead

    Prints row by row changes in a dataframe when sorted by sort_col, whenever
    the value of on_change_col (or the index, if left to default -- careful,
    it may break if there is a column called index) changes. Rest
    self-explanatory.
    """
    df = df.sort_values(sort_col)
    last = {k: None for k in list(df.columns)}
    last_change_val = None
    for i, (idx, row) in enumerate(df.iterrows()):
        print('index: {}'.format(idx))
        if on_change_col == 'index':
                increment = True
        else:
            if i == 0 or row[on_change_col] != last_change_val:
                increment = True
                last_change_val = row[on_change_col]
                print('    {}: {}'.format(on_change_col, last_change_val))
        if increment:
            for col in row.index:
                if ignore_cols is None or col not in ignore_cols:
                    if col != on_change_col:
                        if i == 0 and not omit_first:
                            print('        {}: {}'.format(col, row[col]))
                        elif (not(pd.isnull(row[col]) and pd.isnull(
                              last[col])) and row[col] != last[col]):
                            print('        {}: {} --> {}'.format(col,
                                  last[col], row[col]))
                            last[col] = row[col]


def df_delta2(df, sort_col=None, on_change_col=None, ignore_cols=None,
                            ignore_col_threshold=None, omit_first=False,
                            always_print_index=False, debug=False,
                            renumber_index=False, show_intermediates=True,
                            handle_reversions=None):
    """
    Prints only changes in values in a dataframe from snapshotted row to
    snapshotted row, which may be every row if desired

    Args:
        df: dataframe to be analyzed
        sort_col: str, column by which to sort dataframe.
                  if None, df is taken to be already sorted
        on_change_col: str; column name upon whose change to report changes in
                       all other colums since last snapshot
                       if None, index will be used
        ignore_cols: list, columns to be left out of the comparison
        ignore_col_threshold: float from 0 to 1, only columns that change less often than
                              (not equal to) this will be reported; e.g. if ignore_threshold=1,
                              only columns that change every time on_change_col times will
                              be ignored.
        ignore_values: if list, any column change to this value will be ignored.
                       if dict, should be in form {column_name: [value1, value2...]}
        show_only_values: if list, any column change not to this value will be ignored.
                          if dict, should be in form {column_name: [value1, value2...]}
        omit_first: if False, the first value will show all columns of the dataframe
                    with their initial values. if True, print results will start with
                    first row with a non-ignored change.
        always_print_index: if True, will show every row change, even if it includes
                            no on_change_col change and thus nothing to report.
        renumber_index: if True, uses iloc for loc in index
        show_intermediates: if True, if a column goes through several values
        handle_reversions: if None, nothing different is done to a reversion (a value
                            that changes back to its previous value when
                           on_change_col changes)
                           if 'highlight', show them as usual but highlight them
                           if 'omit_second', highlight the first but not the second
        debug: if True, shows a few things that are happening

    Returns:
        None (just prints), unless debug=True, then returns list of print results
    """
    from collections import defaultdict
    # function for debugging
    def dprint(message, value=None):
        if debug:
        if value is None:
        print('*** {}'.format(message))
    else:
        print('*** {}: {}'.format(message, value))
    # changes in change_col are numbered invisibly to user
    # curr_change_count counts those changes
    # change_counter counts how many column changes are at each such change
    # after results are collected, ignore_threshold may reduce change_counter
    # and end up with changes being omitted afterwards when printing
    curr_change_count = 0
    change_counter = defaultdict(int)    # change_count: #
    col_changes = defaultdict(int)    # column: number
    to_print = [] # tuple of change_counter, column, report
    if sort_col is not None:
        dprint('sorted by', sort_col)
    df = df.sort_values(sort_col)
    if renumber_index:
        df = df.reset_index(drop=True)
    # store last seen values
    last = {k: None for k in list(df.columns)}
    last_change_val = None
    for i, (idx, row) in enumerate(df.iterrows()):
        dprint('idx', idx)
    if on_change_col == 'index':
        increment = True
    curr_change_count += 1
    to_print.append([curr_change_count, '_index', 'index: {}'.format(idx)])
    dprint("added index because it's change col")
    else:
        if always_print_index:
        to_print.append([curr_change_count, '_index', 'index: {}'.format(idx)])
    dprint("added index because always_print_index")
    if i == 0 or not equalorbothnull(on_change_col, last_change_val):
        increment = True
    curr_change_count += 1
    last_change_val = row[on_change_col]
    # if always_print_index:
    # to_print.append([curr_change_count, '_index', 'index: {}'.format(idx)])
    to_print.append([curr_change_count, '_on_change', '    {}: {}'.format(on_change_col, last_change_val)])
    dprint('added change col')
    if increment:
        for col in row.index:
        if ignore_cols is None or col not in ignore_cols:
        if col != on_change_col:
        if i == 0:
        if not omit_first:
        to_print.append([curr_change_count, 'first', '        {}: {}'.format(col, row[col])])
    dprint('added col because i==0', col)
    last[col] = row[col]
    elif not equalorbothnull(row[col], last[col]):
        to_print.append([curr_change_count, col, '        {}: {} --> {}'.format(col,
                        last[col], row[col])])
    dprint('added col because change', col)
    change_counter[curr_change_count] += 1
    col_changes[col] += 1
    last[col] = row[col]
    cols_to_ignore = []
    if ignore_threshold is None:
        new_to_print = to_print
    else:
        for col, num in col_changes.items():
        if num >= curr_change_count * ignore_threshold:
        cols_to_ignore.append(col)
    # decrement counter wherever cols_to_ignore member appears
    new_to_print = []
    for change_num, type_, message in to_print:
        if type_ not in cols_to_ignore:
        new_to_print.append([change_num, type_, message])
    else:
        change_counter[change_num] -= 1
    dprint('change_counter{} decremented to {}'.format(change_num, change_counter[change_num]))
    dprint('cols_to_ignore', cols_to_ignore)
    to_really_print = []
    for change_num, type_, message in new_to_print:
        if change_counter[change_num] > 0:
        to_really_print.append(message)
    if debug:
        return to_really_print
    else:
        for item in to_really_print:
        print(item)
