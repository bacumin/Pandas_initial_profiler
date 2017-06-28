#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""function that creates a single bar chart-style line"""

def bar_chart_line(value, maximum, rounding='middle', num_chars=30, value_char='#', 
                   empty_char='_', name=None, name_width=None, name_align='right', 
                   order=['bar', 'proportion', 'absolute'], proportion_decimals=3, 
                   absolute_width=None, absolute_align='left', spaces=1):
    """
    Function to return a string simply showing the relationship between a portion and
    its whole.

    Outputs something like:
       - 'under_18 #######.......................'
    or - '            under_18 #######.......................'
    or - '            under_18 #######....................... .221 15432/69809'
    The idea is to determine the maximum widths of the names and, if desired, absolute
    values beforehand, and then call this function iterably so everything aligns.

    Arguments:
        :value: {int|float} The value that is part of a whole
        :maximum: {int|float} The whole, which would result in a complete bar
        :rounding: {'ceiling|floor|nearest|middle'} middle rounds always towards half the
                   maximum. This is useful to only ever show a totally blank or totally
                   complete line if the value == 0 or value == maximum
        :num_chars: {int} number of characters in a complete bar
        :value_char: {str} single string to use for bar
        :empty_char: {str} single string to use to show length of maximum bar value
        :name: {str} a name for the bar being shown
        :name_width: {int} name will be padded out to this length, in order to
                     align bars
        :name_align: {'right', 'left', 'center'} I like 'right'
        :order: {list containing one to three of ['bar', 'proportion', 'absolute']}
                The order in which to add items after the name. Any item not listed is 
                not included in the returned string. These items are:
                - bar, e.g. '####......'
                - proportion, from 0. to 1.
                - absolute, <value>/<maximum> recapitulated arguments
        :proportion_decimals: {int}, e.g. 3 gives '.667', '.000' or '1  ' (special case)
        :absolute_width: {int}, width to align 'absolute' component
        :absolute_align: {str} ['left', 'right', 'center'], self-explanatory
        :spaces: {int} number of spaces to put between items

    Exception:
        AssertionError if unexpected argument
        AssertionError if name_width < len(name)
        Note: no AssertionError if absolute_width < len(absolute). For this reason,
              it might be a good idea to put absolute last in order, if used.

    Returns:
        {str} without \n, containing concatenated [name|bar|proportion|absolute]

    """
    def align_char(alignment):
        """
        Returns the pyformat.info character to align text left, right or center.
        """
        if alignment == 'left':
            return ''
        elif alignment == 'right':
            return '>'
        elif alignment == 'center':
            return '^'
        else:
            raise ValueError("Must be 'left', 'center', or 'right', not '{}'".format(alignment))

    from math import floor, ceil
    assert value <= maximum
    assert not (name_width is not None and name is None), "If name_width is specified, \
        name must be specified too."
    assert rounding in ['nearest', 'ceiling', 'floor', 'middle']
    assert name_align in ['left', 'right', 'center']
    assert set(order) in [{'absolute', 'bar', 'proportion'},
                          {'bar', 'proportion'},
                          {'absolute', 'bar'},
                          {'absolute', 'proportion'},
                          {'bar'},
                          {'proportion'},
                          {'absolute'}]
    proportion = value / maximum
    results = {}
    if name is not None:
        order = ['name'] + order
        if name_width is None:
            name_width = len(name)
        assert name_width >= len(name), 'name_width must not be smaller than the name length'
        align_string = '{{:{}{}}}'.format(align_char(name_align), name_width)
        name = align_string.format(name)
        results['name'] = name
    if 'bar' in order:
        num_chars_float = proportion * num_chars
        if rounding == 'nearest':
            num_chars_int = round(num_chars_float)
        elif rounding == 'ceiling':
            num_chars_int = ceil(num_chars_float)
        elif rounding == 'floor':
            num_chars_int = floor(num_chars_float)
        else:
            if value < (maximum / 2):
                num_chars_int = ceil(num_chars_float)
            else:
                num_chars_int = floor(num_chars_float)
        num_on = int(num_chars_int)
        num_off = num_chars - num_on
        results['bar'] = '{}{}'.format(value_char * num_on, empty_char * num_off)
    if 'proportion' in order:
        if round(proportion, proportion_decimals) == 1.:
            results['proportion'] = '1{}'.format(' '*proportion_decimals)
        else:
            align_string='{{:.{}f}}'.format(proportion_decimals)
            results['proportion'] = align_string.format(proportion)[1:]
    if 'absolute' in order:
        absolute = '{}/{}'.format(value, maximum)
        if absolute_width is None or absolute_width < len(absolute):  # will not raise
            absolute_width = len(absolute)                            # an Exception
        align_string = '{{:{}{}}}'.format(align_char(absolute_align), absolute_width)
        results['absolute'] = align_string.format(absolute)
    final = [results[x] for x in order]
    spacing = ' '*spaces
    return spacing.join(final)
