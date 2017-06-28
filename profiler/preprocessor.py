#!/usr/bin/env python

from __future__ import print_function, division, unicode_literals, absolute_import

import datetime
import json
import logging
import os
import re
import sys
import traceback

import numpy as np
import pandas as pd

from scipy import stats

class RuleError(Exception):
    pass

class FailureRuleTriggered(Exception):
    pass

class MissingColumnError(Exception):
    pass

class Preprocessor:
    """Takes a pandas dataframe and applies user-specified rules.
    
    Note that it does not change the original dataframe in place, but instead
    creates a modified copy called with Preprocessor.df
    
    The rules can either be created one by one and saved as json,
    or loaded from json, or a combination of the two.
    
    args:
        df: a pandas DataFrame
        path_to_log: what it sounds like
        overwrite_log: if False, logging will be appended to
                       existing log, if it exists
        train: whether this is a training or test set.
               this affects the behavior of how the right-side expressions
               mean, median and mode are applied.
               call the print_rules_help() method for info
        print_output: If True, prints results. Turn this to False if your IDE
                      prints logging output.
    """
    def __init__(self, df, path_to_log, overwrite_log=False, train=False, print_output=True):
        self.df = df.copy()
        self.train = train
        self.print_output = print_output
        self.path_to_log = path_to_log
        assert '_metadata_' not in self.df.columns, "Dataframe must not contain a column named '_metadata_'"
        assert '_exec_' not in self.df.columns, "Dataframe must not contain a column named '_metadata_'"
        self.rules = []
        self.training_mask = None
        self.median_mean_mode_called = False
        if overwrite_log:
            if os.path.isfile(path_to_log):
                print('log removed')
                os.remove(path_to_log)
        self.logger = logging.getLogger(path_to_log)
        self._log_handler = logging.FileHandler(path_to_log)
        self._log_formatter = logging.Formatter('%(levelname)s %(message)s')
        self._log_handler.setFormatter(self._log_formatter)
        self.logger.addHandler(self._log_handler) 
        self.logger.setLevel(logging.INFO)
        self._log_info('#', suppress_num = True)
        self._log_info('###############################################################################', suppress_num = True)
        self._log_info('#', suppress_num = True)
        self._log_info('Preprocessor class created on {}'.format(datetime.datetime.isoformat(datetime.datetime.now())), suppress_num = True)
        self._log_info('Path to this log: {}'.format(os.path.abspath(path_to_log)), suppress_num = True)
        self._log_info('Dataframe contains {} rows'.format(len(self.df)), suppress_num = True)
        self._log_info('Dataframe contains {} columns'.format(len(self.df.columns)), suppress_num = True)
        self._log_info('Column names: ' + ', '.join(self.df.columns), suppress_num = True)

    def _log_info(self, message, add_num = 0, suppress_num=False):
        """add_num can take a positive number to align messages with the following rules. If suppress_num is True,
        no number will be printed."""
        if not suppress_num:
            message = '{{{}}} {}'.format(len(self.rules)+add_num-1, message)
        if self.print_output:
        	print(message)
        self.logger.info(message)
    def _log_error(self, message, add_num=0, suppress_num=False):
        if not suppress_num:
            message = '{{{}}} {}'.format(len(self.rules)+add_num-1, message)
    	if self.print_output:
        	print('ERROR: '+message, file=sys.stdout)
        self.logger.error(message)
    def add_exec(self, text):
        """Executes pandas code. The dataframe must be referred to as %df%."""
        self.add_rule('_exec_', 'exec|'+text)
    def add_metadata(self, text):
        """Adds a 'column' called '_metadata', to which 'rules' can be appended, e.g. info on the
        dataset, the date, etc."""
        self.rules.append(['_metadata_', text])
        self._log_info('Added metadata: {}'.format(text), suppress_num=True)
    def add_boundary_warnings(self, columns):
        """Shortcut to add warnings for column (which must be numeric) when new data is less than the minimum
        or greater than the maximum value of the training data in that column."""
        self._log_info('add_boundary_warnings method called for column(s) {}; details follow'.format(columns), suppress_num=True)
        if type(columns) is not list:
            columns = [columns]
        for i, column in enumerate(columns):
            rule = 'lt|{0}|->|warn|%Value is less than minimum {0}%'.format(self.df[column].min())
            self.add_rule(column, rule)
            rule = 'gt|{0}|->|warn|%Value is greater than maximum {0}%'.format(self.df[column].max())
            self.add_rule(column, rule)
    def add_categorical_levels_warning(self, columns):
        """Shortcut too add warning for column when new data is not one of the set of unique values of that
        column present in the training set."""
        self._log_info('add_categorical_levels_warning method called for column(s) {}; details follow'.format(columns), suppress_num=True)
        if type(columns) is not list:
            columns = [columns]
        for column in columns:
            rule = 'not|in|' + '%' + '%'.join(sorted(list(self.df[column].unique()))) + '%'
            rule += '|->|warn|%Unrecognized categorical string%'
            self.add_rule(column, rule)
    def add_missing_fail(self, columns):
        "Shortcut to add rule 'null|->|fail|%Missing value not permitted/foreseen/encountered in training set%'"
        self._log_info('add_missing_fail method called for column(s) {}; details follow'.format(columns), suppress_num=True)
        rule = 'null|->|fail|%Missing value not permitted/foreseen/encountered in training set%'
        self.add_rule(columns, rule)
    def add_missing_newflagcol(self, columns):
        "Shortcut to add rule 'null|->|newflagcol|_column_%_NULLFLAG%'"
        self._log_info('add_missing_newflagcol method called for column(s) {}; details follow'.format(columns), suppress_num=True)
        self.add_rule(columns, 'null|->|newflagcol|_column_%_NULLFLAG%')
    def add_missing_median(self, columns):
        "Shortcut to add rule 'null|->|median'"
        self._log_info('add_missing_median method called for column(s) {}; details follow'.format(columns), suppress_num=True)
        self.add_rule(columns, 'null|->|median')
    def add_default_numeric_rules(self, columns, nullflag_threshold=None, force_median=True, every_column=False):
        """If every_column=True, will apply to all numeric columns that have no rules yet, 
        and ignore whatever is in the columns variable. If and only if the column has nulls (or if force_median
        is True), a null flag column will be added and nulls replaced by median. Then boundary warnings
        will be set. A null newflagcol will be added only if the proportion of missing values
        is greater than or equal to nullflag_threshold. Setting nullflag_threshold to None
        disables the nullflag entirely."""
        if every_column:
            columns = []
            for col in self.df.columns:
                found = False
                for column, rule in self.rules:
                    if col == column:
                        found = True
                if not found and self.df[col].dtype != 'O':
                    columns.append(col)
        self._log_info('add_default_numeric_rules method called for column(s) {}; details follow'.format(columns), suppress_num=True)
        if type(columns) != list:
            columns = [columns]
        for column in columns:
            if self.df[column].dtype == 'O':
                self._log_error('Column {} is not numeric; add_default_numeric_rules could not be applied'.format(column), suppress_num=True)
            else:
                if nullflag_threshold is not None and (len(self.df[column]) - len(self.df[column].dropna())) / len(self.df[column]) >= nullflag_threshold:
                    self.add_rule(column, 'null|->|newflagcol|_column_%_NULLFLAG%')
                    self.add_rule(column, 'null|->|median')
                elif force_median:
                    self.add_rule(column, 'null|->|median')
                self.add_boundary_warnings(column)
    def add_boolean_and(self, column, column1, column2, create_new=True):
        """Changes every value in column to the logical and between column1 and column2, which are considered True if 
        non-zero, False if zero. Note that null is True. If create_new is True, will create new column.
        """
        self._log_info('add_boolean_and method called for column {}, taking values from columns {} and {}'.format(column, column1, column2), add_num=1)
        if create_new:
            self.add_column(column, 0)
        else:
            self.add_rule(column, 'all|->|value|0')
        self.add_comment(column, 'The preceding rule and the following rule are the result of the previously mentioned '
            'boolean_and between columns {} and {}'.format(column1, column2))
        self.add_rule(column, '[{}]ne|0|&|[{}]ne|0|->|value|1'.format(column1, column2))
    def add_boolean_or(self, column, column1, column2, create_new=True):
        """Changes every value in column to the logical and between column1 and column2, which are considered True if 
        non-zero, False if zero. Note that null is True. If create_new is True, will create new column.
        """
        self._log_info('add_boolean_or method called for column {}, taking values from columns {} and {}'.format(column, column1, column2))
        if create_new:
            self.add_column(column, 0)
        else:
            self.add_rule(column, 'all|->|value|0')
        self.add_comment(column, 'The preceding rule and the following two rules are the result of the previously mentioned '
            'boolean_or between columns {} and {}'.format(column1, column2))
        self.add_rule(column, '[{}]ne|0|->|value|1'.format(column1))
        self.add_rule(column, '[{}]ne|0|->|value|1'.format(column2))
    def add_default_categorical_rules(self, columns):
        """If and only if the column has nulls, value will be COLUMNNAME_NULL.
        If and only if the column contains numbers, those numbers will be replaced with
        COLUMNNAME_NUMBER. Then categorical warning will be set."""
        self._log_info('add_default_categorical_rules method called for column(s) {}; details follow'.format(columns), suppress_num=True)
        if type(columns) != list:
            columns = [columns]
        for column in columns:
            numeric_present = False
            string_present = False
            nans_present = False
            for value in self.df[column]:
                if pd.isnull(value):
                    nans_present = True
                else:
                    try:
                        temp = float(value)
                        numeric_present = True
                    except:
                        string_present = True
            if string_present and numeric_present:
                self._log_error("Column {} skipped from add_default_categorical_rules method because it contains "
                   "both string and numeric values".format(column), suppress_num=True)
            elif numeric_present:
                if nans_present:
                    self.add_rule(column, 'null|->|value|%{}_NULL%'.format(column))
                    self.add_rule(column, 'ne|%{}_NULL%|->|prepend|_column_'.format(column))
                else:
                    self.add_rule(column, 'all|->|prepend|_column_')
            elif nans_present: # string_present only
                self.add_rule(column, 'null|->|value|%{}_NULL%'.format(column))
            self.add_categorical_levels_warning(column)
    def add_comment(self, columns, comment):
        """Shortcut to add_rule comment"""
        if type(columns) is not list:
            columns = [columns]
        for column in columns:
            self.add_rule(column, 'all|->|comment|{}'.format(comment))     
    def add_rule(self, columns, rule):
        """Adds specified rule to specified column"""
        if type(columns) is not list:
            columns = [columns]
        for column in columns:
            try:
                new_rule, freq, flag = self._execute(column, rule)
                self.rules.append([column, new_rule])
                if flag == 'ignored_missing_column':
                    self._log_info('WARNING: Rule not applied because column {} missing: {}'.format(column, new_rule))
                else:
                    self._log_info('Rule applied to column {} : {}'.format(column, new_rule))
                    if flag in ['ignored_training_mask']:
                        pass
                    if (flag is None and freq > 0) or flag in ['warn', 'drop_rows']:
                        pctrows = 100.0 * freq / len(self.df)
                        if pctrows > 0.0 and pctrows < 0.5:
                            pctrows = "<0.5"
                        elif pctrows > 99.5 and pctrows < 100.0:
                            pctrows = ">99.5"
                        elif pctrows == 100.0:
                            pctrows = "exactly 100"
                        else:
                            pctrows = int(round(pctrows, 0))
                        if flag == 'warn':
                            if not self.train:
                                if freq == 0:
                                    self._log_info('(Warning not triggered)')
                                else:
                                    self._log_error('Warning triggered {} times ({}% of rows)'.format(freq, pctrows))
                        else:
                            self._log_info('Rule applied {} times (to {}% of rows)'.format(freq, pctrows))
                    elif (flag is None and freq == 0):
                        self._log_info('NOTE: LEFT-SIDE CONDITION NEVER MET, RULE NEVER APPLIED')
                    elif flag == 'fail':
                        if freq > 0:
                            raise FailureRuleTriggered
                        else:
                            pass
                    elif flag[:11] == 'newflagcol|':
                        colname = flag[11:]
                        self._log_info('New column {} created, flagged {} times'.format(colname, self.df[colname].sum()))
                    elif flag in ['comment', 'drop_column', 'training_mask', 'new_column', 'exec', 'rename']:
                        pass
                    else:
                        raise Exception('Class logic failure, this if-elif block is never supposed to reach this else condition. flag={}, freq={}'.format(flag, freq))
            except FailureRuleTriggered:
                self._log_error('Failure rule triggered {} times. Python exception raised.'.format(freq), suppress_num=True)
                raise FailureRuleTriggered
            except:
                err_type, err_val, sys_tb_object = sys.exc_info()
                self._log_error('Error occurred during following rule application to column[{}]: {}'.format(column, rule), suppress_num=True)
                self._log_error('Error was {}: {}'.format(err_type, err_val), suppress_num=True)
                self._log_error('Column {} left unchanged due to error'.format(column), suppress_num=True)
                print('EXCEPTION TRACE (not logged):')
                traceback.print_exception(err_type, err_val, sys_tb_object)
    def drop_column(self, columns):
        """Drops specified column from the dataframe."""
        if type(columns) is not list:
            columns = [columns]
        for column in columns:
            try:
                assert column in self.df.columns
            except:
                if self.train:
                    raise RuleError('{} is not a column in the dataframe.'.format(column))
                else:
                    self._log_info('{} column is not present to drop'.format(column), suppress_num=True)
            found_ = []
            for col_, rule_ in self.rules:
                if col_ == column:
                    if rule_.find('|->|comment|') == -1:
                        found_.append(rule_)
            if len(found_) > 0:
                s_ = 's'
                if len(found_) == 1:
                    s_ == ''
                self._log_error('WARNING: Column {} is being dropped after being used in the following rule{}: {}'.format(column, s_, ','.join(found_)), add_num=1)
            self.add_rule(column, 'drop_column')
    def add_column(self, column_name, initial_value):
        """Adds column to dataframe"""
        self.add_rule(column_name, 'new_column|{}'.format(initial_value))
    def rename_column(self, column, new_name):
        self.add_rule(column, 'rename|{}'.format(new_name))
    def get_unprocessed_columns(self, sort_columns=False):
        """Returns a list of columns for which there are no rules applied"""
        unprocessed = []
        cols = list(self.df.columns)
        if sort_columns:
            cols = sorted(cols)
        for column in cols:
            found = False
            for col, rule in self.rules:
                if column == col:
                    found = True
                    break
            if not found:
                unprocessed.append(column)
        return unprocessed
    def get_rules(self, columns=None, to_stdout=False):
        """Returns list of rules for specified column(s), or all rules if columns is None.
        If to_stdout is False, prints list instead of returning it"""
        if columns is not None:
            if type(columns) != list:
                columns = [columns]
        else:
            columns = self.df.columns
        to_return = []
        for col_, rule_ in self.rules:
            if col_ in columns:
                to_return.append([col_, rule_])
        if to_stdout:
            for col_, rule_ in to_return:
                print('{}: {}'.format(col_, rule_))
        else:
            return get_rules
        
    def _is_string(self, value):
        if type(value) == str:
            return True
        else:
            try:
                if type(value) == unicode: # for Python 2
                    return True
                else:
                    return False
            except:
                return False
    def _is_float(self, value):
        return np.issubdtype(type(np.array([value])[0]), float)
    def _is_intlike(self, value):
        if (np.issubdtype(type(np.array([value])[0]), int) or
            abs(int(value) - float(value)) < 1e-8):
            return True
        else:
            return False
    
    def _execute(self, column, rule):
        if column not in self.df.columns:
            if rule.startswith('new_column'):
                _ignore, value = rule.split('|')
                if value[0] == value[-1] == "%":
                    value = str(value[1:-1])
                else:
                    try:
                        value = float(value)
                        if abs(value - int(value)) < 1e-8:
                            value = int(value)
                    except:
                        pass
                self.df[column] = value
                self._log_info('New column {} created, with every row given the value {} of {}'.format(column, value, type(value)), add_num=1)
                return rule, 0, 'new_column'
            elif column == '_exec_':
                print('*** EXECUTE THIS COMMAND ***')
                print(rule)
                return rule, 0, 'exec'
            elif not self.train and rule.find('ignore_if_missing') != -1:
                return rule, 0, 'ignored_missing_column'
            else:
                raise MissingColumnError
        else:
            if rule.startswith('new_column'):
                raise RuleError('Column {} already exists'.format(column))           
            returned_rule = rule[:]
            s = self.df[column].copy()
            if rule.startswith('drop_column'):
                self.df = self.df.drop(column, axis=1)
                return rule, 0, 'drop_column'
            if rule.startswith('rename'):
                self.df = self.df.rename(columns={column: rule.split('|')[1]})
                return rule, 0, 'rename'
            else:
                assert rule.find('|->|') != -1, "Malformed rule: missing |->|"
                before, after = rule.split('|->|')
                befores = before.split('|&|')
                afters = after.split('|')
                # make mask from before
                last_mask = None
                returned_flag = None
                for group in befores:
                    if group == 'ignore_if_missing':
                        pass
                    else:
                        items_before_not_check = group.split('|')
                        if items_before_not_check[0][0] == '[':
                            left_col = re.search('\[(.+?)\](.+)', items_before_not_check[0]).group(1)
                            left_s = self.df[left_col].copy()
                            items_before_not_check[0] = re.search('\[(.+?)\](.+)', items_before_not_check[0]).group(2)
                        else:
                            left_s = s.copy()
                        if items_before_not_check[0] == 'not':
                            not_flag = True
                            items = items_before_not_check[1:]
                        else:
                            not_flag = False
                            items = items_before_not_check[:]
                        if items == ['all']:
                            this_mask = left_s.apply(lambda x: True)
                        elif items == ['null']:
                            this_mask = pd.isnull(left_s)
                        elif items == ['float']:
                            this_mask = left_s.apply(self._is_float)
                        elif items == ['intlike']:
                            this_mask = left_s.apply(self._is_intlike)
                        elif items == ['str']:
                            this_mask = left_s.apply(self._is_string)
                        elif items[0] == 'e':
                            if items[1][0] == '%':
                                assert items[1][-1] == '%'
                                this_mask = left_s.apply(lambda x: x == items[1][1:-1])
                            else:
                                this_mask = left_s.apply(lambda x: abs(float(x) - float(items[1])) < 1e-8)
                        elif items[0] == 'ne':
                            if items[1][0] == '%':
                                assert items[1][-1] == '%'
                                this_mask = left_s.apply(lambda x: x != items[1][1:-1])
                            else:
                                this_mask = left_s.apply(lambda x: abs(float(x) - float(items[1])) >= 1e-8)
                        elif items[0] == 'lt':
                            this_mask = left_s.apply(lambda x: x < float(items[1]))
                        elif items[0] == 'lte':
                            this_mask = left_s.apply(lambda x: x <= float(items[1]))
                        elif items[0] == 'gt':
                            this_mask = s.apply(lambda x: x > float(items[1]))
                        elif items[0] == 'gte':
                            this_mask = left_s.apply(lambda x: x >= float(items[1]))
                        elif items[0] == 'in':
                            this_mask = left_s.apply(lambda x: x in(items[1].split('%')))
                        else:
                            raise RuleError('Unrecognized argument {} on left side of rule'.format(before))
                        if not_flag:
                            this_mask = ~(this_mask)
                        if last_mask is None:
                            mask = this_mask
                        else:
                            mask = this_mask & last_mask
                        last_mask = mask
                if afters == ['drop_rows']:
                    self.df = self.df[~mask]
                    new_rule = rule
                    freq = mask.sum()
                    flag = 'drop_rows'
                    return new_rule, freq, flag
                else:  
                    if afters == ['set_training_mask']:
                        if not self.train:
                            flag = 'ignored_training_mask'
                            returned_rule = 'set_training_mask ignored because this is not a training set'
                        else:
                            if self.training_mask is not None:
                                raise RuleError('training mask has already been set; it cannot be changed')
                            elif self.median_mean_mode_called:
                                raise RuleError('Median, mean or mode already called; training_mask can only be set before first time they are called.')
                            else:
                                flag = 'training_mask'
                                returned_rule = rule
                                self.training_mask = mask
                    elif afters[0] == 'median':
                        if self.df[column].dtype == 'O':
                            raise RuleError('Column is not completely numeric')
                        this_training_mask = ~pd.isnull(s)
                        if self.training_mask is not None:
                            this_training_mask = this_training_mask & self.training_mask
                        if len(afters) == 1:
                            if self.train:
                                median = s[this_training_mask].quantile(0.5)
                            else:
                                raise RuleError("Right-side expression 'median' must be followed"
                                                " by a numeric value for non-training set")
                        else:
                            if self.train:
                                self._log_info('Since this is a training set, existing median value ignored', add_num=1)
                                median = s[this_training_mask].quantile(0.5)
                            else:
                                try:
                                    median = float(afters[1])
                                except ValueError:
                                    raise RuleError('Value of median is not a number')
                        if abs(median - int(median)) < 1e-8:
                            median = int(median)
                        s[mask] = median
                        returned_rule += '|{}'.format(median)
                        self.median_mean_mode_called = True
                    elif afters[0] == 'mean':
                        if self.df[column].dtype == 'O':
                            raise RuleError('Column is not completely numeric')
                        this_training_mask = ~pd.isnull(s)
                        if self.training_mask is not None:
                            this_training_mask = this_training_mask & self.training_mask
                        if len(afters) == 1:
                            if self.train:
                                mean = s[this_training_mask].mean()
                            else:
                                raise RuleError("Right-side expression 'mean' must be followed"
                                                " by a numeric value for non-training set")
                        else:
                            if self.train:
                                self._log_info('Since this is a training set, existing mean value ignored', add_num=1)
                                mean = s[this_training_mask].mean()
                            else:
                                try:
                                    mean = float(afters[1])
                                except ValueError:
                                    raise RuleError('Value of mean is not a number')
                        if abs(mean - int(mean)) < 1e-8:
                            mean = int(mean)
                        s[mask] = mean
                        returned_rule += '|{}'.format(mean)
                        self.median_mean_mode_called = True
                    elif afters == 'mode':
                        this_training_mask = ~pd.isnull(s)
                        if self.training_mask is not None:
                            this_training_mask = this_training_mask & self.training_mask
                        if len(afters) == 1:
                            if self.train:
                                mode = stats.mode(s[this_training_mask])[0][0]
                                freq = stats.mode(s[this_training_mask])[1][0]
                                assert s.value_counts()[1] != freq, "Series is multimodal"
                            else:
                                raise RuleError("Right-side expression 'mode' must be followed"
                                                " by a value for non-training set")
                        else:
                            if self.train:
                                self._log_info('Since this is a training set, existing mode value ignored', add_num=1)
                                mode = stats.mode([this_training_mask])[0][0]
                                freq = stats.mode([this_training_mask])[1][0]
                                assert s.value_counts()[1] != freq, "Series is multimodal"
                            else:
                                mode = afters[1]
                        s[mask] = mode
                        if self._is_string(mode):
                            mode = '%'+mode+'%'
                        returned_rule += '|{}'.format(mode)
                        self.median_mean_mode_called = True
                    elif afters == ['null']:
                        s[mask] = np.nan
                    elif afters == ['tonum']:
                        new_s = []
                        for v, m in zip(list(s), mask):
                            if not m:
                                new_s.append(v)
                            else:
                                try:
                                    v2 = float(v)
                                    if abs(v2 - int(v2)) < 1e-8:
                                        v2 = int(v2)
                                    new_s.append(v2)
                                except:
                                    raise RuleError, "Value of {} could not be turned into number".format(v)
                        s = pd.Series(new_s)
                    elif afters == ['toint_if_all_can']:
                        assert befores == ['all'], "all|toint_if_all_can is only allowed syntax"
                        try:
                            for item in s:
                                assert(abs(item-int(item)) < 1e-8)
                        except:
                            raise RuleError, ("There are null or string values, which cannot be turned into integers,"
                                              "or floats which are not equivalent to an integer within +/- 1e-8")
                        self.df[column] = self.df[column].astype(np.int64)
                    elif afters[0] == 'warn':
                        returned_flag = 'warn'
                    elif afters[0] == 'fail':
                        returned_flag = 'fail'
                    elif afters == ['int']:
                        new_s = [int(x) if y else x for x, y in zip(list(s), list(mask))]
                        s = pd.Series(new_s)
                    elif afters[0] == 'prepend':
                        if afters[1] == '_column_':
                            prefix = column+'_'
                        else:    
                            assert afters[1][0] == '%', 'Prepend value must be in %string% format'
                            assert afters[1][-1] == '%', 'Prepend value must be in %string% format'
                            prefix = afters[1][1:-1]
                        new_s = []
                        for value_, mask_ in zip(list(s), list(mask)):
                            if not mask_:
                                new_s.append(value_)
                            else:
                                try:
                                    if abs(int(value_) - value_) < 1e-8:
                                        suffix = str(int(value_))
                                    else:
                                        suffix = str(value_)
                                except ValueError:
                                    suffix = str(value_)
                                except TypeError:
                                    suffix = str(value_)
                                new_s.append(prefix+suffix)
                        s = pd.Series(new_s)
                    elif afters[0] == 'value':
                        if afters[1][0] == '%':
                            assert afters[1][-1] == '%', 'Prepend string must have trailing %'
                            s[mask] = afters[1][1:-1]
                        else:
                            if self._is_intlike(afters[1]):
                                s[mask] = int(afters[1])
                            else:
                                s[mask] = float(afters[1])
                    elif afters[0] == 'comment':
                        mask = np.array([False for x in mask])
                        returned_flag = 'comment'
                    elif afters[0] == 'newflagcol':
                        returned_flag = 'newflagcol'      
                    else:
                        raise RuleError('Unrecognized expression {} on right side of rule'.format(after))
                    if returned_flag != 'newflagcol':
                        self.df[column] = list(s) # list in order to avoid index issues 
                        return returned_rule, sum(mask), returned_flag
                    else:
                        suffix = afters[1]
                        if suffix.startswith('_column_'):
                            prepend = column
                            suffix = suffix[8:]
                        else:
                            prepend = ''
                        assert suffix[0] == '%', 'Suffix must have leading %'
                        assert suffix[-1] == '%', 'Suffix must have trailing %'
                        suffix = suffix[1:-1]
                        new_column = prepend + suffix
                        assert new_column not in self.df.columns, 'DataFrame already has column named {}'.format(new_column)
                        self.df[new_column] = [1 if x else 0 for x in mask]
                        return returned_rule, sum(mask), 'newflagcol|'+new_column
            
    def save_rules(self, filename):
        """Saves rules as json file"""
        if filename[-5:] != '.json':
            filename += '.json'
        with open(filename, 'w+') as f:
            f.write(json.dumps(self.rules))
            
    def import_and_apply_rules(self, filename, case_insensitive=False, json_columns='_all_'):
        """Opens a previously saved json file of rules and applies them.
        Any columns in the json file that do not match a dataframe column name will be ignored,
        and the user will be warned to that effect.
        Any columns in the json file for which there is already a dataframe rule applied
        will be ignored, and the user will be warned to that effect
        If case_insensitive=True, the function will try to find a case-insensitive
        match between column names; there will be an exception if the rules or the
        json files have columns with identical names when lowercased (which is a
        terrible idea to begin with). The existing rule's capitalization will be kept.
        Only specified columns will be applied if a string or list is passed to the
        columns argument"""
        with open(filename, 'r') as f:
            json_rules = json.load(f)
        if json_columns != '_all_':
            if type(json_columns) != list:
                jsonc_olumns = [json_columns]
        else:
            json_columns = set()
        for col_, rule_ in json_rules:
            json_columns.add(col_)
        json_columns = sorted(list(json_columns))
        self._log_info('Rules file {} loaded with option ' 
                       'case_insensitive={} and columns {}'.format(filename, case_insensitive, ', '.join(json_columns)), suppress_num=True)
        def json_to_df_col(json_col):
            if not case_insensitive:
                if json_col in self.df.columns:
                    return json_col
                else:
                    return None
            else:
                found_cols = []
                for df_col in self.df.columns:
                    if json_col.lower() == df_col.lower():
                        found_cols.append(df_col)
                if len(found_cols) == 0:
                    return None
                elif len(found_cols) == 1:
                    return found_cols[0]
                else:
                    raise Exception('There are {} columns in the preprocessor dataframe that match json column {} '
                                    'when case-insensitive: {}'.format(len(found_cols), json_col, ', '.join(found_cols)))
        def print_ci(json_col):
            if not case_insensitive:
                return json_col
            else:
                if json_col in self.df.columns:
                    return json_col
                else:
                    return '{}(case-insensitive match for {})'.format(json_col, json_to_df_col(json_col))
        # first check for json_cols not in dataframe
        cols_to_ignore = set()
        for json_col in json_columns:
            df_col = json_to_df_col(json_col)
            if df_col is None:
                cols_to_ignore.add(json_col)
                self._log_error('Column {} not in dataframe'.format(print_ci(json_col)), suppress_num=True)
        # next check for columns that already have rules in this instance (before open_rules called)
        for json_col in json_columns:
            df_col = json_to_df_col(json_col)
            if df_col is not None:
                found = False
                for col_, rule_ in self.rules:
                    if df_col == col_:
                        found = True
                if found:
                    cols_to_ignore.add(json_col)
                    self._log_error('Column {} already has rule(s); no rules imported'.format(print_ci(json_col)), suppress_num=True)
        # now apply remaining columns
        for json_col, json_rule in json_rules:
            if json_col not in cols_to_ignore and json_col in json_columns:            
                df_col = json_to_df_col(json_col)
                if df_col == 'metadata':
                    self._log_info('Metadata imported: {}'.format(json_rule), suppress_num=True)
                    self.add_metadata(rule)
                else:
                    self._log_info('Rules imported for column {}'.format(print_ci(json_col)), suppress_num=True)
                    self.add_rule(df_col, json_rule)
        unpcols = self.get_unprocessed_columns()
        if len(unpcols) == 0:
            self._log_info('RULE IMPORTING ENDED. There remain no unprocessed columns.', suppress_num=True)
        else:
            if len(unpcols) == 1:
                s1, s2 = '', 's'
            else:
                s1, s2 = 's', ''
            self._log_info('After imports, the following {} instance dataframe column{} remain{} without '
                           'rules:'.format(len(unpcols), s1, s2), suppress_num=True)
            for col in unpcols:
                self._log_info('  {}'.format(col), suppress_num=True)
            self._log_info('RULE IMPORTING ENDED.', suppress_num=True)
                
    def get_log_as_df(self):
        results = []
        with open(self.path_to_log, 'r') as f:
            logtext = f.read()
        for i, line in enumerate(logtext.split('\n')):
            if line.startswith('INFO '):
                type_ = 'INFO'
                line2 = line[5:]
            elif line.startswith('ERROR '):
                type_ = 'ERROR'
                line2 = line[6:]
            else:
                type_ = 'OTHER' # this should, in theory, never happen
                line2 = line
            if re.search('\{\d+?\} ', line2):
                rulenum = int(re.search('\{(\d+?)\} ', line2).group(1))
                line3 = re.sub('\{(\d+?)\} ', '', line2)
            else:
                rulenum = np.nan
                line3 = line2
            results.append([i, rulenum, type_, line3])
        return pd.DataFrame(results, columns=['log_line_num', 'rule_number', 'type', 'logged_line'])

    def get_rules_as_df(self):
        """Returns a dataframe of rules, with numbers, indicating
        what columns do not have rules"""
        results = []
        for i, (col, rule) in enumerate(self.rules):
            results.append([i, col, rule])
        for col in self.get_unprocessed_columns():
            results.append([np.nan, col, '*** NO RULES ASSIGNED TO THIS COLUMN ***'])
        return pd.DataFrame(results, columns=['rule_number', 'column', 'rule'])

    def print_rules_help(self):
        """Prints the formatting guidelines for rules"""
        text = """RULES FORMAT

----------

1.1 Introduction

Examples:

lt|0|->|value|0    If a value is less than 0, replace it with the value 0
null|->|median     If a value is null, replace it with the (non-null) column median

Rules are made up of "words", divided by pipes(|).

There is a left-side expression and a right-side expression, separated from each other
by the word "->".

In the column, wherever the left-side expression evaluates as true, the right-side
expression will be carried out.

There can be multiple left-side expressions, separated by the word "&" (Logical AND).
For example:
gte|0|&|lt|1|->|value|2   If a value is in the range [0, 1), replace with 2.

Logical NOT is created by prepending the word "not".

Strings are delineated by the % sign, e.g. "one" is rendered %one%.

Lists of strings have % at beginning, end, and between each string, e.g.
%one%two%three%

Integer-to-float comparisons are always done at 1e-8 precision, e.g.
2.000000001 == 2 but
2.00000001 != 2

1.2 Drop

The only expressions without a left and right side are:
drop_column     Column is dropped. If column is not present
                in non-training set, that fact is logged but no error is raised.
new_column|value       Creates a column with that name, with uniform value given.
                       If value is in %value% form, it is coerced to string,
                       otherwise it will be numeric if possible
exec|statement    logs and executes an external statement. The dataframe must be called self.df in all such statements
rename|new_name    self-explanatory

----------

NOTE: you can make the left-side expression match with a column other than
the specified column to be modified by prepending the name of the column
in square brackets, e.g.
[credit_score]gt|700|->|value|%good_credit%

2.1 Valid single-word left-side expressions are:

all
null
float (does not include floatlike strings, e.g. "2.5")
intlike (integers, floats without decimals to 1e-8 precision, and string representations
         of integers or floats without decimals, e.g. "2", "2.0")
str
ignore_if_missing    this only makes sense as a second left side expression with |&|; it means
                     that if the column specified does not exist in a non-training set, the rule 
                     will be ignored with a warning instead of failing

2.2 Valid two-word (separated by a pipe) left-side expressions, shown with "n" representing
a number and %w% representing a string, are:

e|n
e|%w%
ne|n    Note: this is a shortcut for not|e|n
ne|%w%    See note above
lt|n
lte|n
gt|n
gte|n
in|%w1%w2%w3%    check set membership of string in string list, in this case {w1, w2, w3}.
                 There is no equivalent for numeric values; normally combinations of 
                 gt, lt, not and e would work, possibly with several rules

----------

3.1 Valid right-side expressions having to do with central values are:

median
mean
mode

median and mean must be calculated from numeric values only; mode can be numeric or string,
but there must be one uniquely most frequent number or string (i.e. a unimodal distribution).

If the rule is called as, e.g. null|->|median, the median of (non-null, regardless of what the
left-side expression is) values, it will be saved as, e.g. null|->|median|15 if the median 
is 15. Similarly, the values of mean and mode are added to the right-side expression after it
is called.

When train=True upon Preprocessor class instantiation, calling null|->|median and 
null|->|median|15 give the same result: any rightmost value (in this case 15), if present, is
ignored, the median is (re)calculated, and saved with the rule, e.g. null|->|median|20.

When train=False, calling null|->|median results in an error (it theoretically should not
be able to be called, as a rule would never be saved that way unless the user 'hacked' the
internals of the class). Calling null|->|median| replaces every null value with 15. It is the
equivalent of calling null|->|value|15, null|->|mean|15 or null|mode|15, i.e. the word 'median'
is preserved only in order to preserve for the record the method by which the value was
calculated.

3.2 Valid single-word right-side expressions are:

null    Replace with null
tonum    If the value is a string, e.g. "2", replace it with its number equivalent (integer, 
         if possible). Fails if there is even one string value, e.g. "A", that cannot be
         converted to a number.
toint_if_all_can    Converts every value in the column to an integer, whether they be float
                    (e.g. 2.0) or string (e.g. "2" or "2.0"). Fails unless left-side
                    expression is "all", and fails if there is even one value that cannot
                    be converted to an integer, e.g. "2.1" or 2.1. The purpose of this
                    rigt-side expression is to change the dataframe's dtype to np.int64,
                    but only if it is possible to do so without error and without losing
                    information after decimal places.
set_training_mask    Can only be set once per Preprocessor object, and must be set before any
                  median, mean or mode right-hand expressions are called. This mask will let
                  only those rows with value True be used to calculate median, mean or mode.
drop_rows    Drop rows where left-side expression is True from the dataset
ignore    Mark column to be ignored; nothing will be done with it in training, and if it is
          present in non-training, it will be ignored, and if it is absent, its absence will
          be ignored.

3.3 Valid two-word right-side expressions are:

value|n    Replaces with number n
value|%w%    Replaces with string w
warn|%message%    Does not change any values if left-side expression is true, but
                  warns the user, displaying the indicated message. A reasonable use of
                  this would be to warn the user if a test row has a variable value outside
                  the range of this variable in the training set.
fail|%message%    Raises an exception for the entire operation if the left-side criterion is 
                  true, displaying the indicated message. A reasonable use of this would
                  be to fail the operation if the weight column is null. Note that the
                  operation will always automatically fail if a column in the rules is missing
                  in the dataframe.
prepend|_column_    The second word is literally "_column_"; it is a placeholder for the name 
                    of the column in question. A reasonable use of this would be to turn
                    categorical variables encoded as integers into strings, so that models
                    do not think they are numeric/ordinal. For example, if a dataset has
                    a variable "territory" with the possible values 1, 2, and 3 (or even "1",
                    "2" and "3"), these values would be replaced with "territory_1",
                    "territory_2" and "territory_3". This expression will fail if the dataframe
                    has a column literally named "_column_", which on the face of it is
                    highly unlikely.
prepend|%w%    Same as above, except instead of the colum name, the indicted string is
               prepended to all values.
comment|anything   Just what it sounds like. Columns with comments are considered to have rules.

3.4 Valid right-side expressions WHICH CREATE A NEW COLUMN are:

newflagcol|_column_%w%    Creates a new column containing the value 1 where the left-side expression
or                        is true, 0 where it is false. The new column's name will be the string w,
newflagcol|%w%            prepended with the evaluated column if _column_ is before it.
                          A reasonable use of this for a column named "credit_score" would be 
                          null|->|newflagcol|_column_%_NULL%, which would create a new column called 
                          "credit_score_NULL" with 1 where credit_score is null, 0 otherwise.
                          This would likely be followed by null|->|median.

3.5 Currently invalid right-side expressions that may be implemented in future are:

formula|%f%    Some sort of implementation that could create a calculated column based on the
               value of one or more columns, e.g. 
               all|->|formula|%losscost=gross_incurred/claim_frequency% where losscost is the 
               name of a new column, and gross_incurred and claim_frequency are current columns.
"""
        print(text)
