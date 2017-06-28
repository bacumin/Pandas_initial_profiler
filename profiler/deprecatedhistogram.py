"""should use barline function"""

def value_counts_hist(df, col, pareto=True):
    """Prints quick and dirty horizontal histograms of
    value counts. If Pareto == True, returns in descending
    numerical order, otherwise alphabetical. """
    x = pd.DataFrame(df[col].value_counts())
    if not pareto:
        x.sort_index(inplace=True)
    x.columns = ['cnt']
    x['pct'] = 0.0
    x['bar'] = ''
    for idx, row in x.iterrows():
        pct = row.cnt*100.0/x.cnt.sum()
        reps = 30*row.cnt//x.cnt.sum()
        x.loc[idx, 'bar'] = '▮' * reps + ' ' * (30-reps)
        x.loc[idx, 'pct'] = round(pct,1)
    print(col)
    print(x)
    print('')
    return df
