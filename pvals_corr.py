from scipy.stats import pearsonr

def calculate_pvalues(df):
    df = df._get_numeric_data()
    print(df.shape)
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df.dropna(subset=[r,c])
            pvalues[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)
    return pvalues
