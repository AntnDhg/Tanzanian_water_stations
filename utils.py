from pandas import option_context, DataFrame, merge, Series, concat, get_dummies
from collections import Counter, OrderedDict
from datetime import datetime
from numpy import NaN

# Function that displays all columns and rows when you use commands like df.head() and df_tail().
season_mapper = {
    1:"winter",
    2:"spring",
    3:"summer",
    4:"autumn"
}

def display_all(df):
    with option_context('display.max_rows', 1000):
        with option_context('display.max_columns', 1000):
            display(df)
            

def cumulatively_categorise_number(column,n_categories=10,return_categories_list=False):
    """
    This function operates on a column of a pandas DataFrame.
    It is used to deal with categorical values with a high cardinality.
    """
    #Initialise an empty list for our new minimised categories
    categories_list=[]
    #Initialise a variable to calculate the sum of frequencies
    s=0
    #Create a counter dictionary of the form unique_value: frequency
    counts=Counter(column)

    #Loop through the category name and its corresponding frequency after sorting the categories by descending order of frequency
    for i,j in counts.most_common():
        s+=1
        #Add the category to the list of categories
        categories_list.append(i)
        #Check if the global sum has reached the threshold value, if so break the loop
        if s>=n_categories-1:
          break
    #Append the category Other to the list
    categories_list.append('Other')

    #Replace all instances not in our new categories by Other  
    new_column=column.apply(lambda x: x if x in categories_list else 'Other')
    new_column.fillna('Other', inplace = True)
    #Return transformed column and unique values if return_categories=True
    if(return_categories_list):
        return new_column,categories_list
    #Return only the transformed column if return_categories=False
    else:
        return new_column
    
def extract_date_features(df: DataFrame, fieldname_date: str):
    """
    Clearly, the exact date has a high cardinality. 
    To deal with this, some new columns are created to extract possibly useful features from the data.
    While the exact date is likely not very important, the season in which it was recorded probably is important!
    """
    df["month_recorded"] = df[fieldname_date].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").month)
    df["year_recorded"] = df[fieldname_date].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").year)
    df["season_recorded"] = df["month_recorded"].apply(lambda x: x % 12 // 3 + 1).map(season_mapper)
    df["years_operational"] = df["year_recorded"] - df["construction_year"]
    df.loc[df["years_operational"] < 0, "years_operational"] = 0
    df = df.drop(columns = [fieldname_date])
    return df

def fix_locations(X_train, X_test):
    cols = ['longitude', 'latitude', 'gps_height', 'population']
    ### wherever the longitude is 0, set the latitude to 0 (the latitude has values close to 0, but not 0)
    for i in [X_train, X_test]:
        i.loc[i.longitude == 0, 'latitude'] = 0
        
    for z in cols:
        ### set all 0/1 values to NaN so that they can be easily imputed
        for i in [X_train, X_test]:
            i[z].replace(0., NaN, inplace = True)
            i[z].replace(1., NaN, inplace = True)
            
        # use the subvillage, district_code and basin subsequently to fill any remaining NA values
        for j in ['subvillage','ward','lga', 'district_code', 'basin']:
        
            X_train['mean'] = X_train.groupby([j])[z].transform('mean')
            X_train[z] = X_train[z].fillna(X_train['mean'])
            o = X_train.groupby([j])[z].mean()
            fill = merge(X_test,DataFrame(o), left_on=[j], right_index=True, how='left').iloc[:,-1]
            X_test[z] = X_test[z].fillna(fill)
        
        # if there are still NA values, use the column mean
        X_train[z] = X_train[z].fillna(X_train[z].mean())
        X_test[z] = X_test[z].fillna(X_train[z].mean())
        del X_train['mean']
    return X_train, X_test

def fix_construction_year(X_train, X_test):
    cols = ['construction_year']
    
    for z in cols:
        ### set all 0/1 values to NaN so that they can be easily imputed
        for i in [X_train, X_test]:
            i[z].replace(0., NaN, inplace = True)
            i[z].replace(1., NaN, inplace = True)
            
        # use the subvillage, district_code and basin subsequently to fill any remaining NA values
        for j in ['subvillage','ward','lga', 'district_code', 'basin']:
        
            X_train['mean'] = round(X_train.groupby([j])[z].transform('mean'),0)
            X_train[z] = X_train[z].fillna(X_train['mean'])
            o = round(X_train.groupby([j])[z].mean(),0)
            fill = merge(X_test,DataFrame(o), left_on=[j], right_index=True, how='left').iloc[:,-1]
            X_test[z] = X_test[z].fillna(fill)
        
        # if there are still NA values, use the column mean
        X_train[z] = X_train[z].fillna(round(X_train[z].mean(), 0))
        X_test[z] = X_test[z].fillna(round(X_train[z].mean(), 0))
        del X_train['mean']
    return X_train, X_test

def fix_funder_installer(X_train, X_test):
    cols = ['funder', 'installer']

    for z in cols:
        ### set all 0/1 values to NaN so that they can be easily imputed
        for i in [X_train, X_test]:
            i[z].replace("0", NaN, inplace = True)
            
        # use the subvillage, district_code and basin subsequently to fill any remaining NA values
        for j in ['district_code', 'basin']:
            X_train['mode'] = X_train.groupby([j])[z].agg(lambda x: Series.mode(x)[0])
            X_train[z] = X_train[z].fillna(X_train['mode'])
            o = X_train.groupby([j])[z].agg(lambda x: Series.mode(x)[0])
            fill = merge(X_test,DataFrame(o), left_on=[j], right_index=True, how='left').iloc[:,-1]
            X_test[z] = X_test[z].fillna(fill)
        
        # if there are still NA values, use the column mean
        X_train[z] = X_train[z].fillna(X_train[z].agg(lambda x: Series.mode(x)[0]))
        X_test[z] = X_test[z].fillna(X_train[z].agg(lambda x: Series.mode(x)[0]))
        del X_train['mode']
    return X_train, X_test

def dummies(X_train, X_test):
    columns_train = list(X_train.dtypes[X_train.dtypes == 'object'].index)
    columns_test = list(X_test.dtypes[X_test.dtypes == 'object'].index)
    assert columns_train == columns_test, "train and test have different categorical columns!!"
    
    for column in columns_train:
        # X_train[column].fillna('NULL', inplace = True)
        good_cols = [column+'_'+i for i in X_train[column].unique() if i in X_test[column].unique()]
        X_train =concat((X_train, get_dummies(X_train[column], prefix = column)[good_cols]), axis = 1)
        X_test = concat((X_test, get_dummies(X_test[column], prefix = column)[good_cols]), axis = 1)
        del X_train[column]
        del X_test[column]
    return X_train, X_test