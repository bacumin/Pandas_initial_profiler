# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 08:53:04 2017

@author: e84605
"""

def dtype_profiler(df, subset=None, check_nan=True, cols=[], rec_types=True, 
                   nan_count=True, unique_count=True, top_x=0, vc_nonan=False, sorter=False, 
                   file_save=False, hist=0, explanatory_dict=None):
    '''
    This function profiles each column in a dataframe, and prints formatted results
    to the screen.
    
    Returns: nothing. Prints a profile of each column in a pandas df.
    
    Parameters:
        - df: pandas df to profile
        - subset: random subset that will be used, with replacement
        - check_nan: if false, will not look for nans in data type profiling. Setting
            false will speed up the program
        - cols: empty list = profile all. Feed subset of columns as a list of strings
            to limit the function to some columns.
        - rec_types : if true, will give the variable types of each record, and print
             bar graphs to represent distribution.
        - nan_count : if true and nans present in col, returns nan proportion
            as a bar graph
        - unique_count: if true, returns proportion unique vals as bar graph
        - top_x : if >0, prints top x values of a feature.
        - vc_nonan : if true, prints the proportion of column (excluding nans)
            that have this value next to the bar graph. Has no effect if top_x=0
        - sorter: if true and top_x>0, returns not the top x highest values but the
            first x alphabetically/numerically sorted values
        - file_save: if true, saves "dtype_output.txt" which pipes output
        - hist: if >0, gives a histogram with 90th/10th percentile, and x middle categories
    '''
    
    #THIS ASSUMES YOU COPY THE PDTOOLS DIRECTORY INTO THE FOLDER WHERE YOU ARE RUNNING THE PROGRAM
    
    from Pandas_initial_profiler.profiler import barchartline
    #import barchartline
    import pandas as pd
    import numpy as np
    import math 
    
    if file_save:
        import sys
        orig_stdout=sys.stdout
        f=open("./dtype_output.txt", "w")
        sys.stdout=f
    
    def dtype_column_profiler(df, col, subset_c=None, check_nan=True):
        """
        This function profiles columns in a pandas dataframe to determine a more specific
        datatype than the numpy profiler.
        
        Returns: tuple (pandas datatype, specific datatype, hasnan, unique)
            - pandas datatype can be int64, object, float64 or bool
            - specific datatype can be 'true int', 'int-like', 'true float', 'float-like',
                'true bool', 'bool-like', 'possibly bool-like', 'true string'
            - hasnan is 0 or 1
            - unique is an int = to number of unique values
        
        Parameters:
            - subset: random subset of data to use. Subsets with replacement. Subset
                recommended since function has to loop through data to find nans
            - df: pandas dataframe
            - check_nan : False to cancel looping through to find nans hasnan will be 0
        
        Datatype keys:
            true-int    -> pandas detects an int, has more than 2 unique values
            int-like    -> factor or float, but all values are integers
            true bool   -> all values are bool
            bool-like   -> factor, float or int, values are 1 and 0 or 1.0 and 0.0
            possibly bool-like -> factor, float or int. 2 unique values
            true string -> factor, strings. Catches mixed types as well
            true float
            float-like  -> factor with float values
            
            All of these are evaluated after NaNs are stripped
                                                    
        """
        pd_dt=str(df[col].dtype)
    
        
        #subset=max(subset_c, len(df))   
        if(subset_c is None):
            subset=len(df)
        else:
            subset=subset_c
        
        #sample with replacement
        try:
            rand_smp = df[[col]].sample(subset, replace=False).iloc[:,0].tolist() 
        except ValueError:
            rand_smp = df[[col]].sample(subset, replace=True).iloc[:,0].tolist() 
        
        #check for nans
        nan_check=0
        if check_nan:
            for e in rand_smp:
                import math
                try:
                    if math.isnan(float(e)):
                        nan_check=1
                        break
                except ValueError:
                    pass
    
        #int profiler - can be bool, possible bool (2 unique values) or int. No nans
        if(pd_dt=='int64'):
            
            #bool
            myset=set(rand_smp) 
            
            if((myset == set([1, 0]))):
                return (pd_dt, 'bool-like', nan_check, len(myset), 0)
            
            #possible bool
            elif((len(myset)==2)):
                return (pd_dt, 'possibly bool-like', nan_check, len(myset), 0)
            
            return (pd_dt, 'true int', nan_check, len(myset), 0)
            
        elif(pd_dt=='bool'):
            return (pd_dt, 'true bool', nan_check, 2, 0)
        
        #String profiler - can be int, float, bool or string, all with/without nan 
        elif(pd_dt == 'object'):

            #int, float and bool with nan
            if nan_check==1 :    
                import math
                rand_smp_noNaN=[]
                
                #build list of non nan elements
                for e in rand_smp:
                    try:
                        if(not math.isnan(e)):
                            rand_smp_noNaN+=[e]
                    except TypeError:
                        rand_smp_noNaN+=[e]
                
                num_nan=len(rand_smp)-len(rand_smp_noNaN)
                
                #bool with nan check
                myset=set(rand_smp_noNaN)
                if((myset == set(['1', '0'])) | (myset == set(['1.0', '0.0']))):
                    return (pd_dt, 'bool-like', nan_check, len(myset), num_nan)
                
                #int with nan check
                try:
                    rand_smp_int_noNaN=[int(e) for e in rand_smp_noNaN]
                    if(rand_smp_int_noNaN == rand_smp_noNaN):
                      return (pd_dt, 'int-like', nan_check, len(myset), num_nan)
                except ValueError:
                    pass
                
                #float with nan
                try:
                    rand_smp_float_noNaN=[float(e) for e in rand_smp_noNaN]
                    if(rand_smp_float_noNaN == rand_smp_noNaN):
                      return (pd_dt, 'float-like', nan_check, len(myset), num_nan)
                except ValueError:
                    pass                
                
                
                #possible bool with nan check. If 2 or 1 elements
                if(len(set(rand_smp_noNaN)) in [1,2] ):
                    return (pd_dt, 'possibly bool-like', nan_check, len(myset), num_nan)
                
                #true string with nan
                return (pd_dt, 'true string', nan_check, len(myset), num_nan)
                
            #int, float, bool nonan
            
            #bool detection
            myset=set(rand_smp)
            if((myset == set(['1', '0'])) | (myset == set(['1.0', '0.0']))):
                return (pd_dt, 'bool-like', nan_check, len(myset))
            elif (len(myset)==2):
                return (pd_dt, 'possibly bool-like',nan_check, len(myset), 0)
           
            #int detection
            try:        
                rand_smp_int = [str(int(e)) for e in rand_smp]
                if(rand_smp_int == rand_smp): 
                    return (pd_dt,'int-like',nan_check, len(myset), 0)        
            except ValueError:      
                pass
            
            #float detection
            try:
                rand_smp_float = [str(float(e)) for e in rand_smp]
                if(rand_smp_float == rand_smp): 
                    return (pd_dt,'float-like',nan_check, len(myset), 0)
            except ValueError as e:
                pass
                

            #true string nonan
            return (pd_dt, 'true string', nan_check, len(myset), 0)
            
        #float profiler - can be int, bool, bool-like or float, all with/without nan
        else:
            
            #int and bool with nan
            if nan_check==1 :    
                import math
                rand_smp_noNaN=[e for e in rand_smp if not math.isnan(e)]
                num_nan=len(rand_smp)-len(rand_smp_noNaN)
                
                #bool with nan check
                myset=set(rand_smp_noNaN)
                if(myset == set([1.0, 0.0])):
                    return (pd_dt, 'bool-like', nan_check, len(myset), num_nan)
                
                #possible bool with nan check. If 2 or 1 elements
                if(len(set(rand_smp_noNaN)) in [1,2]):
                    return (pd_dt, 'possibly bool-like', nan_check, len(myset), num_nan)
                
                #int with nan check
                rand_smp_int_noNaN=[float(int(e)) for e in rand_smp_noNaN]
                if(rand_smp_int_noNaN == rand_smp_noNaN):
                  return (pd_dt, 'int-like', nan_check, len(myset), num_nan)
                    
                
                #float with nan
                return (pd_dt, 'true float', nan_check, len(myset), num_nan)
                
            #bool no nan
            myset=set(rand_smp)
            if((myset == set([1.0, 0.0]))):
                return (pd_dt, 'bool-like', nan_check, len(myset), 0)
            
            #possible bool no nan
            elif((len(myset)==2)):
                return (pd_dt, 'possibly bool-like', nan_check, len(myset), 0) 
            
            #int no nan
            try:        
                rand_smp_int = [float(int(e)) for e in rand_smp]
                if(rand_smp_int == rand_smp): 
                    return (pd_dt, 'int-like', nan_check, len(myset), 0)       
            except ValueError:      
                pass 
            
            #float without nan    
            return (pd_dt, 'true float', nan_check, len(myset), 0)

    
    def record_profiler(col):
        """
        Takes as input a pandas series, returns a dict of dtype:value.
        
        
        Possible dict keys are:
            - bool-like string (T, True, F, False etc...)
            - bool
            - int
            - int-like string
            - int-like float
            - float
            - float-like string
            - nan
            - string
        """
        types={}
        for t in ['bool-like string', 'bool', 'int', 'int-like float', 'int-like string',
        'float', 'float-like string', 'string', 'nan']:
            types[t]=0
        
        for e in col:
            
            #bool. Note numbers are not counted as booleans
            if  (isinstance(e, bool)):
                types['bool']+=1
        
            #int
            elif (isinstance(e, int)):
                types['int']+=1
            
            #float and float-like int
            elif(isinstance(e, float)):
                if(math.isnan(e)):
                    types['nan']+=1
                elif(float(int(e))==e):
                    types['int-like float']+=1
                else:
                    types['float']+=1
            
            
            #string, float-like string, int-like string, bool-like string
            elif(isinstance(e, str)):
                next=0 #next variable to avoid double assigning as int-like and string
                
                try:
                    if(e.lower in ['t', 'true', 'f', 'false', 'yes', 'y', 'no', 'n']):
                        types['bool-like string']+=1
                        next=1

                    if(str(int(e))==e):
                        types['int-like string']+=1
                        next=1
                    elif(str(float(e))==e):
                        types['float like string']+=1
                        next=1
                except ValueError:
                    types['string']+=1
                    next=1
                
                if (next==0):
                    types['string']+=1
            
            else:
                try:
                    if(math.isnan(e)):
                        types['nan']+=1
                except:
                    pass
                
        return types
            
    def distribute_weights(inputs, total=30):
        inp_sum=sum(inputs)
        weight_remaining=total
        normal_w_sum=0
        sol={}
        
        for e in inputs:
            #if under th, set weight of 1
            if ((e < (inp_sum/total)) & (e>0)):
                sol[e]=1
                weight_remaining-=1
            #if over th, set weight of 0 for now, increment remainder sum
            else:
                sol[e]=0
                normal_w_sum+=e

        #remainder sums
        for e in inputs:
            if sol[e]==0:
                sol[e]=round(e*(weight_remaining/normal_w_sum))
        
        return sol
    
    def value_counts_hist(df, col, n_bins, pareto=True):
        """Prints quick and dirty horizontal histograms of
        value counts. If Pareto == True, returns in descending
        numerical order, otherwise alphabetical. """
        
        #get bins
        bins=[df[col].min()-0.00001, df[col].quantile(0.1)]        
        for i in range(1,n_bins+1):
            bins+=[df[col].quantile(0.1)+i*(df[col].quantile(0.9)-df[col].quantile(0.1))/(n_bins+1)]

        bins+=[df[col].quantile(0.9), df[col].max()+0.00001]
        
        #histogram index
        try:
            idx=["< "+str(round(bins[1],2))]
            idx+=[str(round(bins[i],3))+" - "+str(round(bins[i+1],3)) for i in range(1, len(bins)-2)]
            idx+=["> "+str(round(bins[len(bins)-2],3))]
            x = pd.DataFrame(df[col].value_counts(bins=bins, sort=False))
            x.index=idx
        except ValueError:
            print("histogram unnecessary - see value counts")
            return
            
        #pareto
        if not pareto:
            x.sort_index(inplace=True)
        
        #rendering
        x.columns = ['cnt']
        x['pct'] = 0.0
        x['bar'] = ''
        for idx, row in x.iterrows():
            pct = row.cnt*100.0/x.cnt.sum()
            reps = 30*row.cnt//x.cnt.sum()
            x.loc[idx, 'bar'] = '#' * reps + '_' * (30-reps)
            x.loc[idx, 'pct'] = round(pct,1)
        print(x)
        print('')
        return df
    
    
    
    
    
    '''
    Main function call. For each column, profiles the default datatype. Performs
    additional operations depending on options, as specified in description above.
    '''        
    i=1
    
    if cols:
        columns=cols
    else:
        columns=df.columns
    
    
    #table print
    for col in columns:
        profile=dtype_column_profiler(df, col, subset, check_nan)
        
        nanflag='none'
        if(profile[2]==1):
            nanflag="HAS NAN"

        #header
        print("COL ", i, "OF ", len(df.columns), ": ",col)        
        
        #nested data dict
        if (explanatory_dict != None):
            if(col in explanatory_dict):
                for it in explanatory_dict[col].items():
                    print("{}:{}".format(k,v))
        
        
        #basic types
        print("BASICS: numpy dtype: ", profile[0], (8-len(profile[0]))*" ",
              "| dtype: ", profile[1], (12-len(profile[1]))*' ', 
              "| flag: ", nanflag, ' '*(8-len(nanflag)),
              "| non null: ", len(df)-sum(df[col].isnull()) ,
                                  "\n")
        
        #unique proportion
        if(unique_count):
            print("UNIQUE:"," "*10, barchartline.bar_chart_line(profile[3], len(df)))
        
        #nan proportion
        if((nan_count) & (nanflag=='HAS NAN')):
            print("NAN COUNT: "," "*6, barchartline.bar_chart_line(sum(df[col].isnull()), len(df)))
        
        
        #variable types breakdown
        if((profile[0] in ['object', 'float64']) and (rec_types)):
            print('DTYPES:')
            dic=record_profiler(df[col])
            weights=distribute_weights(dic.values())
            spacer=0
            
            #dict for dtypes
            char_dt={'bool-like string':'b', 'bool':'B', 
            'int':'I','int-like float':'i', 'int-like string':'i',
            'float':'F', 'float-like string':'f', 
            'string':'S', 
            'nan':'n'}            
            
            for key, value in dic.items():
                if value > 0:
                    print(" ", key,(16-len(key)+spacer)*" " +
                         char_dt[key]*weights[value] + 
                         "."*(30-weights[value]-spacer) +
                         " "*2)
                    spacer+=weights[value]
                    
            print()
        
        
        #print top x values
        if(top_x>0):
            print('VALUE COUNTS :')
            
            if sorter==False:
                v_c=df[col].value_counts()
            else:
                v_c=df[col].values_counts(ascending=True)
                
            if(vc_nonan):
                nonan_count=len(df)-sum(df[col].isnull())
                
            for k in range(0,top_x):
                try:
                    if(vc_nonan):
                        propor_nonan=round(v_c[v_c.index[k]]/nonan_count,4)
                    else:
                        propor_nonan=''
                    print(v_c.index[k],":", (30-len(str(v_c.index[k])))*" ",
                          barchartline.bar_chart_line(value=v_c[v_c.index[k]], 
                                                      maximum=len(df))
                          , " |", propor_nonan)
                
                #if top_x is overshooting - top_x=3 but 2 unique values
                except IndexError:
                    break
            print()
        
        #histogram
        if(hist>0 and profile[0] in ['float64', 'int64']):
            print("DESCRIPTION:")
            print(df[col].describe())
            print()
            print("HIST:")
            value_counts_hist(df, col, hist)
        
        #footer
        print("-"*120)
        i+=1
        
        
        
    if file_save:
        sys.stdout=orig_stdout
        f.close()
        
if __name__=='__main__':
  #code testing
  import warnings
  warnings.filterwarnings('ignore')
  import barchartline
  import pandas as pd
  import numpy as np
  import math 
  data=np.array([np.arange(10)]*3).T
  index=range(10)
  column=['a', 'b', 'c']
  df=pd.DataFrame(data, index=index, columns=column)
  df['a']=range(10)
  df['a'].iloc[4]=6
  df['b']='0'
  df['b'].iloc[6:8]='1'
  df['b'].iloc[1]=np.nan
  df['c']=range(10)
  df['c'].iloc[1]=1.6
  df['c'].iloc[5]=np.nan
  print(df)
  dtype_profiler(df, hist=2, top_x=3, vc_nonan=True)
