# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:22:30 2018
@author: bsutton
"""
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
from pathlib import Path
import os
import joblib
import glob
from IPython.display import display
from collections import Counter
import sqlite3
import gc
import sys
gc.enable() #During debugging
#gc.set_debug(gc.DEBUG_STATS)
#######################################################################################################
#  Parameters for which columns to keep and work with as certain data types.

# Columns that look like integers, but need to be seen as binary
bin_features = ['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8', 'eng1', 'bin9', 'bin10']

intCodes = ['int1', 'int2', 'int3','int4','eng2', 'int5', 'eng3']

# Columns that may look like integers, but need to be treated as categorical
catCodes = ['cat1', 'cat2', 'cat3', 'cat4',  'cat5', 'cat6',  'cat7', 'cat8', 'int4', 'cat9']

status_dict = {'No Show':2,'Completed':0, 'Canceled':1}

excl = [list_of_features_that_would_identify_labels]

### Switch these options to toggle model building and evaluation of new data etc. ###
raw_test_suite='no'
#########################################################################################################

class raw_data:
    def __init__(self, data_file, model_name='models'):
        (self.data_dir, self.filename) = os.path.split(os.path.normpath(data_file))
        self.home_dir, data_dir_name = os.path.split(self.data_dir)  # Goes back one more level
        self.model_dir = str(Path(self.home_dir,model_name))
        self.model_name = self.model_dir.split('\\')[-1]
        self.suffix = self.model_name.split('_')[-1]
        if self.suffix == self.model_name:
            self.suffix=''
        # Columns that need to be excluded b/c they give away the appointment status
        excl.extend([('idx_'+ self.suffix),('chosen_'+self.suffix)]) #  Not sure if this criteria should be in or not.
        self.excl = excl
        if self.suffix == 'id7':
            self.excl.remove('chosen_id7')
        self.basename = self.filename.split('.')[0]
        self.db_file = str(Path(home_dir,'historical_db.sqlite'))
        self.table='time_frame_of_query'
        pkl_file = Path(self.data_dir,'clean_' + self.model_name +  '_' + self.basename + '.pkl')
        print('Searching for {}'.format(pkl_file))
        if os.path.isfile(pkl_file):
            """The easiest of all solutions - the initial steps have already been done and saved."""
            self.pkl_check = 'complete'
        else:
            self.pkl_check = 'incomplete'
            self.upload_data()  # Create the structures and databases needed for further analysis
            
    def __repr__(self):
        return ("\n".join([self.data_dir, self.model_dir, self.db_file, self.pkl_check]))
    
    def __str__(self):
        return """The module, raw_data, is looking for a dirty, predictable data file and model directory (optional)
    e.g., dat = glob.glob(str(Path(home_dir,'data','historical_goLiveTo1018.csv')))
    fileA=raw_data(dat[0],self.model_name)
    """

    def upload_data(self):
        """Need to setup the database and/or select the data to include."""
        db_check = os.path.isfile(self.db_file)
        conn = sqlite3.connect(self.db_file)
        historical_touch = str(Path(self.home_dir,'historical_touch.txt'))
        chunksize = 10**4
        if not os.path.isdir(str(Path(self.model_dir))):
            os.makedirs(str(Path(self.model_dir)))
        if (db_check==True) & (os.path.isfile(historical_touch)) & (('historical') in self.basename) & (self.pkl_check == 'complete'):
            # If the DB already exists, do not append the table with duplicate entries and eat up memory. Skip to the end of this method.
            self.df = joblib.load(str(Path(self.data_dir, self.basename + '_dedup.pkl')))
        elif (db_check==True) & os.path.isfile(str(Path(self.data_dir, self.basename+ '_dedup.pkl'))):
            self.df = joblib.load(str(Path(self.data_dir, self.basename + '_dedup.pkl')))
        elif ~os.path.isfile(str(Path(self.data_dir, self.basename+ '_dedup.pkl'))):
            """The db has already been initialized and the data just needs to be pulled. This method is more memory efficient than re-reading a df every time."""
            print('Building df from db')
            if ('historical') in self.basename:
                tbl = self.table
            else:
                 tbl = self.basename
            crsr = conn.cursor()
            tbl_check = crsr.execute("""SELECT name FROM sqlite_master WHERE type='table' AND name='{}'""".format(tbl)).fetchall()
            self.df = pd.DataFrame()
            # TODO Check if including WHERE clause will automatically detect if there are more "full" rows of data (depending on the importance of a particular factor)
            if len(tbl_check) == 1:
                for i, chunk in enumerate(pd.read_sql("""SELECT * from '{}'""".format(tbl), conn, chunksize=chunksize)):
                    print('Chunk {} - DF size = {} + {}'.format(i,self.df.shape,chunk.shape))
                    self.df = self.df.append(chunk)
                    self.df.drop_duplicates(subset=['identifier1'],inplace=True) # Just a double-check
                    gc.collect()
                dump_file = str(Path(self.data_dir, self.basename + '_dedup.pkl'))
                joblib.dump(self.df,dump_file)
                conn.close()
            else:
                print('Reading in data...')
                data_file = str(Path(self.data_dir,self.filename))
                for i,chunk in enumerate(pd.read_csv(data_file,chunksize=chunksize)):
                    print('Loop {}'.format(i))
                    self.df = pd.DataFrame(chunk)
                    print('Adding calculated columns')
                    self.df.sort_values('identifier2', ascending = False, inplace=True) # Completed to ensure that the most recent of the possible data points is kept.
                    self.df.drop_duplicates(subset='identifier1',inplace=True)
                    self.df = self.df[~self.df['identifier3'].isin(['Arrived','Left without seen'])]
                    self.df['apptBin'] = np.where(np.less_equal(self.df['int4'], 900),'early_morning',
                           (np.where(np.logical_and(np.greater_equal(self.df['int4'], 900),np.less(self.df['int4'],1200)),'before_noon',(np.where(np.logical_and(np.greater_equal(self.df['int4'], 1200),np.less(self.df['int4'],1500)),'afternoon','late_day')))))
                    print('Determining disposition')                   
                    tmp_df = pd.DataFrame(self.df.groupby(['identifier4','identifier3']).agg({'identifier3':['count'],'int4':['mean'],'apptBin':[lambda x:x.value_counts().index[0]]}).round(1).unstack().to_records())
                    tmp_df.columns = ['identifier4','total_canceled','total_completed','total_noShows','apptTime_canceled','apptTime_completed','apptTime_noShows','apptBin_canceled','apptBin_completed','apptBin_noShows']
                    # Fill the NaNs, so the percentage can be calculated
                    tmp_df.loc[:,['total_canceled','total_completed','total_noShows']] = tmp_df.loc[:,['total_canceled','total_completed','total_noShows']].fillna(value=0, axis=1)
                    def ns_avg(row):
                        import math
                        return math.ceil(row['total_noShows']/(row['total_noShows']+row['total_completed']+row['total_canceled']))
                    tmp_df['percent_noShows'] = tmp_df.apply(ns_avg, axis=1)
                    #Figure out how closely the appts are scheduled
                    tmp_df['schedule_gap'] = (tmp_df['apptTime_completed'] - tmp_df['apptTime_noShows']) # negative happens when no shows are later in the day
                    # If the person's avg gap is greater than two hours, keep the number for consideration
                    tmp_df['schedule_gap'] = np.where(np.absolute(tmp_df['schedule_gap']) > 200, tmp_df['schedule_gap'],np.nan)
                    #tmp_df['preferredApptTime'] = np.where(((tmp_df['percent_noShows'] > .4) & (tmp_df['total_completed'] + tmp_df['total_noShows'] > 2)), np.where((tmp_df['schedule_gap'] > 0.0),'afternoon_person','morning_person'),np.nan) # Secondary algorithm possiblity
                    tmp_df['preferredApptTime'] = np.where(np.logical_and(tmp_df['apptBin_completed'].isnull(), ~tmp_df['apptBin_noShows'].isnull()),'not_'+tmp_df['apptBin_noShows'],
                          np.where(np.greater_equal(tmp_df['total_completed'], tmp_df['total_noShows']), tmp_df['apptBin_completed'],
                                   np.where(np.logical_and(~tmp_df['apptBin_noShows'].isnull(), np.greater_equal(tmp_df['total_noShows'],2)), 'not_'+tmp_df['apptBin_noShows'], 'Not enough data')))
                    self.df = self.df.merge(tmp_df, on='identifier4', how='left')
                    
                    try:
                        if ('historical') in self.basename:
                            if i == 0 :
                                print('Creating database')
                                crsr = conn.cursor()
                                sql_idx = ('idx_'+self.suffix)
                                sql_stmt = 'CREATE TABLE "time_frame_of_query" ( "{}" INTEGER, "int5" INTEGER, "shortKey" INTEGER, "cat9" TEXT, "cat8" REAL, "demo1" TEXT, "demo2" TEXT, "cat5" TEXT, "demo3" TEXT, "cat2" TEXT, "demo4" TEXT, "demo5" TEXT, "demo6" TEXT, "id6" INTEGER, "eng1" INTEGER, "bin9" INTEGER, "rowOrder" INTEGER, "identifier4" INTEGER, "int10" INTEGER, "identifier3" TEXT, "bin4" TEXT, "identifier1" INTEGER, "identifier2" INTEGER, "cat6" TEXT, "cat7" TEXT, "int4" INTEGER, "cat11" TEXT, "bin1" INTEGER, "int1" INTEGER, "int2" INTEGER, "int3" INTEGER, "cat1" INTEGER, "bin10" INTEGER, "bin5" INTEGER, "eng3" REAL, "eng2" REAL, "identifier4.1" REAL, "cat3" REAL, "cat4" REAL, "apptBin" TEXT, "total_canceled" INTEGER, "total_completed" INTEGER, "total_noShows" INTEGER, "apptTime_canceled" REAL, "apptTime_completed" REAL, "apptTime_noShows" REAL, "apptBin_canceled" TEXT, "apptBin_completed" TEXT, "apptBin_noShows" TEXT, "percent_noShows" REAL, "schedule_gap" INTEGER, "preferredApptTime" TEXT, PRIMARY KEY (identifier1));'.format(sql_idx)
                                crsr.execute(sql_stmt)
                                conn.commit()
                                enc_list = [] # Build the list of encounters so that they can be eliminated, rather than throw a constraint error
                            
                            ##### Update the DF to abide by the contraint #####
                            self.df = self.df.loc[~self.df['identifier1'].isin(enc_list),:]
                            enc_list.extend(self.df['identifier1'].values.tolist())
                            print(self.df.shape)
                            
                            self.df.to_sql(name=self.table, con=conn, index=False, if_exists='append')
                            conn.commit()
                            (self.data_dir, self.filename) = os.path.split(os.path.abspath(data_file))
                            self.basename = self.filename.split('.')[0]
                        else:
                            print('Adding to the database')
                            print(self.df.shape)
                            self.df.to_sql(name=self.basename,con=conn, index = False, if_exists='append')
                            conn.commit()
                        Path(historical_touch).touch()
                    except:
                        print('SQL DB initialization failed.')

            gc.collect() # Attempt at memory management
        conn.close() # Back-up close, if the cmd fails

    def clean_df(self):
        """Perform the same cleaning on every self.df for the model"""
        if ('historical' in self.basename):
            tbl=self.table
        else:
            tbl=self.basename
        dedup_file  = str(Path(self.data_dir, self.basename + '_dedup.pkl'))
        chunksize = 10**4
        if self.pkl_check == 'complete' and os.path.isfile(self.db_file) and os.path.isfile(dedup_file):
            print("Loading previously deduped file")
            self.df = joblib.load(dedup_file)
        elif os.path.isfile(self.db_file):
            conn = sqlite3.connect(self.db_file)
            try:
                print("Finding data...")
                self.df = pd.DataFrame()
                j = 0
                for chunk in pd.read_sql("""SELECT * FROM '{}'""".format(tbl),conn, chunksize=chunksize): # should fail, if the table doesn"t exist
                    j+=1
                    print("Chunk " + str(j))
                    self.df = self.df.append(chunk)
                print("Selected available, partially cleaned data. (1/1)")
                print(self.df.shape)
            except:
                if not ('historical') in self.basename:
                    print('Error pulling down data from DB (within clean_df method)')
                    print('Tried to find {}, but table [{}] did not exist'.format(self.basename,tbl))
                    sys.exit()
                else:
                    conn = sqlite3.connect(self.db_file)
                    self.df = pd.DataFrame()
                    print('Fetching data from the original data (1/4)')
                    for tmp in pd.read_sql("SELECT * FROM {} ;".format(self.table),conn, chunksize=chunksize):
                        print('Sorting {} (2/4)'.format(tmp.shape))
                        tmp.sort_values('identifier2', ascending = False, inplace=True) # Completed to ensure that the most recent of the possible data points is kept.
                        tmp.drop_duplicates(subset=['identifier1'],inplace=True) # On the marginal chance that two dups ended up in different chunks on upload
                        print('Cleaning df (3/4) - - ->')
                        print('Began with {}'.format(tmp.shape))
                        tmp = tmp.loc[tmp['prop1'] != 1] # Same-day appointments cannot be predicted and historically have a high rate of attendance. Must be discarded for model fairness.
                        print('W/o same day appointments...\n{}'.format(tmp.shape))
                        tmp = tmp.dropna(axis = 1, how='all')  # in case the supposed feature is empty in Epic
                        tmp = tmp.dropna(how='all') # Any blank rows
                        idx = tmp.loc[tmp['identifier3'] != 'Left without seen'].index #Super rare, not useful behavior
                        tmp = tmp.loc[idx,:]
                        # Change over the actual time admitted to a string, 'yes', for simplicity
                        tmp['bin4'] = np.where(tmp['bin4'] == "no",'no','yes')
                        tmp['cat8'].astype(int, errors='ignore',copy=False)
                        if self.df.shape[0] == 0:
                            self.df = tmp
                        else:
                            self.df = self.df.append(tmp)
                        print('Combined df now {}'.format(self.df.shape))
                        gc.collect()
                    print('Writing cleaned data to DB (4/4)')
                    try:
                        joblib.dump(self.df, dedup_file)
                        self.df.to_sql(tbl, conn, index = True, if_exists = 'fail')
                    except:
                        print('{} not written'.format(tbl))
            conn.close()
        self.df.sort_values('identifier1',inplace=True)
        self.df.reset_index(inplace=True) # Attempt to make the index make sense for matching later.
        keep_cols = self.df.columns.difference(self.excl)
        self.X = self.df[keep_cols]  # The clean dataframe that can be split and trained.
        # Separate out the key reportable elements for a weekly update to the managers
        mrn_df = self.df[['identifier1', 'bin4','prob1', 'identifier2','hsp1']]
        mrn_df.to_csv(Path(self.data_dir,'mrn_clean_' + self.filename))
        #joblib.dump(mrn_df,Path(self.data_dir, model_dir,'mrn_clean_' + self.filename + '.pkl'))
        print('Cleaned data represented as {}\n'.format(self.X.shape))
        gc.collect()


    def create_labels(self):
        """Isolate and encode the labels for the models"""
        from sklearn.preprocessing import LabelEncoder
        self.Y = self.df['identifier3']
        self.Y = self.Y.map({'Arrived':'Completed','Completed':'Completed','No Show':'No Show','Canceled':'Canceled'}) # Don't penalize someone who showed up, but had an incorrect prep OR was not updated by office staff as completed
        self.y = self.Y.map(status_dict)
        print('Encoded target classes = {}\n'.format(status_dict))

        # Try to automate the continued processing of the data prior to make_groups by calling "external" method
        if not os.path.isfile(str(Path(self.model_dir,'clean_touch.txt'))):
            self.X = xfrm_features(self.X,str(Path(self.model_dir,'clean_models_' + self.filename.split('.')[0] + '.pkl')))

    def make_groups(self, percent=.2):
        if not os.path.isfile(str(Path(self.model_dir, 'X_train.pkl'))):
            conn = sqlite3.connect(self.db_file)
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=percent)
            train_dict = {'X_train':self.X_train, 'X_test':self.X_test, 'y_train':self.y_train, 'y_test':self.y_test}
            
            
############ Incorporate the index and the labels into self.df to update the DB with all demographics as well as test labels and groupings #################
            #Potentially frivolous double-check for the order to match on the index
            # This same sort was completed prior to creating self.X
            self.df.sort_values('identifier1',inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            
            # For compatibility with new uploads, change the index column to a better name
            if ('idx_'+self.suffix) in self.df.columns.tolist():
                self.df.drop(columns=('idx_'+self.suffix)) # Drop duplicates
                self.df['idx_'+self.suffix] = self.df.index  # Define anindex will match with the test/train indicies
            elif 'index' in self.df.columns.tolist():
                self.df.rename(columns={'index':('idx_'+self.suffix)},inplace=True)
            elif 'level_0' in self.df.columns.tolist():
                self.df.rename(columns={'level_0':('idx_'+self.suffix)},inplace=True)
            cols = [ x for x in ['level_0','index'] if x in self.df.columns]
            for col in cols:
                self.df.drop(columns=[col],inplace=True)
            self.df = self.df.loc[:,~self.df.columns.duplicated()]
                
            # For validation, print X_test to viewable file and compare the indices with self.df to make sure the correct people are being identified as the test cohort
#            le_dict = joblib.load(str(Path(self.model_dir,'le_dict.pkl')))
#            self.X_test2 = pd.DataFrame()
#            for dict_key, dict_value in le_dict.items():
#                inv_le_dict = {v: k for k,v in le_dict[dict_key].items()}
#                self.X_test2[dict_key] = self.X_test[dict_key].map(inv_le_dict)
#            self.df2 = self.df.merge(self.X_test2, how='left', left_index=True, right_index=True, suffixes=('_main','_test'))
#            self.df2.to_csv(str(Path(self.home_dir,'validation_set','index_validation_label_encoded.csv')))
    
            # Update the test/train categories for future pulls and verifications
            col = ('chosen_'+self.suffix) # specific to the various models

            self.df[col] = np.where(self.df[('idx_'+self.suffix)].isin(set(self.X_test.index)),'test','train')
            print('Reworking SQLite DB\nTest labels.')

            self.df.to_sql(self.table, conn, if_exists = 'replace', chunksize=10**4)

# For evaluation purposes later, build the test table + specific features now.
            ytest_file = str(Path(self.data_dir,'clean_'+ self.model_name + '_TEST_historical_db.csv'))
            if not os.path.isfile(ytest_file):
                # Go find the data that match the y_test set, but has all the pertinent identifiers
                #tmp = pd.read_sql("""SELECT * FROM test_cases""",conn)
                tmp = pd.DataFrame()
                for chunk in pd.read_sql("""SELECT * FROM {} WHERE {} = 'test'""".format(self.table,('chosen_'+self.suffix)), conn, columns = ['idx_'+self.suffix, 'bin4', 'identifier1', 'prob1', 'identifier2', 'hsp1'], chunksize=10**5):
                    tmp1 = chunk
                    tmp = tmp.append(tmp1)
                tmp.loc[:,['identifier1', 'bin4', 'prob1', 'identifier2', 'hsp1','identifier3']]
                tmp.to_csv(ytest_file, index=False)

            # Save the chosens for easy access later
            conn.close()
            joblib.dump(self.df,str(Path(self.data_dir,self.basename+'.pkl')))
            self.df.to_csv(Path(self.data_dir,self.basename+'_'+ col + '.csv'),index=False)
            if not os.path.isdir(self.model_dir):
                # Quick check, in case the db has been constructed and various files exist. Will make the necessary intervening directories
                os.makedirs(self.model_dir)
                
            for k,v in train_dict.items():
                joblib.dump(v, Path(self.model_dir, k + '.pkl'))
            print('Data should be ready for modeling')
            gc.collect()

    def clean_wrapper(self,historical=None):
        """Quick wrapper that includes all initial cleaning and saving methods, but does not split the chosens"""
        self.clean_df()
        self.create_labels()
        if self.X.isnull().values.any():
            self.X = xfrm_features(self.X, str(Path(self.model_dir, self.filename.split('.')[0] + '.pkl')))
        if historical:
            self.make_groups()

# Separate the xfrm_function, so it is callable at the modelling level.        
def xfrm_features(X, model_filepath=None):
    """In lieu of running a pipeline with a columnTransformer (the other option), fill nan's and perform specific cleaning methods. Scaling and imputation are not included, as these steps should only be applied to test and new case data."""
    
    # Check whether the data set has already undergone transformation
    if model_filepath:
        if '.' in model_filepath:
            model_dir, filename = os.path.split(model_filepath)
            model_name = os.path.abspath(model_dir).split(os.sep)[-1]
            basename = filename.split('.')[0]
        else:
            _, model_name = os.path.split(model_filepath)
            model_dir = model_filepath
            basename = 'X_test' #default value
        clean = str(Path(model_dir,'clean_touch.txt'))    
        if os.path.isfile(clean):
                os.remove(clean) # If this method has been entered, but doesn't complete, want the next run to try to re-run this method.
        ii = [i+1 for i,st in enumerate(model_dir.split(os.sep)) if 'predictiveScheduling' in st]
        home_dir = os.sep.join(os.path.abspath(model_dir).split(os.sep)[:ii[0]])
        data_dir = str(Path(home_dir,'data'))
        
    else:
        model_dir = ''
        print('File name not provided.\nVariables will be transformed, but not saved.')
        
    if not isinstance(X, pd.DataFrame):
        try:
            X = joblib.load(model_filepath)
        except:
            print('Usage is xfrm_features(X, filepath)\nTrying to see if I can load {}'.format(X))
            try:
                X = joblib.load(X)    
            except:
                print('Nope. Data cannot be located or loaded. Is the file pickled?')
                exit
        
    print('Transforming features')
    X[['eng2','eng4']].fillna(0,inplace=True)
    # Second checkpoint, so that the processing will be different from categorical numbers: Change dates to type int
    X.fillna({x: 0 for x in intCodes},inplace=True)
    X[intCodes].astype('int',copy=False)

    # Make the dtype explicit for "floats" that are actually categorical
    X[catCodes].astype('category',copy=False)

    # Expand the definition of eng1 to try to capture those with temporary housing, who are actually eng1
    X.loc[:,'eng1'] = np.where(X.loc[:,'bin9']==0, 1, X.loc[:,'eng1'])

    # Find the patients who have been admitted within the past 10 days
    bin_admitted = np.where(X.loc[:,'bin4']=='no',0,1)
    X.loc[:,'bin4'] = bin_admitted  # overwrite the values

    # Traditional filling for missing values in each feature type
    numerical_features = list(col for col in X.columns if X[col].dtypes in ('int','float') and col not in bin_features and col not in catCodes)
    X[numerical_features].fillna(np.nan, inplace=True)
    numerical_features.extend(intCodes)
    ctgr_features = list(col for col in X.columns if col not in numerical_features)  # Don't add bin_features to the exclusion or else the nan's are not replaced and cause the models to fail.
    print(ctgr_features)
    X.loc[:,ctgr_features] = X.loc[:, ctgr_features ].astype('str')
    X.loc[:,ctgr_features].fillna('missing',inplace=True)

    # To keep the number of one hot features down...
    # https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-66041f734512

    for f in ctgr_features:
        print('Detecting rare codes in {}'.format(f))
        X.loc[X[f].value_counts()[X[f]].values < 20,f] = 'rare'

################### Comment out for OHE??? #############################
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le_dict = {}
    for col,feat in enumerate(ctgr_features):
        X.loc[:,feat] = le.fit_transform(X.loc[:,feat])
        X.loc[:,feat] = X.loc[:,feat].astype('category')
        label_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        le_dict[feat] = label_dict

        # At the last loop, dump the dictionary
        if col == len(ctgr_features)-1:
            print(X.shape)
            # Create a loadable record of the clean data
            if model_dir:
                joblib.dump(X, str(Path(data_dir, basename + '.pkl')))
                if not os.path.isdir(model_dir):
                    os.mkdir(model_dir)
                Path(clean).touch(exist_ok=True)
                joblib.dump(le_dict, str(Path(model_dir, 'le_dict.pkl')))
#######################################################################
            
    gc.collect()
    return X
    
### Test case ###
home_dir = $path_to_dir
if raw_test_suite == 'yes':
    model_name = 'models'
    dat = glob.glob(str(Path(home_dir,'data','$filename.csv')))
    fileA=raw_data(dat[0],model_dir)
    fileA.clean_df()
    fileA.create_labels()
    fileA.X = xfrm_features(fileA.X)
    fileA.make_groups()