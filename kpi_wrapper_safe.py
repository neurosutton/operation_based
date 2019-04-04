#import sqlite3 as db
import pandas as pd
import numpy as np

# For organization
from pathlib import Path  # Use Path to add the OS-specific file separators
import os.path
from datetime import datetime, timedelta
import glob
import gc

# For SQL
import pyodbc

# For visualizations
import matplotlib.pyplot as plt
from pandas.plotting import table
import six
import seaborn as sns
sns.set(palette='winter',context='talk',style='whitegrid')

home_dir =  $path_to_dir
# In order to only run once a week (but to allow that frequency for a data refresh), get the Monday of the data pull week for the file label
today = datetime.strptime(str(datetime.now().strftime('%Y%m%d')),'%Y%m%d')
week = (today - timedelta(days=today.weekday())).strftime('%Y%m%d')
print('Working on KPIs for {}'.format(week))
dt = (today - timedelta(days=today.weekday())).strftime('%Y%m')
mon = today.strftime('%B')[0:3].lower()
data_dir = str(Path(home_dir, 'data', dt))

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

class grab_metric_data():
    def __init__(self):
        pass
    
    def __str__(self):
        return('Run sql code for established KPIs using pyodbc.')
        
    def __repr__(self):
        pass
    
    def connect(self):
        """Handshake method for networked server (Caboodle)"""
        conn_str = ('SERVER={server};'     +
                    'DATABASE={database};' +
                    #'UID={usrname};' +
                    #'PWD={get_pass}' +
                    'Trusted_Connection=yes;')
        config = dict(server = $serverName,
              port = 1433,
              database = 'STAR_PRD')
        self.conn = pyodbc.connect(r'Driver={ODBC Driver 13 for SQL Server};' + conn_str.format(**config))


    def get_data(self,statement):
        """Use the agreed upon SQL queries to pull data from Caboodle"""
        self.outname = str(dt) + '_' + statement.split('\\')[-1].split('.')[0]
        print('Checking {}'.format(self.outname))
        if not os.path.isfile(str(Path(data_dir, self.outname + '.csv'))):
            #Debugging driver
            #csr = self.conn.cursor()
            #print(list(csr.getTypeInfo(93)))
            # Make sure the connection has been established
            try:
                self.connect()
            except:
                print('SQL server connection issue')
    
            df = pd.DataFrame()
            with open(str(Path(home_dir,statement))) as sql:
                print('opened file: {}'.format(statement))
                # Often times, errors that read [ODBC driver blah-blah-blah] are due to not putting "SET NOCOUNT ON" at the top of the sql query
                # Virtually any error 
                try:
                    for chunk in pd.read_sql(sql.read(), self.conn, chunksize = 2000):
                        tmp=chunk
                        print(tmp)
                        df = df.append(tmp)
                        print(df.shape)
                except:
                # Can try without chunking for debugging purposes
                    print('PROBLEM WITH {}'.format(sql.read()))
                    c = self.conn.cursor()
                    c.execute(sql.read())
                    c.fetchone()
            # Due to left join resulting in nulls, odbc has trouble with the calculation that runs independently within SQL. For the simplest solution, calculate the percentage here.
            if 'referral' in self.outname.lower():
                df['reject_count'] = np.where(df['reject_count'].isnull(),0,df['reject_count'])
                #df['pct_accepted'] = np.where((df['reject_count'].isnull() & df['accept_count'] > 5),100,np.where(df['reject_count'].isnull(), np.nan, (df['accept_count']/(df['accept_count']+df['reject_count']))*100))
                df['pct_accepted'] = np.where((df['reject_count'].isnull() & df['accept_count'] > 0),100, np.where((df['reject_count'].isnull() & df['accept_count'] == 0), np.nan, (df['accept_count']/(df['accept_count']+df['reject_count']))*100))
                df['pct_rejected'] = np.where((df['reject_count'].isnull() & df['accept_count'] > 5),0,(df['reject_count']/(df['accept_count']+df['reject_count']))*100)
            
            if 'attendance' in self.outname.lower():
                df = df[df['AppointmentStatus'].isin(['Completed','Canceled','No Show'])]
                
            # Write the data set to disk for reproducibility 
            df.to_csv(str(Path(data_dir, self.outname + '.csv')),index=False)
            self.conn.close()
            gc.collect()
            
    def prep_for_graphing(self,statement):
        """Though the content varies, some of the cleaning steps are the same across KPIs. This method seeks to unify the preprocessing as much as possible"""
        
        if not self.outname:
            self.outname = str(week) + '_' + statement.split('\\')[-1].split('.')[0]
        self.inName = str(Path(data_dir, self.outname + '.csv'))
        
        self.df = pd.DataFrame(pd.read_csv(self.inName))
        self.df.columns = map(str.lower, self.df.columns)
        self.mntCol = [x for x in self.df.columns.tolist() if '_month' in x][0] 
        self.yrCol = [x for x in self.df.columns.tolist() if '_yr' in x][0] 
        self.dept = [x for x in self.df.columns.tolist() if 'spec' in x][0] 
        
        # Standardize month names
        self.df[self.mntCol] = self.df[self.mntCol].str[:3]
        
        if (self.mntCol in self.df.columns.tolist() and self.yrCol in self.df.columns.tolist()):
            self.df['timeframe'] = (self.df[self.mntCol] + self.df[self.yrCol].map(str))
            
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        self.df.loc[:,self.mntCol] = pd.Categorical(self.df[self.mntCol], categories=months, ordered=True)
        self.df.sort_values(by=self.mntCol, inplace=True)
        self.df.to_csv(str(Path(data_dir, self.outname + '.csv')),index=False)
        
         
    def visit_volume_charts(self, depts=None):
        """For current month, display the number of appointments attended, canceled, and no showed"""
        sns.set(palette='tab20', context='talk',style = 'whitegrid')
        #print(self.df.columns.tolist())
        if not depts:
            depts = set(self.df[self.dept])
        else:
            depts = [depts]
        for dept in depts:
            if dept:
                # Individual division option 
                tmpBig = self.df[self.df[self.dept] == dept].reset_index()
                dept = dept.replace('/','-')
                print(dept)
            else:
                # Department-wide, roll-up option
                tmpBig = self.df.reset_index()
            
            for t in set(tmpBig.loc[tmpBig['appt_month'].str.lower() == mon,'timeframe']):
                fname = os.path.normpath(Path(data_dir,('visits_' + t + "_" + dept + ".png")))
                if not os.path.isfile(fname):               
                    fig, ax = plt.subplots()
                    tbl = tmpBig.loc[tmpBig['timeframe']==t,['departmentname','appointmentstatus','count','timeframe']].reset_index().drop(['index'],axis=1)
                    tbl.columns = ['Location','Status','Count','Time Frame']
                    render_mpl_table(tbl.sort_values(['Location','Status']))
                    
                    plt.savefig(fname,dpi=600, quality=100, transparent=True)                              


##################### Previously required by managers: Keep for the likely case that these pie charts will be requested again #############################################################################
#            for t in set(tmpBig['timeframe']):
#                fname = os.path.normpath(Path(data_dir,('visitPie_' + t + "_" + dept + ".png")))
#                if not os.path.isfile(fname):
#                    tmp = tmpBig[tmpBig['timeframe']==t]
#                    labels = []
#                    sizes = []
#                    for x in tmp.index.values:
#                        labels.append(tmp.loc[x,'appointmentstatus'])
#                        sizes.append(tmp.loc[x,'count'])
#            
#                    fig1,ax1 = plt.subplots() # Initialize the chart object
#                    expl = [.2].extend(np.repeat(0,len(tmp.index.values)).tolist())
#                    palette_dict = {'Completed':'tab:blue' , 'Canceled':'xkcd:lightblue'  ,'No Show':'tab:orange'  }
#                    palette = [palette_dict[x] for x in labels]
#                    wedges,texts,autotexts = ax1.pie(sizes, labels = labels, colors=palette, pctdistance = 0.65, autopct = '%1.1f%%',explode = (expl), startangle=0)
#                    for at in autotexts:
#                        at.set_fontsize(11)
#                        at.set_color('white')
#                        at.set_weight('extra bold')
#                    for tt in texts:
#                        tt.set_fontsize(13)
#                            #tt.set_color('white')
#                    for w in wedges:
#                        w.set_linewidth(.1)        
#                    ax1.axis('equal')  # Makes sure that the ratios are properly distributed.
#                    ax1.set_title(dept + '\nVisit Volume - '+ t + '\n',fontsize=13)
#                    plt.tight_layout(pad = 3.0)
#                    #plt.show()
#                    plt.savefig(fname, dpi=600, quality=100, transparent=True, frameon=True)
#                    gc.collect()
    
    def timelapse_charts(self, statement):
        sns.set(palette='winter',context='talk',style='white')
        stmt = statement.split('\\')[-1].split('.')[0]
        prop_dict = {'CommercialPayorTrending_title':'Number of commerical payers', 
                     'CommercialPayorTrending_y_axis':'Commercial Payers', 
                     'CommercialPayorTrending_y':'count', 
                     'Referrals_title':'% Referrals Rejected', 
                     'Referrals_y_axis':'% rejected',
                     'Referrals_y' : 'pct_rejected',
                     'CloseTheLoop_title':'% loops closed', 
                     'CloseTheLoop_y_axis':'% closed',
                     'CloseTheLoop_y' : 'pct_closed',
                     'CloseTheLoop_target' : 80,
                     'VisitVolumebyMonth_title':'Number of completed visits', 
                     'VisitVolumebyMonth_y_axis':'Visits',
                     'VisitVolumebyMonth_y':'count',
                     'apptAttendance_title':'No Shows',
                     'apptAttendance_y_axis': 'Number of no shows',
                     'apptAttendance_y': 'count',
                     }
        if not stmt + '_title' in prop_dict:
            print('Details for graphing {} not yet defined. Please advise.'.format(stmt))
        else:
            for dept in set(self.df[self.dept]):
                fname = os.path.normpath(Path(data_dir,(dept + "_" + stmt + '.png')))
                if not os.path.isfile(fname):
                    tmp=self.df.loc[self.df[self.dept]==dept,:]
                    if 'attendance' in self.outname.lower():
                        tmp=tmp.loc[tmp['appointmentstatus']=='No Show',:]
                        tmp.sort_values(['appointmentstatus'], inplace=True)
                    
                    # Try to order the data for lineplots
                    tmp.sort_values(by=[self.mntCol,self.yrCol], inplace=True)                   
                    
                    # Build flexible color definitions with DH colors
                    years = set(tmp[self.yrCol])
                    colors = ['xkcd:lightblue', 'tab:blue','tab:orange']
                    palette = dict(zip(years,colors))
    
                    fig, ax = plt.subplots(figsize=(12,8))
                    ax.set_title(dept + '\n' + prop_dict[stmt+'_title']+'\n')
                            
                    ax = sns.pointplot(x=self.mntCol, y=prop_dict[stmt+'_y'], 
                                      hue=tmp[self.yrCol], 
                                      palette=palette, 
                                      legend='full', marker="o", data=tmp) # Add  estimator=None, if using lineplot
                    ax.set_ylabel(prop_dict[stmt+'_y_axis']+'\n')
                    ax.set_xlabel('')
                    if stmt+'_target' in prop_dict:
                        # If there is am established target condition, then plot it.
                        xmin = 1 
                        xmax = 12
                        plt.plot([xmin, xmax],[prop_dict[stmt+'_target'],prop_dict[stmt+'_target']], 'r', linewidth=3,dashes=[20,5,10,5])
    
                    sns.despine()
                    ax.legend(fancybox=True, bbox_to_anchor=(1.15,1), borderpad=.3, loc='upper right')
                    
                    plt.tight_layout(pad = 3.0)
                    plt.savefig(fname, dpi = 400, quality = 90, transparent = True, frameon=True)
                    plt.show()
                    
        gc.collect()
        
        
def render_mpl_table(data, col_width = 3, row_height = 0.625, font_size=14,
                             header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                             bbox=[0, 0, 1, 1], header_columns=0,
                             ax=None, **kwargs):
            """Customizable table saving function"""
            if ax is None:
                size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height]) # Set the overall figure size. Yields tuple.
                fig, ax = plt.subplots(figsize=size)
                ax.axis('off')
        
            mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, colWidths=[4,2,2,2], colLoc='center', **kwargs)
        
            mpl_table.auto_set_font_size(False)
            mpl_table.set_fontsize(font_size)
            mpl_table.auto_set_column_width(-1)
        
            for k, cell in six.iteritems(mpl_table._cells):
                cell.set_edgecolor(edge_color)                
                if k[0] == 0 or k[1] < header_columns:
                    cell.set_text_props(weight='bold', color='w')
                    cell.set_facecolor(header_color)
                else:
                    cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

            return ax
        
test = grab_metric_data()
# Find all the sql queries that will be run consistently
sql_statements = glob.glob(str(Path(home_dir,'finalized_code','*')))
#print(sql_statements)
for stmt in sql_statements:
    test.get_data(stmt)
    test.prep_for_graphing(stmt)
    print(stmt.lower())
    if stmt.lower().find('attendance') > 0:
        test.visit_volume_charts()
    test.timelapse_charts(stmt)
