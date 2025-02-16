import pandas as pd
import numpy as np
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.utils import shuffle

def preprocess(df):
    """
### Steps:
1. **Feature Selection & Cleaning**  
   - Drops irrelevant columns (`housing`, `loan`, `previous`, `marital`).  
   - Converts 'y' (yes/no) to binary (1/0).  

2. **Binning & Transformation**  
   - Bins 'campaign' into ordinal categories (`Low`, `Medium`, `High`).  
   - Groups 'day_of_week' into (`Early Week`, `Midweek`, `End of Week`).  
   - Applies log transformation to 'duration' to reduce skewness.  

### Returns:
    - Processed DataFrame ready for modeling, including transformed and binned features.
    """

    # **Step 1: Drop Unnecessary Columns**
    drop_columns = ['housing', 'loan', 'previous', 'marital']
    df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')
    

    if 'day_of_week' in df.columns:
        day_bins = {
            'mon': 'Early Week', 'tue': 'Early Week',
            'wed': 'Midweek', 'thu': 'Midweek',
            'fri': 'End of Week'
        }
        df['day_of_week'] = df['day_of_week'].map(day_bins)
    
    # bin campaign into categories for easier interpretation
    if 'campaign' in df.columns:
        bins = [0, 2, 5, float('inf')]  
        labels = ['Low (1-2)', 'Medium (3-5)', 'High (>5)']
        df['campaign'] = pd.cut(df['campaign'], bins=bins, labels=labels, right=True)


    # Convert 'y' (yes/no) to 1/0**
    if 'y' in df.columns:
        df['y'] = df['y'].map({'yes': 1, 'no': 0})

    # log transformation to handle skewness 
    if 'duration' in df.columns:
        df['duration'] = np.log1p(df['duration']) 
       
    return df







# Function to compute Cramér’s V
def cramers_v(contingency_table):
    """Computes Cramér’s V statistic for measuring association between two categorical variables."""
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    k = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * k))




