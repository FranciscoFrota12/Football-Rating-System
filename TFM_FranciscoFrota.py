#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from scipy.stats import norm, cauchy, gamma, lognorm, expon, beta, t, weibull_min
from scipy.optimize import OptimizeWarning


# In[3]:


excel_file_path = 'C:/Users/kikon/Desktop/TFM/Jugadores Wyscout/Players.xlsx'

df = pd.read_excel(excel_file_path)

# My excel contains the information from Wyscout platform but the data has been previously cleaned
# and the column names may vary from the original


# In[8]:


# Select all numeric columns in the Dataframe
numeric_columns = df.select_dtypes(include=[float,int]).columns

# List of possible distributions
distributions = [norm, cauchy, gamma, lognorm, expon, beta, t, weibull_min]

# Returns the description of each column and the distribution. Has code to go through errors
for column in numeric_columns:
    column_stats = df[column].describe()
    print("Column: ", column)
    print(column_stats)
    
    # Skip the column if it contains non-finite values
    if not np.isfinite(df[column]).all():
        print("Column:", column)
        print("Skipped (contains non-finite values)")
        print("\n")
        continue
    
    # Fit each candidate distribution to the column data
    best_fit_name = None
    best_fit_params = {}
    best_fit_sse = float('inf')  # Initialize with a high value
    
    fitting_error = False
    
    for distribution in distributions:
        try:
            # Fit the distribution to the column data
            params = distribution.fit(df[column])
            
            # Calculate the sum of squared errors (SSE)
            sse = sum((distribution.pdf(df[column], *params) - df[column]) ** 2)
            
            # Check if SSE is lower than the current best fit
            if sse < best_fit_sse:
                best_fit_name = distribution.name
                best_fit_params = params
                best_fit_sse = sse
        except ValueError as e:
            # Print the fitting error message
            print("Fitting Error for Column:", column)
            print("Distribution:", distribution.name)
            print("Error:", str(e))
            print("\n")
            
            fitting_error = True
    
    # Print the best-fitting distribution for the column
    if not fitting_error:
        print("Best-fitting Distribution:", best_fit_name)
        print("Parameters:", best_fit_params)
        print("\n")
    
    print("\n")


# In[9]:


# exclude the columns we dont want to normalize
exclude_columns = ["Market_Value", "Contract expires", "Id", "Playing_Team_ID", "Age", "Height", "Weight", "Matches_Played", "Minutes_Played"]

numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

# filter the DataFrame by position (CB, FullBack, DefMid, AttMid, CentreMid, Wingers and Forwards)

# we get a Dataframe for each position to see the percentile regarding the distribution of the players in that position

# we also filter by age (<=33) and matches played (<=10), to get the most adequate results to the scouting approach

# final Dataframes are normalized with z-score to mantain the distribution of the data. 

# the calc is Z = (x - mean)/ (stddev) where x is the value, mean is the mean of the column of the value and

# stddev is the standard deviation of the column of the value

######

#Centre Backs

cb_position = ["RCB", "CB", "LCB"]

cb_filtered_df = df[df['Position_Main'].isin(cb_position) | df['Position_secondary_1'].isin(cb_position) | df['Position_secondary_2'].isin(cb_position)].copy()

# z-score normalization
cb_filtered_df[numeric_columns] = (cb_filtered_df[numeric_columns] - cb_filtered_df[numeric_columns].mean()) / cb_filtered_df[numeric_columns].std()

cb_filtered_df = cb_filtered_df[cb_filtered_df['Matches_Played'] >= 10] 

cb_filtered_df = cb_filtered_df[cb_filtered_df['Age'] <= 33] 

######

# Full Backs

fb_position = ["RB", "RWB", "LB", "LWB"]

fb_filtered_df = df[df['Position_Main'].isin(fb_position) | df['Position_secondary_1'].isin(fb_position) | df['Position_secondary_2'].isin(cb_position)].copy()

# z-score normalization
fb_filtered_df[numeric_columns] = (fb_filtered_df[numeric_columns] - fb_filtered_df[numeric_columns].mean()) / fb_filtered_df[numeric_columns].std()

fb_filtered_df = fb_filtered_df[fb_filtered_df['Matches_Played'] >= 10]

fb_filtered_df = fb_filtered_df[fb_filtered_df['Age'] <= 33] 

######

# Centre Mids

cm_position = ["LCMF", "RCMF"]

cm_filtered_df = df[df['Position_Main'].isin(cm_position) | df['Position_secondary_1'].isin(cm_position) | df['Position_secondary_2'].isin(cm_position)].copy()

# z-score normalization
cm_filtered_df[numeric_columns] = (cm_filtered_df[numeric_columns] - cm_filtered_df[numeric_columns].mean()) / cm_filtered_df[numeric_columns].std()

cm_filtered_df = cm_filtered_df[cm_filtered_df['Matches_Played'] >= 10]

cm_filtered_df = cm_filtered_df[cm_filtered_df['Age'] <= 33] 

######

# Centre Defensive Mids

cdm_position = ["LDMF", "RDMF", "DMF"]

cdm_filtered_df = df[df['Position_Main'].isin(cdm_position) | df['Position_secondary_1'].isin(cdm_position) | df['Position_secondary_2'].isin(cdm_position)].copy()

# z-score normalization
cdm_filtered_df[numeric_columns] = (cdm_filtered_df[numeric_columns] - cdm_filtered_df[numeric_columns].mean()) / cdm_filtered_df[numeric_columns].std()

cdm_filtered_df = cdm_filtered_df[cdm_filtered_df['Matches_Played'] >= 10]

cm_filtered_df = cm_filtered_df[cm_filtered_df['Age'] <= 33] 

######

# Centre Attacking Mids

cam_position = ["AMF"]

cam_filtered_df = df[df['Position_Main'].isin(cam_position) | df['Position_secondary_1'].isin(cam_position) | df['Position_secondary_2'].isin(cam_position)].copy()

# z-score normalization
cam_filtered_df[numeric_columns] = (cam_filtered_df[numeric_columns] - cam_filtered_df[numeric_columns].mean()) / cam_filtered_df[numeric_columns].std()

cam_filtered_df = cam_filtered_df[cam_filtered_df['Matches_Played'] >= 10]

cam_filtered_df = cam_filtered_df[cam_filtered_df['Age'] <= 33] 

######

# Wingers

wing_position = ["LW", "RW", "LAMF", "RAMF"]

wing_filtered_df = df[df['Position_Main'].isin(wing_position) | df['Position_secondary_1'].isin(wing_position) | df['Position_secondary_2'].isin(wing_position)].copy()

# z-score normalization
wing_filtered_df[numeric_columns] = (wing_filtered_df[numeric_columns] - wing_filtered_df[numeric_columns].mean()) / wing_filtered_df[numeric_columns].std()

wing_filtered_df = wing_filtered_df[wing_filtered_df['Matches_Played'] >= 10]

wing_filtered_df = wing_filtered_df[wing_filtered_df['Age'] <= 33] 

######

# Centre Forwards

cf_position = ["CF", "LWF", "RWF"]

cf_filtered_df = df[df['Position_Main'].isin(cf_position) | df['Position_secondary_1'].isin(cf_position) | df['Position_secondary_2'].isin(cf_position)].copy()

# z-score normalization
cf_filtered_df[numeric_columns] = (cf_filtered_df[numeric_columns] - cf_filtered_df[numeric_columns].mean()) / cf_filtered_df[numeric_columns].std()

cf_filtered_df = cf_filtered_df[cf_filtered_df['Matches_Played'] >= 10]

cf_filtered_df = cf_filtered_df[cf_filtered_df['Age'] <= 33] 


# In[21]:


# # Selection of the key metrics regarding the style of play and game model desired

cb_key_metrics = ["Accurate_long_passes_pc","Goals","Progressive_passes_p90","PAdj_Sliding_tackles","Aerial_duels_won_pc","Padj_Interceptions","Defensive_duels_won_pc"]

fb_key_metrics = ["Received_long_passes_p90","Accelerations_p90","Aerial_duels_won_pc", "Accurate_crosses_pc", "Successful_def_actions_p90", "Accurate_short_medium_passes_pc"]

cdm_key_metrics = ["PAdj_Sliding_tackles", "Successful_def_actions_p90", "Progressive_passes_p90", "Accurate_short_medium_passes_pc", "Aerial_duels_won_pc"]

cm_key_metrics = ["Accelerations_p90", "Progressive_runs_p90","Successful_def_actions_p90", "Accurate_short_medium_passes_pc", "xG_p90", "Touches_in_box_p90", "Aerial_duels_won_pc", "Accurate_through_passes_pc"]

cam_key_metrics = ["xG_p90", "xA_p90", "Touches_in_box_p90", "Deep_completions_p90", "Accurate_short_medium_passes_pc", "Shot_assists_p90", "Successful_att_actions_p90"]

wing_key_metrics = ["Progressive_runs_p90","Accelerations_p90","Successful_att_actions_p90","Accurate_passes_to_final_third_pc","Shot_assists_p90","xA_p90","Successful_def_actions_p90"]

cf_key_metrics = ["xG_p90", "Aerial_duels_won_pc", "Goals", "Successful_def_actions_p90", "Offensive_duels_won_pc", "Shots_on_target_pc"]

# Rating calculation

# First we chose weights for each metric selected above.

# Then we calculate the percentiles of each metric.

# Finally we calculate the weighted mean and thats the result of our rating

# The number doesn't tell much but because we then rank the players 

#in descending order, we can see who is the best suited for each position

# the result will present a list of the top 50 in each position (if needed we can make it bigger)

###### Centre Backs ######

# Weight for each metric
cb_metric_weights = {
    "Accurate_long_passes_pc": 0.75,
    "Goals": 0.5,
    "Progressive_passes_p90": 0.8,
    "PAdj_Sliding_tackles": 1.0,
    "Aerial_duels_won_pc": 1.0,
    "Padj_Interceptions": 1.0,
    "Defensive_duels_won_pc": 1.0
}

# Calculate the percentiles for each metric 
percentiles_cb = cb_filtered_df[cb_key_metrics].rank(pct=True)

# Calculate the mean of percentiles for each player
cb_filtered_df['cb_rating'] = percentiles_cb.mul(cb_metric_weights, axis=1).mean(axis=1)

# Sort the DataFrame by cb_rating in descending order
sorted_df_cb = cb_filtered_df.sort_values('cb_rating', ascending=False)

# Print the information of the top 50 players with the highest cb_rating
top_50_cb = sorted_df_cb.head(50)[['Player', "Team_selected_period", "League", 'cb_rating']]
print(top_50_cb)
print("\n")

###### Full Backs ######

# Weight for each metric
fb_metric_weights = {
    "Accelerations_p90": 0.7,
    "Accurate_crosses_pc": 1.0,
    "Successful_def_actions_p90": 1.0,

    "Aerial_duels_won_pc": 1.0,
    "Accurate_short_medium_passes_pc": 0.8,
    "Received_long_passes_p90": 0.75
}

# Calculate the percentiles for each metric 
percentiles_fb = fb_filtered_df[fb_key_metrics].rank(pct=True)

# Calculate the mean of percentiles for each player
fb_filtered_df['fb_rating'] = percentiles_fb.mul(fb_metric_weights, axis=1).mean(axis=1)

# Sort the DataFrame by cb_rating in descending order
sorted_df_fb = fb_filtered_df.sort_values('fb_rating', ascending=False)

# Print the information of the top 50 players with the highest cb_rating
top_50_fb = sorted_df_fb.head(50)[['Player',"Team_selected_period", "League", 'fb_rating']]
print(top_50_fb)
print("\n")

###### Defensive Midfielders ######

# Weight for each metric
cdm_metric_weights = {
    "PAdj_Sliding_tackles": 0.7,
    "Successful_def_actions_p90": 1.0,
    "Progressive_passes_p90": 0.9,
    "Aerial_duels_won_pc": 1.0,
    "Accurate_short_medium_passes_pc": 0.8
}

# Calculate the percentiles for each metric 
percentiles_cdm = cdm_filtered_df[cdm_key_metrics].rank(pct=True)

# Calculate the mean of percentiles for each player
cdm_filtered_df['cdm_rating'] = percentiles_cdm.mul(cdm_metric_weights, axis=1).mean(axis=1)

# Sort the DataFrame by cb_rating in descending order
sorted_df_cdm = cdm_filtered_df.sort_values('cdm_rating', ascending=False)

# Print the information of the top 50 players with the highest cb_rating
top_50_cdm = sorted_df_cdm.head(50)[['Player',"Team_selected_period", "League", 'cdm_rating']]
print(top_50_cdm)
print("\n")

###### Centre Midfielders ######

# Weight for each metric
cm_metric_weights = {
    "Accelerations_p90": 0.7,
    "Successful_def_actions_p90": 1.0,
    "xG_p90": 1.0,
    "Aerial_duels_won_pc": 0.75,
    "Accurate_short_medium_passes_pc": 1.0,
    "Touches_in_box_p90": 1.0,
    "Accurate_through_passes_pc": 0.8,
    "Progressive_runs_p90": 0.7
}

# Calculate the percentiles for each metric 
percentiles_cm = cm_filtered_df[cm_key_metrics].rank(pct=True)

# Calculate the mean of percentiles for each player
cm_filtered_df['cm_rating'] = percentiles_cm.mul(cm_metric_weights, axis=1).mean(axis=1)

# Sort the DataFrame by cb_rating in descending order
sorted_df_cm = cm_filtered_df.sort_values('cm_rating', ascending=False)

# Print the information of the top 50 players with the highest cb_rating
top_50_cm = sorted_df_cm.head(50)[['Player',"Team_selected_period", "League", 'cm_rating']]
print(top_50_cm)
print("\n")

###### Attacking Midfielders ######

# Weight for each metric
cam_metric_weights = {
    "xG_p90": 0.75,
    "xA_p90": 1.0,
    "Successful_att_actions_p90": 1.0,
    "Accurate_passes_to_final_third_pc": 0.75,
    "Accurate_short_medium_passes_pc": 1.0,
    "Touches_in_box_p90": 0.8,
    "Shot_assists_p90": 1.0
}

# Calculate the percentiles for each metric 
percentiles_cam = cam_filtered_df[cam_key_metrics].rank(pct=True)

# Calculate the mean of percentiles for each player
cam_filtered_df['cam_rating'] = percentiles_cam.mul(cam_metric_weights, axis=1).mean(axis=1)

# Sort the DataFrame by cb_rating in descending order
sorted_df_cam = cam_filtered_df.sort_values('cam_rating', ascending=False)

# Print the information of the top 50 players with the highest cb_rating
top_50_cam = sorted_df_cam.head(50)[['Player',"Team_selected_period", "League", 'cam_rating']]
print(top_50_cam)
print("\n")

###### Wingers ######

# Weight for each metric
wing_metric_weights = {
    "Accelerations_p90": 0.75,
    "xA_p90": 1.0,
    "Successful_att_actions_p90": 1.0,
    "Shot_assists_p90": 1.0,
    "Accurate_passes_to_final_third_pc": 1.0,
    "Successful_def_actions_p90": 0.9,
    "Progressive_runs_p90": 1.0
}

# Calculate the percentiles for each metric 
percentiles_wing = wing_filtered_df[wing_key_metrics].rank(pct=True)

# Calculate the mean of percentiles for each player
wing_filtered_df['wing_rating'] = percentiles_wing.mul(wing_metric_weights, axis=1).mean(axis=1)

# Sort the DataFrame by cb_rating in descending order
sorted_df_wing = wing_filtered_df.sort_values('wing_rating', ascending=False)

# Print the information of the top 50 players with the highest cb_rating
top_50_wing = sorted_df_wing.head(50)[['Player',"Team_selected_period", "League", 'wing_rating']]
print(top_50_wing)
print("\n")

###### Centre Forwards ######

# Weight for each metric
cf_metric_weights = {
    "xG_p90": 1.0,
    "Goals": 1.0,
    "Successful_def_actions_p90": 0.75,
    "Offensive_duels_won_pc": 0.8,
    "Aerial_duels_won_pc": 1.0,
    "Shots_on_target_pc": 0.75
}

# Calculate the percentiles for each metric 
percentiles_cf = cf_filtered_df[cf_key_metrics].rank(pct=True)

# Calculate the mean of percentiles for each player
cf_filtered_df['cf_rating'] = percentiles_cf.mul(cf_metric_weights, axis=1).mean(axis=1)

# Sort the DataFrame by cb_rating in descending order
sorted_df_cf = cf_filtered_df.sort_values('cf_rating', ascending=False)

# Print the information of the top 50 players with the highest cb_rating
top_50_cf = sorted_df_cf.head(50)[['Player',"Team_selected_period", "League", 'cf_rating']]
print(top_50_cf)
print("\n")


# In[ ]:




