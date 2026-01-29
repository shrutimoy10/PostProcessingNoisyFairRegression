#!/usr/bin/env python
# coding: utf-8

# In[1]:

#In this file, we consider only two groups, black and white

import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing
import tqdm.auto as tqdm
from tqdm.contrib.concurrent import process_map
from matplotlib.ticker import StrMethodFormatter
import random

import postprocess as postprocess_true
import postprocess_deconv_soft_assignment as postprocess

import warnings
warnings.filterwarnings('ignore')

split_ratio = 0.3
# seeds =  range(33, 83)
seeds = range(35,55)
data_dir = "data/communities"

max_workers = 32

#noise levels
# gamma = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
# gamma = np.round(np.arange(0.05, 0.45, 0.05),decimals=2)
gamma = [0.4]
import scipy.spatial




# ## Download UCI Communities and Crime dataset

# In[3]:



data_path = f"{data_dir}/communities.data"
if not os.path.exists(data_path):
  os.makedirs(data_dir, exist_ok=True)
  urllib.request.urlretrieve(
      "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data",
      data_path)

# To simpliy the case study, we will only use the columns that will be used for
# our model.
column_names = [
    # "state", 
    "county", "community", "communityname", "fold", "population",
    "householdsize", "racepctblack", "racePctWhite", "racePctAsian",
    "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up",
    "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf",
    "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc",
    "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap",
    "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov",
    "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed",
    "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu",
    "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv",
    "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par",
    "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom",
    "NumIlleg", "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5",
    "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5",
    "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell",
    "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous",
    "PersPerOwnOccHous", "PersPerRentOccHous", "PctPersOwnOccup",
    "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant",
    "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos",
    "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart",
    "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ",
    "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg",
    "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState",
    "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT",
    "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop",
    "LemasTotalReq", "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop",
    "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp",
    "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits",
    "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens",
    "PctUsePubTrans", "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr",
    "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop",
    "ViolentCrimesPerPop"
]

#Keeping the genders as is, flipping the race information only for now.
protected_columns = ["ispctblacklarge","ispctblacksmall"]

# Returns proxy column names given protected columns and noise param.
def get_proxy_column_names(noise_param):
    return ['PROXY_' + '%0.2f_' % noise_param + column_name for column_name in protected_columns]


# https://github.com/wenshuoguo/robust-fairness-code/blob/master/data.py
def generate_proxy_columns(df, noise_param=1):
    """Generates proxy columns from binarized protected columns.

    Args: 
      df: pandas dataframe containing protected columns, where each protected 
        column contains values 0 or 1 indicating membership in a protected group.
      protected_columns: list of strings, column names of the protected columns.
      noise_param: float between 0 and 1. Fraction of examples for which the proxy 
        columns will differ from the protected columns.

    Returns:
      df_proxy: pandas dataframe containing the proxy columns.
      proxy_columns: names of the proxy columns.
    """
    proxy_columns = get_proxy_column_names(noise_param)
    num_datapoints = len(df)
    num_groups = len(protected_columns)
    noise_idx = random.sample(range(num_datapoints), int(noise_param * num_datapoints))
    proxy_groups = np.zeros((num_groups, num_datapoints))
    df_proxy = df.copy()
    for i in range(num_groups):
        df_proxy[proxy_columns[i]] = df_proxy[protected_columns[i]]
    for j in noise_idx:
        group_index = -1
        for i in range(num_groups):
            if df_proxy[protected_columns[i]][j] == 1:
                df_proxy.at[j, protected_columns[i]] = 0
                group_index = i
                allowed_new_groups = list(range(num_groups))
                allowed_new_groups.remove(group_index)
                new_group_index = random.choice(allowed_new_groups)  
                df_proxy.at[j, protected_columns[new_group_index]] = 1
                break
        if group_index == -1:
            print('missing group information for datapoint ', j)
    return df_proxy

# Courtesy : https://github.com/wenshuoguo/robust-fairness-code/blob/master/softweights_training.py
def build_b(input_df, proxy_groups, true_groups, include_simplex_constraints=False):
    # If a proxy group has zero examples, appends 0.
    num_groups = len(proxy_groups)
    b = []
    for j in range(num_groups):
        for k in range(num_groups):
            # number of examples with proxy = k
            num_proxy = input_df[proxy_groups[k]].sum()
            if num_proxy == 0:
                b.append(0)
            else:
                # number of examples with true = j, proxy = k
                true_and_proxy = np.multiply(input_df[true_groups[j]],input_df[proxy_groups[k]])
                # print(f" j : {j}, k : {k} num_proxy : {num_proxy}, true_and_proxy : {true_and_proxy.sum()}")
                num_true_and_proxy = true_and_proxy.sum()
                # b_{jk} = P(G = j | \hat{G} = k)
                b.append(num_true_and_proxy / num_proxy)
    # if include_simplex_constraints:
    #     W_num_rows = num_groups*4
    #     for i in range(W_num_rows):
    #         b.append(1)
    return np.array(b)

def data_transform(df, noise_param=0.0, noisy=0):
  """Normalize features."""
  binary_data = pd.get_dummies(df, dtype=int)
  if noisy:
    binary_data = generate_proxy_columns(df, noise_param=noise_param)
  # scaler = sklearn.preprocessing.StandardScaler()
  # data = pd.DataFrame(scaler.fit_transform(binary_data),
  #                     columns=binary_data.columns)
  data = pd.DataFrame(binary_data, columns=binary_data.columns)
  data.index = df.index
  return data

#Groupwise KS Distance
def ks_dist(scores, groups):
  """Maximum pairwise KS distance"""
  n_groups = len(np.unique(groups))
  max_ks = 0
  for i in range(n_groups):
    max_ks = max(max_ks,
                   ks_2samp(scores[groups == i], scores).statistic)
  return max_ks



original = pd.read_csv(data_path,
                       index_col=0,
                       names=column_names,
                       sep=r",",
                       engine="python",
                       na_values="?")

original.dropna()

# Drop community name, state, and county, and columns with missing values
original = original.drop(["communityname", "county"],
                         axis=1).dropna(axis=1)

# Transform data, and remove fold and target
data = original.copy()
data = data.drop(["ViolentCrimesPerPop", "fold"], axis=1)
data = data_transform(data)

targets = original["ViolentCrimesPerPop"].to_numpy()

groups = (original["racepctblack"] > 0.06).to_numpy().astype(int)
data['ispctblacklarge'] = groups ## WE ARE ADDING A NEW COLUMN. SINCE THERE IS NO MODEL TRAINING, THIS
#SHOULD NOT EFFECT ANYTHING
data['ispctblacksmall'] = 1 - groups
group_names = ["white", "black"]

data.reset_index(inplace=True, drop=True)

n_groups = len(group_names)


bound = (0, 1)

# Show data statistics
df = pd.DataFrame(
    zip(np.array(group_names)[groups], targets),
    columns=["Group", "Target"],
).groupby(["Group"]).agg({"count"})
df.columns = df.columns.droplevel(0)
df["%"] = df["count"] / df["count"].sum()
# display(df)

df = pd.DataFrame({
  name: pd.Series(targets[groups == a]) for a, name in enumerate(group_names)
}).plot.kde()
plt.savefig("communities/kde_plot.png")
plt.close()


#arrays for plotting the scores
mse_no_postprocess_train = []
mse_no_postprocess_test = []
mse_true_postprocess_train = []
mse_true_postprocess_test = []
mse_naive_postprocess_train = []
mse_naive_postprocess_test = []
mse_reg_postprocess_train = []
mse_reg_postprocess_test = []
mse_correct_postprocess_train = []
mse_correct_postprocess_test = []

ks_no_postprocess_train = []
ks_no_postprocess_test = []
ks_true_postprocess_train = []
ks_true_postprocess_test = []
ks_naive_postprocess_train = []
ks_naive_postprocess_test = []
ks_reg_postprocess_train = []
ks_reg_postprocess_test = []
ks_correct_postprocess_train = []
ks_correct_postprocess_test = []

"""
CREATING NOISY DATA
 -- targets remain same, only the group membership changes with probability gamma.
 -- loop over different levels of noise
"""

for noise_level in gamma :  
  df_protected_column_names = ['ispctblacklarge','ispctblacksmall']
  df_proxy_column_names = get_proxy_column_names(noise_param=noise_level)

  noisy_data = data.copy()
  noisy_data = data_transform(noisy_data, noise_level, noisy=1)
  # noisy_data.reset_index(drop=True, inplace=True)


  #noisy group assignment
  noisy_groups = np.empty(len(noisy_data),dtype=int)

  # for i in range(len(noisy_data)):
  #   for j in range(len(df_protected_column_names)):
  #     if noisy_data[df_protected_column_names[j]][i] == 1:
  #       noisy_groups[i] = j

  noisy_groups = noisy_data[protected_columns[0]].to_numpy()
  
  print(f"Group assignments differ in {sum(groups != noisy_groups)} indices.")

   # Creating the noise model matrix, Pr( G = j | G_hat = i)
  b = build_b(noisy_data, df_proxy_column_names, df_protected_column_names)
  # if len(b.shape) > 1:
  b = b.reshape((len(df_protected_column_names),len(df_proxy_column_names)))
  b = b.T  # noisy_rows x true rows
  # print(b)


  # removing true group information from the dataframe
  noisy_data.drop(df_protected_column_names, axis=1, inplace=True)

  print(f"Noisy Data : {noisy_data}")

  df = pd.DataFrame({
    name: pd.Series(targets[noisy_groups == a]) for a, name in enumerate(df_protected_column_names)
  }).plot.kde()
  plt.savefig("communities/noisy_kde_plot_" + str(noise_level) +  ".png")
  plt.close()

  # ## Fair post-processing


  # ### Results and baselines for $\alpha = 0$


  n_bins = 12
  epsilons = [np.inf]

  results = np.empty((2, 11, len(seeds)))
  methods = [
      "no postprocessing(train)", "no postprocessing(test)", "Chzhen et al. (2020)(in-processing)", "binning( true fair PP, train)", "binning(true fair PP, test)", "binning(naive fair PP, train)",
        "binning(naive fair PP, test)", "binning(reg PP, train)", "binning(reg PP, test)" , "binning(new fair PP, train)", "binning(new fair PP, test)"]#, "binning + fair"] 
      #+ [f"binning + private and fair (eps={eps})" for eps in epsilons[1:]]

  for k, seed in enumerate(tqdm.tqdm(seeds)):

    # Split into train and test
    _, _, train_targets, test_targets, train_groups, test_groups, noisy_train_groups, noisy_test_groups = train_test_split(
        data,
        targets,
        groups,
        noisy_groups,
        test_size=split_ratio,
        random_state=seed,
    )

    noisy_train_targets = train_targets
    noisy_test_targets = test_targets

    # No postprocessing
    results[0, 0, k] = 0
    results[1, 0, k] = ks_dist(train_targets, train_groups)
    

    results[0, 1, k] = 0
    results[1, 1, k] = ks_dist(test_targets, test_groups)
    

    # Chzhen et al. (2020)
    postprocessor = postprocess.WassersteinBarycenterFairPostProcessor().fit(
        train_targets,
        train_groups,
        rng=np.random.default_rng(seed),
    )
    targets_test_fair = postprocessor.predict(test_targets, test_groups)
    results[0, 2, k] = mean_squared_error(test_targets, targets_test_fair)
    results[1, 2, k] = ks_dist(targets_test_fair, test_groups)


    # Binning
    postprocessor = postprocess_true.PrivateHDEFairPostProcessor().fit(
        train_targets,
        train_groups,
        alpha=0.0,
        bound=bound,
        n_bins=n_bins,
        rng=np.random.default_rng(seed),
    )
    targets_train_fair = postprocessor.predict(train_targets, train_groups)
    results[0, 3, k] = mean_squared_error(train_targets, targets_train_fair)
    results[1, 3, k] = ks_dist(targets_train_fair, train_groups)
   

    targets_test_fair = postprocessor.predict(test_targets, test_groups)
    results[0, 4, k] = mean_squared_error(test_targets, targets_test_fair)
    results[1, 4, k] = ks_dist(targets_test_fair, test_groups)
    


    # Binning Noisy Naive
    postprocessor = postprocess_true.PrivateHDEFairPostProcessor().fit(
        noisy_train_targets,
        noisy_train_groups,
        alpha=0.0,
        bound=bound,
        n_bins=n_bins,
        rng=np.random.default_rng(seed),
    )
    targets_train_fair = postprocessor.predict(train_targets, train_groups)
    results[0, 5, k] = mean_squared_error(train_targets, targets_train_fair)
    results[1, 5, k] = ks_dist(targets_train_fair, train_groups)
   

    targets_test_fair = postprocessor.predict(test_targets, test_groups)
    results[0, 6, k] = mean_squared_error(test_targets, targets_test_fair)
    results[1, 6, k] = ks_dist(targets_test_fair, test_groups)


    # Binning Noisy -- with Entropic Regularization
    postprocessor = postprocess_true.PrivateHDEFairPostProcessor().fit(
        noisy_train_targets,
        noisy_train_groups,
        alpha=0.0,
        bound=bound,
        n_bins=n_bins,
        regularize=1,
        rng=np.random.default_rng(seed),
    )
    targets_train_fair = postprocessor.predict(train_targets, train_groups)
    results[0, 7, k] = mean_squared_error(train_targets, targets_train_fair)
    results[1, 7, k] = ks_dist(targets_train_fair, train_groups)
   

    targets_test_fair = postprocessor.predict(test_targets, test_groups)
    results[0, 8, k] = mean_squared_error(test_targets, targets_test_fair)
    results[1, 8, k] = ks_dist(targets_test_fair, test_groups)
    


    
    # Correction postprocessing
    postprocessor = postprocess.PrivateHDEFairPostProcessor().fit(
        noisy_train_targets,
        noisy_train_groups,
        true_groups = train_groups, # true group assignments
        alpha=0.0,
        bound=bound,
        n_bins=n_bins,
        rng=np.random.default_rng(seed),
        noise=noise_level,
        noise_model=b,
    )

    if not postprocessor is None :
      targets_train_fair = postprocessor.predict(train_targets, train_groups)
      results[0, 9, k] = mean_squared_error(train_targets, targets_train_fair)
      results[1, 9, k] = ks_dist(targets_train_fair, train_groups)
      

      targets_test_fair = postprocessor.predict(test_targets, test_groups)
      results[0, 10, k] = mean_squared_error(test_targets, targets_test_fair)
      results[1, 10, k] = ks_dist(targets_test_fair, test_groups)
    else:
      continue
    

    
  print(f"Noise level : {noise_level}")
  means = results.mean(axis=2)
  stds = results.std(axis=2)
  remarks = ["No PP(train).", "No PP(test).","PP  true groups using Chzhen,  Res on test data.", \
  "PP on true groups. Res on true train data.","PP on true groups. Res on test data.", " Naive PP on noisy train data. Res on true train data.", \
  "Naive PP on noisy data. Res on test data.", " Reg. PP on noisy train data. Res on true train data.",\
  "Reg. PP on noisy train data. Res on test data." , " New PP on noisy train data. Res on true train data.",\
  "PP on noisy train data. Res on test data."]
  df = pd.DataFrame(np.stack([means[0], stds[0], means[1], stds[1], remarks], axis=1))
  df.columns = ["mse", "mse std", "ks", "ks std", "remarks"]
  df.index = methods
  # display(df)

  print(df)
  mse_no_postprocess_train.append(means[0,0])
  ks_no_postprocess_train.append(means[1,0])
  mse_no_postprocess_test.append(means[0,1])
  ks_no_postprocess_test.append(means[1,1])
  mse_true_postprocess_train.append(means[0,3])
  ks_true_postprocess_train.append(means[1,3])
  mse_true_postprocess_test.append(means[0,4])
  ks_true_postprocess_test.append(means[1,4])
  mse_naive_postprocess_train.append(means[0,5])
  ks_naive_postprocess_train.append(means[1,5])
  mse_naive_postprocess_test.append(means[0,6])
  ks_naive_postprocess_test.append(means[1,6])
  mse_reg_postprocess_train.append(means[0,7])
  ks_reg_postprocess_train.append(means[1,7])
  mse_reg_postprocess_test.append(means[0,8])
  ks_reg_postprocess_test.append(means[1,8])
  mse_correct_postprocess_train.append(means[0,9])
  ks_correct_postprocess_train.append(means[1,9])
  mse_correct_postprocess_test.append(means[0,10])
  ks_correct_postprocess_test.append(means[1,10])


plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places


plt.plot(np.asarray(mse_no_postprocess_train), "o" , linestyle="--" ,label="No_PP_train")
plt.plot( np.asarray(mse_true_postprocess_train), "*", linestyle="--" , label="True_PP_train")
plt.plot( np.asarray(mse_naive_postprocess_train), "^", linestyle="--", label="Naive_PP_train")
plt.plot( np.asarray(mse_reg_postprocess_train), "^", linestyle="--", label="Reg_PP_train(0.7)")
plt.plot( np.asarray(mse_correct_postprocess_train), ".", linestyle="--", label="Correct_PP_train")
plt.xticks(np.arange(len(gamma)), gamma)
plt.xlabel("Noise level")
plt.ylabel("MSE")
plt.title("Train MSE")
plt.legend()
plt.savefig("communities/MSE_train_deconv.pdf", dpi=200)
plt.close()

plt.plot( np.asarray(mse_no_postprocess_test), "o", linestyle="--", label="No_PP_test")
plt.plot( np.asarray(mse_true_postprocess_test), "*", linestyle="--", label="True_PP_test")
plt.plot( np.asarray(mse_naive_postprocess_test), "^", linestyle="--", label="Naive_PP_test")
plt.plot( np.asarray(mse_reg_postprocess_test), "^", linestyle="--", label="Reg_PP_test(0.7)")
plt.plot( np.asarray(mse_correct_postprocess_test), "." , linestyle="--", label="Correct_PP_test")
plt.xticks(np.arange(len(gamma)), gamma)
plt.xlabel("Noise level")
plt.ylabel("MSE")
plt.title("Test MSE")
plt.legend()
plt.savefig("communities/MSE_test_deconv.pdf", dpi=200)
plt.close()

plt.plot( np.asarray(ks_no_postprocess_train), "o" , linestyle="--", label="No_PP_train")
plt.plot( np.asarray(ks_true_postprocess_train), "*", linestyle="--", label="True_PP_train")
plt.plot( np.asarray(ks_naive_postprocess_train), "^", linestyle="--", label="Naive_PP_train")
plt.plot( np.asarray(ks_reg_postprocess_train), "^", linestyle="--", label="Reg_PP_train(0.7)")
plt.plot( np.asarray(ks_correct_postprocess_train), ".", linestyle="--", label="Correct_PP_train")
plt.xticks(np.arange(len(gamma)), gamma)
plt.xlabel("Noise level")
plt.ylabel("Ks_dist")
plt.title("Train KS dist")
plt.legend()
plt.savefig("communities/KS_dist_train_deconv.pdf", dpi=200)
plt.close()

plt.plot( np.asarray(ks_no_postprocess_test), "o", linestyle="--", label="No_PP_test")
plt.plot( np.asarray(ks_true_postprocess_test), "*", linestyle="--", label="True_PP_test")
plt.plot( np.asarray(ks_naive_postprocess_test), "^", linestyle="--", label="Naive_PP_test")
plt.plot( np.asarray(ks_reg_postprocess_test), "^", linestyle="--", label="Reg_PP_test(0.7)")
plt.plot( np.asarray(ks_correct_postprocess_test), ".", linestyle="--",label="Correct_PP_test")
plt.xticks(np.arange(len(gamma)), gamma)
plt.xlabel("Noise level")
plt.ylabel("Ks_dist")
plt.title("Test KS dist")
plt.legend()
plt.savefig("communities/KS_dist_test_deconv.pdf", dpi=200)
plt.close()