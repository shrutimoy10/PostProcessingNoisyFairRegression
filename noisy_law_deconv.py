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
import postprocess_deconv as postprocess

import warnings
warnings.filterwarnings('ignore')

split_ratio = 0.3
# seeds =  range(33, 83)
seeds = range(35,55)
data_dir = "data/law"

max_workers = 32

#noise levels
# gamma = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45] XXXXX
# gamma = np.round(np.arange(0.05, 0.5, 0.05),decimals=2)
# gamma = [0.4]
gamma = [0.15, 0.25, 0.35, 0.45]

import scipy.spatial




# ## Download LSAC Law School dataset


data_path = f"{data_dir}/bar_pass_prediction.csv"
if not os.path.exists(data_path):
  os.makedirs(data_dir, exist_ok=True)
  urllib.request.urlretrieve(
      "https://storage.googleapis.com/lawschool_dataset/bar_pass_prediction.csv",
      data_path)

# To simpliy the case study, we will only use the columns that will be used for
# our model.
column_names = [
    'dnn_bar_pass_prediction',
    'gender',
    'lsat',
    'race1',
    'pass_bar',
    'ugpa',
]

#Keeping the genders as is, flipping the race information only for now.
protected_columns = ['race1_asian', 'race1_black', 'race1_hisp', 'race1_white']

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
    # proxy_columns = get_proxy_column_names(noise_param)
    num_datapoints = len(df)
    num_groups = len(protected_columns)
    noise_idx = random.sample(range(num_datapoints), int(noise_param * num_datapoints))
    proxy_groups = np.zeros((num_groups, num_datapoints))
    df_proxy = df.copy()
    # for i in range(num_groups):
    #     df_proxy[protected_columns[i]] = df_proxy[protected_columns[i]]
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
                       sep=r",",
                       engine="python",
                       na_values="?")

original.dropna()
original['gender'] = original['gender'].astype(str)
original['race1'] = original['race1'].astype(str)
original = original[column_names]
race = ['asian', 'black', 'hisp', 'white']
original = original[original['race1'].isin(race)]
"""
ORIGINAL DATA
"""
# Transform data, and remove fold and target
data = original.copy()
data = data.drop(["ugpa"], axis=1)
data = data_transform(data)
data.reset_index(drop=True, inplace=True)

targets = original["ugpa"].to_numpy()
# targets = (targets - 1.0) / (4.0 - 1.0)
group_names, groups = np.unique(original["race1"], return_inverse=True)
n_groups = len(group_names)


bound = (1, 4)

# Show data statistics
df = pd.DataFrame(
    zip(np.array(group_names)[groups], targets),
    columns=["Group", "Target"],
).groupby(["Group"]).agg({"count"})
df.columns = df.columns.droplevel(0)
df["%"] = df["count"] / df["count"].sum()
print(df)
true_w = df['%'].to_numpy()

df = pd.DataFrame({
  name: pd.Series(targets[groups == a]) for a, name in enumerate(group_names)
}).plot.kde()
plt.savefig("law/kde_plot.png")
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

diff_in_w = []

"""
CREATING NOISY DATA
 -- targets remain same, only the group membership changes with probability gamma.
 -- loop over different levels of noise
"""

for noise_level in gamma :  
  noisy_data = data.copy()
  noisy_data = data_transform(noisy_data, noise_level, noisy=1)
  noisy_data.reset_index(drop=True, inplace=True)


  df_protected_column_names = ['race1_asian', 'race1_black', 'race1_hisp', 'race1_white']

  #noisy group assignment
  noisy_groups = np.empty(len(noisy_data),dtype=int)

  for i in range(len(noisy_data)):
    for j in range(len(df_protected_column_names)):
      if noisy_data[df_protected_column_names[j]][i] == 1:
        noisy_groups[i] = j

  print(f"Group assignments differ in {sum(groups != noisy_groups)} indices.")

  df = pd.DataFrame({
    name: pd.Series(targets[noisy_groups == a]) for a, name in enumerate(df_protected_column_names)
  }).plot.kde()
  plt.savefig("law/noisy_kde_plot_" + str(noise_level) +  ".png")
  plt.close()

  # ## Fair post-processing


  # ### Results and baselines for $\alpha = 0$


  n_bins = 36
  epsilons = [np.inf]

  results = np.empty((2, 11, len(seeds)))
  methods = [
      "no postprocessing(train)", "no postprocessing(test)", "Chzhen et al. (2020)(in-processing)", "binning( true fair PP, train)", "binning(true fair PP, test)", "binning(naive fair PP, train)",
        "binning(naive fair PP, test)", "binning(reg PP, train)", "binning(reg PP, test)" , "binning(new fair PP, train)", "binning(new fair PP, test)"]#, "binning + fair"] 
      #+ [f"binning + private and fair (eps={eps})" for eps in epsilons[1:]]

  w_err = []
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
    postprocessor, diff_w = postprocess.PrivateHDEFairPostProcessor().fit(
        noisy_train_targets,
        noisy_train_groups,
        true_groups = train_groups, # true group assignments
        alpha=0.0,
        bound=bound,
        n_bins=n_bins,
        rng=np.random.default_rng(seed),
        noise=noise_level,
        true_w=true_w,
    )

    if not postprocessor is None:
      targets_train_fair = postprocessor.predict(train_targets, train_groups)
      results[0, 9, k] = mean_squared_error(train_targets, targets_train_fair)
      results[1, 9, k] = ks_dist(targets_train_fair, train_groups)
      

      targets_test_fair = postprocessor.predict(test_targets, test_groups)
      results[0, 10, k] = mean_squared_error(test_targets, targets_test_fair)
      results[1, 10, k] = ks_dist(targets_test_fair, test_groups)

      w_err.append(diff_w)
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

  diff_in_w.append(np.mean(w_err))

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
plt.rcParams.update({'font.size': 14}) 

plt.plot(np.asarray(mse_no_postprocess_train), "o" , linestyle="--" ,label="No_PP_train")
plt.plot( np.asarray(mse_true_postprocess_train), "*", linestyle="--" , label="True_PP_train")
plt.plot( np.asarray(mse_naive_postprocess_train), "^", linestyle="--", label="Naive_PP_train")
plt.plot( np.asarray(mse_reg_postprocess_train), "^", linestyle="--", label="Reg_PP_train(0.0)")
plt.plot( np.asarray(mse_correct_postprocess_train), ".", linestyle="--", label="Correct_PP_train")
plt.xticks(np.arange(len(gamma)), gamma)
plt.xlabel("Noise level")
plt.ylabel("MSE")
plt.title("Train MSE")
plt.legend()
plt.savefig("law/MSE_train_deconv_test_0.6.pdf", dpi=200)
plt.close()

plt.plot( np.asarray(mse_no_postprocess_test), "o", linestyle="--", label="No_PP_test")
plt.plot( np.asarray(mse_true_postprocess_test), "*", linestyle="--", label="True_PP_test")
plt.plot( np.asarray(mse_naive_postprocess_test), "^", linestyle="--", label="Naive_PP_test")
plt.plot( np.asarray(mse_reg_postprocess_test), "^", linestyle="--", label="Reg_PP_test(0.0)")
plt.plot( np.asarray(mse_correct_postprocess_test), "." , linestyle="--", label="Correct_PP_test")
plt.xticks(np.arange(len(gamma)), gamma)
plt.xlabel("Noise level")
plt.ylabel("MSE")
plt.title("Test MSE")
plt.legend()
plt.savefig("law/MSE_test_deconv_test_0.6.pdf", dpi=200)
plt.close()

plt.plot( np.asarray(ks_no_postprocess_train), "o" , linestyle="--", label="No_PP_train")
plt.plot( np.asarray(ks_true_postprocess_train), "*", linestyle="--", label="True_PP_train")
plt.plot( np.asarray(ks_naive_postprocess_train), "^", linestyle="--", label="Naive_PP_train")
plt.plot( np.asarray(ks_reg_postprocess_train), "^", linestyle="--", label="Reg_PP_train(0.0)")
plt.plot( np.asarray(ks_correct_postprocess_train), ".", linestyle="--", label="Correct_PP_train")
plt.xticks(np.arange(len(gamma)), gamma)
plt.xlabel("Noise level")
plt.ylabel("Ks_dist")
plt.title("Train KS dist")
plt.legend()
plt.savefig("law/KS_dist_train_deconv_test_0.6.pdf", dpi=200)
plt.close()

plt.plot( np.asarray(ks_no_postprocess_test), "o", linestyle="--", label="No_PP")
plt.plot( np.asarray(ks_true_postprocess_test), "*", linestyle="--", label="True_PP")
plt.plot( np.asarray(ks_naive_postprocess_test), "^", linestyle="--", label="Naive_PP")
# plt.plot( np.asarray(ks_reg_postprocess_test), "^", linestyle="--", label="Reg_PP_test(0.7)")
plt.plot( np.asarray(ks_correct_postprocess_test), ".", linestyle="--",label="DeRFP")
plt.xticks(np.arange(len(gamma)), gamma)
plt.xlabel("$\\epsilon$")
plt.ylabel("$\\Delta_{SP}$", fontsize=17)
# plt.title("Test KS dist")
plt.legend()
plt.savefig("law/KS_dist_test_deconv_law.pdf", dpi=200)
plt.close()

plt.plot( diff_in_w, ".", linestyle="--",label="w estimate error")
plt.xticks(np.arange(len(gamma)), gamma)
plt.xlabel("Noise level")
plt.ylabel("Error")
plt.title("W Estimation Error")
plt.legend()
plt.savefig("law/w_est_error_test_0.6.pdf", dpi=200)
plt.close()