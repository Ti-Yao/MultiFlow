import os
import csv
from time import time
import numpy as np
import pandas as pd
import glob
import random
import copy


# Other libraries
import scipy.stats as stats
from scipy.stats import f_oneway, kruskal
import scikit_posthocs as sp
from tslearn.clustering import silhouette_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Lifelines
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test

# DTC 
from TAE import *
from DTC import *
from matplotlib.ticker import FormatStrFormatter

from statannotations.Annotator import Annotator

from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

from lifelines.fitters.cox_time_varying_fitter import CoxTimeVaryingFitter


colors = {'lpa':['#FF8357','#E45825','#B83100','#BE4417'],
        'rpa':['#FAC172','#F69813','#D98508','#DC901E'],
        'ao':['#89D5C9','#59C5B4','#2A796D','#DC901E'],
        'svc':['#ADC965','#94B540','#576A25','#DC901E'],
        'ivc':['#CFBAE1','#AD8DCE','#523172','#DC901E']}
ordered_colors = [colors[key][i] for i in range(len(colors['lpa'])) for key in list(colors.keys())]


def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    tf.keras.utils.set_random_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def get_true_gpu_ids():
    from tensorflow.python.client import device_lib
    devices = device_lib.list_local_devices()
    return {d.name: d.physical_device_desc for d in devices if d.device_type == "GPU"}


def interpolate_curve(flow, curve_points):
    int_times = np.linspace(0, len(flow), num = curve_points)
    int_flow = np.interp(int_times, range(len(flow)), flow)
    return int_flow


def create_survival_event_table(df, patient_col, stop_col, status_col, cluster_col):
    """
    Generates a survival event table tracking patient counts, deaths, and numbers at risk over time.

    Parameters:
        df (DataFrame): The input dataset.
        patient_col (str): Column name for patient IDs.
        stop_col (str): Column name for event time (death/transplant/censoring).
        status_col (str): Column name for event status (1 = event, 0 = censored).
        cluster_col (str): Column name for cluster/grouping.

    Returns:
        DataFrame: A survival event table with patient counts, deaths, and numbers at risk over time.
    """
    # Filter required columns and drop NaN values
    event_df = df[[patient_col, stop_col, status_col, cluster_col]].dropna()

    # Sort by event time
    event_df = event_df.sort_values(stop_col)

    # Extract unique time points
    time_points = np.sort(event_df[stop_col].unique())

    # Get unique clusters
    clusters = np.arange(event_df[cluster_col].nunique())

    # Initialize results list
    results = []

    # Initial patient count per cluster
    all_patients = {cluster: event_df[event_df[cluster_col] == cluster][patient_col].nunique() for cluster in clusters}
    
    # Initial numbers at risk are the same as total patients at time = 0
    numbers_at_risk = {cluster: len(event_df[event_df[cluster_col] == cluster][patient_col]) for cluster in clusters}

    # Create initial row (time = 0)
    row = {'time': 0}
    for cluster in all_patients.keys():
        row[f'cluster_{cluster}_n_patients'] = all_patients[cluster]
        row[f'cluster_{cluster}_n_deaths'] = 0  # No deaths at time 0
        row[f'cluster_{cluster}_n_at_risk'] = numbers_at_risk[cluster]  # Initial at-risk count
    results.append(row)

    # Iterate over each event time point
    for time in time_points:
        # Patients still alive at this time
        alive_at_time = event_df[event_df[stop_col] >= time]
        cluster_counts = alive_at_time.groupby(cluster_col)[patient_col].nunique().to_dict()

        # Count deaths and censoring at this time
        deaths_at_time = event_df[(event_df[stop_col] == time) & (event_df[status_col] == 1)]
        cluster_deaths = deaths_at_time.groupby(cluster_col)[patient_col].nunique().to_dict()
        
        censored_at_time = event_df[(event_df[stop_col] == time) & (event_df[status_col] == 0)]
        cluster_censored = censored_at_time.groupby(cluster_col)[patient_col].nunique().to_dict()

        # Update patient counts and numbers at risk
        for cluster in clusters:
            numbers_at_risk[cluster] = numbers_at_risk.get(cluster, 0) - cluster_deaths.get(cluster, 0) - cluster_censored.get(cluster, 0)

        # Create row for this time
        row = {'time': time}
        for cluster in clusters:
            row[f'cluster_{cluster}_n_patients'] = cluster_counts.get(cluster, 0)
            row[f'cluster_{cluster}_n_deaths'] = cluster_deaths.get(cluster, 0)
            row[f'cluster_{cluster}_n_at_risk'] = max(0, numbers_at_risk.get(cluster, 0))  # Ensure non-negative values
        results.append(row)

    # Convert results to DataFrame
    final_table = pd.DataFrame(results)

    # Set time as index
    final_table.set_index('time', inplace=True)

    # Reshape to multi-index columns
    final_table.columns = pd.MultiIndex.from_tuples(
        [(f'Cluster {cluster + 1}', metric) for cluster in clusters for metric in ['n_patients', 'n_deaths', 'n_at_risk']],
        names=['cluster', 'metric']
    )

    return final_table



def plot_simon_makuch(ax, kaplan_df, ordered_colors, name):
    """
    Plots the Simon-Makuch survival curves based on the provided survival event table.

    Parameters:
        survival_table (DataFrame): Survival event table with patient and death counts.
        ordered_colors (list): List of colors for each cluster.
    """
    survival_table = create_survival_event_table(kaplan_df, 
                                                 patient_col = 'force_id', 
                                                 stop_col = f'stop_{name}', 
                                                 status_col = f'status_{name}_varying', 
                                                 cluster_col = 'cluster')
    
    n_at_risk_df = survival_table.xs("n_at_risk", axis=1, level=1)
    mask = n_at_risk_df.lt(10)
    first_index = int(mask.any(axis=1).idxmax())



    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        save_plot = True
    else:
        save_plot = False

    labels = []
    min_survival = 1.0  # Track the minimum survival probability

    for cluster_id, cluster in enumerate(survival_table.columns.levels[0]):
        # Initialize survival probability
        survival_probabilities = [1.0]
        patients_at_risk = [survival_table[(cluster, 'n_patients')].iloc[0]]
        death_counts = [survival_table[(cluster, 'n_deaths')].iloc[0]]

        # Calculate survival probabilities
        for i in range(1, len(survival_table)):
            patients_at_risk.append(survival_table[(cluster, 'n_patients')].iloc[i])
            death_counts.append(survival_table[(cluster, 'n_deaths')].iloc[i])

            deaths_today = death_counts[i]
            at_risk_today = patients_at_risk[i - 1]
            survival_today = survival_probabilities[i - 1] * (1 - deaths_today / at_risk_today)
            survival_probabilities.append(survival_today)

        survival_probabilities = survival_probabilities[:first_index]
        times = survival_table.index[:first_index]

        # Plot survival function
        labels.append(cluster)
        if ordered_colors:
            ax.step(times, survival_probabilities, where='post', color=ordered_colors[cluster_id], linewidth=4)
        else:
            ax.step(times, survival_probabilities, where='post', linewidth=4)

        # Track the minimum survival probability
        min_survival = min(min_survival, min(survival_probabilities))

    p_df = cox_time_varying_corrected(kaplan_df, name = name)
    correction = 'All'
    corrected_p_df = p_df[('p',correction)]
    corrected_p_df.index = corrected_p_df.index.set_levels([corrected_p_df.index.levels[i].astype(int) for i in range(corrected_p_df.index.nlevels)])

    for reference_index in corrected_p_df.index.levels[0]:
        for index in corrected_p_df.index.levels[1]:
            if reference_index != index:
                p = corrected_p_df.loc[reference_index, index]
                if p < 0.05:
                    labels[reference_index] = f'{labels[reference_index]} {index + 1}*'

    corrected_p_df = p_df[('p',correction)]
    corrected_p_df.index = corrected_p_df.index.set_levels([corrected_p_df.index.levels[i].astype(int) for i in range(corrected_p_df.index.nlevels)])


    y_min = np.floor(min_survival * 10) / 10  # Floor to nearest 0.1
    y_min = y_min + 0.1
    ax.set_yticks(np.arange(y_min, 1.1, 0.1))  # Ticks from y_min to 1 in 0.1 steps
    ax.set_ylim(y_min,1)
    ax.set_xlim(0,first_index)
    
    # Customize plot
    label_dict = {'death':'Mortality','liver':'Liver Disease','death_transplant':'Transplant/Death'}
    ax.set_title(f'Simon-Makuch:{label_dict[name]}', fontsize=16)
    ax.set_xlabel("Time (Days)", fontsize=16)
    ax.set_ylabel("Probability", fontsize=16)
    ax.legend(loc='lower left', labels=labels, framealpha=0, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=14)

    if save_plot:
        plt.savefig(f'results/simon_makuch_plot_{name}.png', dpi=600, bbox_inches='tight')
        plt.close(fig)


def plot_autoencoder_output(X_scaled, output, name,  chosen_vessels):
    for i in range(20):
        channel_dim = X_scaled.shape[-1]
        fig, axs = plt.subplots(1,channel_dim , figsize=(channel_dim * 5, 5))
        if channel_dim == 1:
            axs = [axs]
        for vessel_id, ax in enumerate(axs):
            ax.plot(X_scaled[i,:,vessel_id], label = 'input')
            ax.plot(output[i,:,vessel_id], label = 'autoencoder')
            ax.set_title(chosen_vessels[vessel_id])
            ax.legend()
        plt.savefig(f'results/{name}_{i}.png', bbox_inches='tight')
        plt.close()



def plot_kaplan_only(ax, ordered_colors, name, kaplan_df):
    kaplan_df = kaplan_df.dropna(subset =  f'status_{name}_varying') #[['force_id','patient','cmr_date_year', f'status_{name}_varying', f'start_{name}', f'stop_{name}', 'cluster']].dropna()

    n_clusters = kaplan_df.cluster.nunique()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        save_plot = True
    else:
        save_plot = False

    labels = []
    kmf = KaplanMeierFitter()
    for cluster in range(n_clusters):
        cluster_data = kaplan_df[kaplan_df['cluster'] == cluster]
        
        kmf.fit(cluster_data[f'stop_{name}'], event_observed=cluster_data[f'status_{name}_varying'])
        
        kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=4, color=ordered_colors[cluster])
        labels.append(f'Cluster {cluster + 1} (N = {len(cluster_data)})')

    p_df = cox_time_varying_corrected(kaplan_df, name = name)
    correction = 'All'
    corrected_p_df = p_df[('p',correction)]
    corrected_p_df.index = corrected_p_df.index.set_levels([corrected_p_df.index.levels[i].astype(int) for i in range(corrected_p_df.index.nlevels)])

    for reference_index in corrected_p_df.index.levels[0]:
        for index in corrected_p_df.index.levels[1]:
            if reference_index != index:
                p = corrected_p_df.loc[reference_index, index]
                if p < 0.05:
                    labels[reference_index] = f'{labels[reference_index]} {index + 1}*'

    corrected_p_df = p_df[('p',correction)]
    corrected_p_df.index = corrected_p_df.index.set_levels([corrected_p_df.index.levels[i].astype(int) for i in range(corrected_p_df.index.nlevels)])

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel('Time to Event (days)', fontsize=16)
    ax.set_ylabel('Probability', fontsize=16)
    label_dict = {'death':'Mortality','liver':'Liver Disease','death_transplant':'Transplant/Mortality'}
    ax.set_title(f'Simon-Makuch:{label_dict[name]}', fontsize=16)
    ax.legend(loc='lower left', labels=labels, framealpha=0, fontsize=12)

    if save_plot:
        plt.savefig(f'results/kaplan_meier_plot_{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)



def plot_cluster_kaplan(clean_df, chosen_vessels, ordered_colors, vessel_pair, kaplan_df):

    if vessel_pair == 'pa':
        cluster_names = {0:'Norm-P',1:'Low-Low',2:'Dia-P',3:'RPA-P',4:'LPA-P'}
    elif vessel_pair == 'vc':
        cluster_names = {0:'IVC-P',1:'SVC-P',2:'Norm-P',3:'Low-P'}
    else:
        cluster_names = {0:'IVC-P',1:'SVC-P',2:'Norm-P',3:'',4:'',5:''}


    kaplan_df = kaplan_df.merge(clean_df.reset_index())
    n_clusters = kaplan_df.cluster.nunique()
    n_plots = 6

    # Now find rows and columns for this aspect ratio.
    grid_rows = int(np.sqrt(n_plots) + 0.5)  # Round.
    grid_cols = (n_plots + grid_rows - 1) // grid_rows     # Ceil.        
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize = (grid_cols*4.75,grid_rows*4.6))
    middle_subplot = axes[0, 1]

    axes = axes.ravel()

    plot_axes = [axes[0],axes[1],axes[3],axes[4]] if n_clusters == 4 else axes
    for cluster_id, ax in enumerate(plot_axes):
        ax.tick_params(axis='both', which='major', labelsize=14)

        if cluster_id < n_clusters:
            patient_df = kaplan_df.loc[kaplan_df['cluster'] == cluster_id]
            means = []

            for vessel_id, vessel in enumerate(chosen_vessels):
                flow_curve = np.stack(patient_df[f'flow_{vessel}'], 1)
                mean_flow = round(np.mean(np.mean(flow_curve, 0)) * 0.06, 1)
                means.append(mean_flow)

                df = pd.DataFrame(flow_curve)
                df['Time'] = range(1, flow_curve.shape[0] + 1)
                df_melted = df.melt(id_vars=['Time'], var_name='Curve', value_name='Flow')

                # Seaborn lineplot
                sns.lineplot(
                    data=df_melted, x="Time", y="Flow", ax=ax,
                    color=ordered_colors[cluster_id + vessel_id * 5],
                    linestyle='-' if vessel_id == 0 else (0, (3, 1)),
                    errorbar='ci', estimator='mean',
                    linewidth=4, label=f'{vessel.upper()}: {mean_flow} L/min/m$^2$'
                )

            ax.set_xlabel('Cardiac Cycle', fontsize=16)
            ax.set_ylabel(r'Flow (mL/s/m$^2$)', fontsize=16)
            
            
            ax.set_title(f'Cluster {cluster_id + 1} ({cluster_names[cluster_id]}, N = {len(patient_df)})', fontsize = 16)#, color = ordered_colors[cluster_id])
            # ax.legend(loc='upper left', ncol = 1, fontsize = 12, framealpha = 0, handlelength = 1)  

            # Retrieve legend handles and labels
            handles, labels = ax.get_legend_handles_labels()

            # Determine legend positions based on the highest mean value
            legend_positions = ('upper center', 'lower center') if np.argmax(means) == 0 else ('lower center', 'upper center')

            # Create and add legends
            first_legend = ax.legend([handles[0]], [labels[0]], loc=legend_positions[0], framealpha=0, fontsize=14, handlelength=2.3)
            second_legend = ax.legend([handles[1]], [labels[1]], loc=legend_positions[1], framealpha=0, fontsize=14, handlelength=2.3)

            ax.add_artist(first_legend)


            ax.set_xlim(1,flow_curve.shape[0])
            max_ylim = 300 if 'ao' in chosen_vessels[0] else 45
            min_ylim = -10 if 'ao' in chosen_vessels[0] else 0
            ax.set_ylim(min_ylim,max_ylim)
            

        else:
            pass

    if n_clusters == 4:
        plot_simon_makuch(axes[2], kaplan_df, ordered_colors, 'liver')
    plot_simon_makuch(axes[-1], kaplan_df, ordered_colors, 'death_transplant')


    # Get the bounds of the middle subplot
    bbox = middle_subplot.get_position()

    sub_label = 'a)' if 'lpa' in chosen_vessels[0] else 'b)'
    if len(chosen_vessels) == 2:
        fig.suptitle(f'{sub_label} {chosen_vessels[0].upper()}/{chosen_vessels[1].upper()} (N = {len(clean_df)} Exams)', fontsize = 18, fontweight = 'bold', x=bbox.x0 + bbox.width / 2, ha='center')

    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.26, hspace = 0.35)

    plt.savefig(f'results/clusters_kaplan_{vessel_pair}.png', bbox_inches='tight', dpi = 600)

    plt.show()




def plot_latent_clusters(latent, cluster_labels, ordered_colors, name):
    s_score = silhouette_score(latent, cluster_labels)

    N = latent.shape[0]
    latent_flat = latent.reshape(N, -1)

    reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    title = 't-SNE'

    latent_2d = reducer.fit_transform(latent_flat)

    # Convert to DataFrame for easier plotting
    df_plot = pd.DataFrame({
        title + " 1": latent_2d[:, 0],
        title + " 2": latent_2d[:, 1],
        'Cluster': cluster_labels
    })

    plt.figure(figsize=(4.5, 4.3))
    ax = sns.scatterplot(
        data=df_plot, 
        x=title + " 1", 
        y=title + " 2", 
        hue="Cluster", 
        palette=ordered_colors[:df_plot['Cluster'].nunique()], 
        alpha=0.7,
        legend = False
    )
    if 'before' in name:
        plt.title(f"Latent Space Before Joint Optimization")
    else:
        plt.title(f"Latent Space After Joint Optimization")

    ax.text(0.5, 0.92, f"Silhouette Score: {s_score:.2f}", fontsize=10, ha='center', transform=ax.transAxes)
    


    for cluster_id in range(df_plot['Cluster'].nunique()):
        coords = df_plot.loc[df_plot['Cluster'] == cluster_id].mean(0).values
        plt.text(
            coords[0], coords[1], 
            f'{cluster_id + 1}', 
            fontsize=14, 
            color='black',
            ha='center', 
            va='center',
            bbox=dict(facecolor=ordered_colors[cluster_id], alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
        )

    plt.tight_layout()
    lim = 70
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    # Save figure
    plt.savefig(f"{name}_projection.png", dpi=600)
    plt.close()

def analyse_cluster(df, variable, name,  ordered_colors):
    df = df[['patient','cluster',variable]].dropna().sort_values('cluster')

    data = [df.loc[df['cluster'] == cluster_id][variable].dropna().values for cluster_id in range(df.cluster.nunique())]
    stat, kruskal_pvalue = kruskal(*data)
    print(f"Kruskal-Wallis Test p-value: {kruskal_pvalue:.4f}")

    posthoc_results = sp.posthoc_dunn(df, val_col=variable, group_col='cluster', p_adjust='fdr_bh')
    # Extract significant pairs
    pairs = []
    p_values = []

    df['cluster'] = df['cluster'].astype(str)
    print(df.cluster.unique())
    # Iterate over the DataFrame (upper triangle)
    for i, group1 in enumerate(posthoc_results.index):
        for j, group2 in enumerate(posthoc_results.columns):
            if i < j:  # Avoid duplicate pairs
                p_value = posthoc_results.loc[group1, group2]
                if p_value < 0.05:  # Consider only significant comparisons
                    pairs.append((str(group1), str(group2)))
                    p_values.append(p_value)

    # Print extracted pairs
    print("Significant pairs:", pairs)
    print("Corresponding p-values:", p_values)


    print("Dunn's Post-hoc test results:")
    print(posthoc_results)
    posthoc_results.to_csv(f'results/posthoc_{name}.csv')

    # Create the boxplot using seaborn
    plt.figure(figsize=(4, 6))
    ax = sns.boxplot(x='cluster', y=variable, data=df, palette = ordered_colors)

    # Add significance annotations
    if pairs:
        annotator = Annotator(ax, pairs, data=df, x="cluster", y=variable)
        annotator.configure(test=None,text_format='star')
        annotator.set_pvalues(p_values)
        annotator.annotate()


    # Print extracted pairs
    print("Significant pairs:", pairs)
    print("Corresponding p-values:", p_values)


    print("Dunn's Post-hoc test results:")
    print(posthoc_results)


    # Add the p-value to the plot title
    title = 'EF' if name == 'EF' else 'Exercise Tolerance'
    variable_name = 'Ejection Fraction' if name == 'EF' else 'Percent Predicted Peak VO2 (%)'
    plt.ylabel(variable_name)
    plt.xlabel('')

    plt.title(f'{title} p-value = {kruskal_pvalue:.4f}')
    plt.savefig(f'results/{name}.png', bbox_inches='tight')
    plt.close()


def fit_cox(covariates, name):
    ctv = CoxTimeVaryingFitter()

    ctv.fit(covariates.dropna(), 
            id_col = 'force_id', 
            event_col = f'status_{name}_varying', 
            start_col=f'start_{name}',
            stop_col= f'stop_{name}')
    ctv_df = ctv.summary
    return ctv_df


def cox_time_varying_corrected(df, name):
    # One-hot encoding the clusters
    n_clusters = df.cluster.nunique()
    df = pd.get_dummies(df, columns=['cluster'])  
    p_df = []

    # List of corrections
    corrections = ['', 'age_at_cmr', 'pt_sex', 'cardiac_output', 'cmr_sv_ef', 
                    ['age_at_cmr', 'pt_sex', 'cardiac_output', 'cmr_sv_ef']]


    for reference_cluster_id in range(n_clusters):
        N_death = df.loc[df[f'cluster_{reference_cluster_id}'] == True][f'status_{name}_varying'].value_counts().get(1, 0) # get death = 1 or if it fails return 0

        cox_df = df.drop(columns = [f'cluster_{reference_cluster_id}'])
        cluster_columns = [col for col in cox_df.columns if 'cluster' in col]
        # Iterate through correction sets
        for correction in corrections:
            if correction:
                if not isinstance(correction, list):
                    correction = [correction]  # Ensure it's a list
            else:
                correction = []

            # Prepare covariates
            covariates = cox_df[['force_id', f'status_{name}_varying', f'start_{name}', f'stop_{name}'] + correction + cluster_columns]

            # Fit the Cox model
            ctv_df = fit_cox(covariates, name)

            # Process results
            for col in cluster_columns:
                new_row = {}
                p = np.round(ctv_df.loc[col, 'p'], 3)
                exp_coef = np.round(ctv_df.loc[col, 'exp(coef)'], 2)
                new_row['reference_cluster'] = reference_cluster_id
                new_row['p'] = p

                new_row['exp_coef'] = exp_coef#f'{exp_coef} (Death = {N_death})'
                new_row['N_death'] = N_death#f'{exp_coef} (Death = {N_death})'

                new_row['correction'] = 'All' if len(correction) > 1 else (correction[0] if correction else 'Uncorrected')
                new_row['cluster'] = col.replace('cluster_', '')

                p_df.append(new_row)

    # Convert to DataFrame and pivot
    p_df = pd.DataFrame.from_records(p_df)
    p_df = p_df.pivot(index=['reference_cluster','cluster'], columns='correction', values=['p','exp_coef','N_death'])

    # Save results
    p_df.to_csv(f'results/time_varying_corrected_cox_{name}.csv')
    return p_df


def plot_heatmap(p_df, vessel_pair, name, correction):
    columns_all = [col for col in p_df.columns if col[1] == correction]
    all_p_df = p_df[columns_all].reset_index()


    heatmap_p = all_p_df.pivot(index='reference_cluster', columns='cluster', values=['p'])
    heatmap_hz = all_p_df.pivot(index='reference_cluster', columns='cluster', values=['exp_coef'])
    heatmap_death = all_p_df.pivot(index='reference_cluster', columns='cluster', values=['N_death'])

    print(heatmap_p.min().min())

    heatmap_hz.columns = [f'Cluster {int(i[-1]) + 1}' for i in heatmap_hz.columns]
    heatmap_hz.index = [f'Cluster {int(i) + 1}' for i in heatmap_hz.index]


    # # Create the mask to only show the upper triangle (upper part of the heatmap)
    mask = np.tril(np.ones_like(heatmap_hz, dtype=bool), k=0)  # Mask lower triangle (everything below the diagonal)

    # Plot the heatmap with hazard ratios (exp_coef) as the values
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(
        heatmap_hz,  
        linewidths = 0.5,
        annot=False, 
        cbar=False,  
        mask=mask,  
    )

    # Move y-ticks (row labels) to the right while keeping alignment
    ax.yaxis.set_label_position("right")  # Move the y-axis label
    ax.yaxis.tick_right()  # Move only the ticks to the right

    # Adjust tick label position for better alignment
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='left')  # Align labels properly

    significant_colors = {'vc':'#6A7D3B','pa':'#D98508','ao':'#2A796D'}
    normal_colors = {'vc':'#E5EAD6','pa':'#FEEED8','ao':'#E1F4F1'}

    # After plotting the heatmap, adjust the colors based on p-values
    for i in range(len(heatmap_p.index)):  # Iterate over rows
        for j in range(len(heatmap_p.columns)):  # Iterate over columns
            p_value = heatmap_p.iloc[i, j]  # Get the p-value from the p pivot table
            death = heatmap_death.fillna(0).iloc[i,j]
            hz = heatmap_hz.iloc[i, j]
            # text = f'{hz}\nN = {int(death)}\n(p = {p_value})'
            text = f'{hz}\nN = {int(death)}\n(p = {p_value})'

            if i > j:  # Mask the lower triangle: Do not color cells below the diagonal
                continue  # Skip coloring for the lower triangle cells (below the diagonal)



            significant_color = significant_colors[vessel_pair]
            normal_color = normal_colors[vessel_pair]
            
            # For upper triangle and diagonal, color the cells based on p-value
            if i == j:  # Check if it's a diagonal element
                plt.gca().add_patch(plt.Rectangle(
                    (j, i), 1, 1, color='white'))  # Set diagonal cells to white
                text_color = 'white'  # Text on diagonal is white
            elif p_value < 0.05:  # If p-value is significant
                plt.gca().add_patch(plt.Rectangle(
                    (j, i), 1, 1, color=significant_color))  
                text_color = 'white'  # Text in significant cells is white
            else:
                plt.gca().add_patch(plt.Rectangle(
                    (j, i), 1, 1, color=normal_color))  
                text_color = 'black'  # Text in non-significant cells is black

            # # # Add text annotation with the correct color
            if not mask[i, j]:  # Only annotate cells in the upper triangle
                plt.text(j + 0.5, i + 0.5, text, 
                            ha='center', va='center', color=text_color)

    # Set labels for axes
    plt.xlabel('')
    plt.tick_params(axis='x', direction='inout', length=6, labeltop=True, labelbottom=False)
    plt.ylabel('Reference', rotation=270, labelpad=20)  

    plt.savefig(f'results/cox_heatmap_{correction}_{name}.png', bbox_inches = 'tight', dpi = 600)
    plt.close()
