# Utilities
from utils import *
import sys

def main(vessel_pair, 
         n_clusters, 
         seed, 
         cluster_init, 
         dist_metric, 
         model_name, 
         pool_size, 
         gamma,
         continue_training, 
         inference_only, 
         pretrain_only,
         load_pretrain, 
         ae_model_name):
    set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    colors = {'lpa':['#FF8357','#E45825','#B83100','#BE4417'],
            'rpa':['#FAC172','#F69813','#D98508','#DC901E'],
            'ao':['#89D5C9','#59C5B4','#2A796D','#DC901E'],
            'svc':['#ADC965','#94B540','#576A25','#DC901E'],
            'ivc':['#CFBAE1','#AD8DCE','#523172','#DC901E']}
    ordered_colors = [colors[key][i] for i in range(len(colors['lpa'])) for key in list(colors.keys())]

    hostname = os.getenv("HOSTNAME", "Unknown")

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('models'):
        os.mkdir('models')


    if continue_training:
        run = neptune.init_run(
        # project="ti-yao/DTC",
        project="CFD/Deep-Temporal-Clustering",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2Nzk0ZmU4Zi00YjQxLTQ1YTMtOWU2Ny0xMjU0YjY5ZTU2OWUifQ==",
        with_id = model_name,
        source_files=["*.ipynb", "*.py","*.csv"])  
        
    else:
        run = neptune.init_run(
        # project="ti-yao/DTC",
        project="CFD/Deep-Temporal-Clustering",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2Nzk0ZmU4Zi00YjQxLTQ1YTMtOWU2Ny0xMjU0YjY5ZTU2OWUifQ==",
        source_files=["*.ipynb", "*.py","*.csv"])  
        model_name = list(run.__dict__.values())[-10]




    curve_points = 30
    vessel_pair_dict = {'pa':['lpa','rpa'], 'vc':['svc','ivc'],'all':['lpa','rpa','svc','ivc'],'ao':['ao']}
    chosen_vessels = vessel_pair_dict[vessel_pair]

    vessel_df = pd.read_csv(f'data_{vessel_pair}.csv')
    curve_df = pd.read_csv('curve_data.csv')
    vessel_curve_df = curve_df.merge(vessel_df, on = 'patient')

    # vessel_curve_df['file'] = vessel_curve_df['file'].apply(lambda x: x.replace(docker_path, real_path)) # working locally



    clean_df = []
    for patient in vessel_curve_df.patient.unique()[:]:
        patient_df = vessel_curve_df.loc[vessel_curve_df['patient'] == patient]
        for vessel in chosen_vessels:
            patient_vessel_df = patient_df.loc[patient_df['vessel'] == vessel]
            bsa = patient_vessel_df.cmr_pt_bsa_hidden.values[0]
            vessel_flow = np.load(patient_vessel_df.file.values[0])
            vessel_flow = interpolate_curve(vessel_flow, curve_points)/bsa
            status_death = patient_vessel_df.status_death.values[0]
            clean_df.append({'patient':patient, 'vessel':vessel,'flow':vessel_flow,'status_death':status_death})
    clean_df = pd.DataFrame.from_records(clean_df).fillna(0)
    clean_df = pd.pivot_table(clean_df, values='flow', index=['patient','status_death'],columns=['vessel'])

    patients = clean_df.index.get_level_values(0).values


    X = []
    for vessel in chosen_vessels:
        X.append(np.stack(clean_df[vessel].values, 0))
    X = np.stack(X, axis = -1)

    # X = X / np.sum(X, axis=(-2, -1), keepdims=True)

    status_death = clean_df.index.get_level_values(1).values

    print(X.shape)
    X_mean = np.mean(X)
    X_scaled = (X - X_mean)/np.std(X)

    print(f'Mean = {X_mean:.2f}')


    pretrain_optimizer = 'adam'
    tae_version = 2


    lr = 1e-3
    pt_lr = 1e-3
    run['learning_rate'] = lr
    run['pretraining_learning_rate'] = pt_lr
    
    run['cluster_init'] = f'{cluster_init}:{dist_metric}'
    run['dist_metric']  = dist_metric
    run['tae_version'] = tae_version
    run['N'] = X.shape[0]
    run['n_clusters'] = n_clusters  
    run['vessel'] = vessel_pair
    run['pool_size'] = pool_size
    run['Host'] = hostname
    run['GPU'] = get_true_gpu_ids()
    run['seed'] = seed



    # Instantiate model
    dtc = DTC(n_clusters=n_clusters,
                input_dim=X_scaled.shape[-1],
                timesteps=X_scaled.shape[1],
                run = run,
                model_name = model_name,
                tae_version = tae_version,
                vessel_pair = vessel_pair,
                n_filters=50,
                kernel_size=10,
                strides=1,
                pool_size=pool_size,
                n_units=[50,1],#len(vessel_pair)],
                alpha=1,
                dist_metric=dist_metric,
                cluster_init=cluster_init, # kmeans, hierarchical
                heatmap=False)


    pretrain_epochs = 100
        
    batch_size = 64
    save_dir = 'models'

    # Initialize model
    dtc.initialize()
    dtc.model.summary()

    if continue_training and not pretrain_only: 
    
        run[f'{model_name}_{vessel_pair}'].download(f'{save_dir}/{model_name}_{vessel_pair}.h5')
        dtc.load_weights(f'{save_dir}/{model_name}_{vessel_pair}.h5')

    else:
        if load_pretrain:
            dtc.load_ae_weights(f'models/{ae_model_name}_ae.h5')
            dtc.init_cluster_weights(X_scaled)

        else:
            # pretrain
            dtc.pretrain(X=X_scaled, optimizer=pretrain_optimizer, run = run, 
                epochs=pretrain_epochs, batch_size=batch_size, learning_rate = pt_lr,
                save_dir=save_dir, model_name = model_name)

            # Initialize clusters
            dtc.init_cluster_weights(X_scaled)


    if not continue_training:
        output, q = dtc.model.predict(X_scaled)
        cluster_labels = q.argmax(axis=1)
        latent = dtc.encode(X_scaled)
        np.save('latent_before.npy',latent)
        run['latent_before'].upload('latent_before.npy')
        plot_latent(latent, name = 'latent_before', run = run)
        plot_average_clusters(cluster_labels, X, 'clusters_before', chosen_vessels, run, colors)
        plot_latent_clusters(latent, cluster_labels, ordered_colors, projection_type= 'tsne', name = 'tsne_before', run = run)
        plot_latent_clusters(latent, cluster_labels, ordered_colors, projection_type= 'umap', name = 'umap_before', run = run)


    if not inference_only and not pretrain_only:
        # Evaluate
        output, q = dtc.model.predict(X_scaled)
        plot_autoencoder_output(X_scaled, output, name = 'autoencoder_output1', run = run, chosen_vessels = chosen_vessels)

        run['gamma'] = gamma
        dtc.compile(gamma=gamma, optimizer=tf.keras.optimizers.Adam(learning_rate = lr))

        t0 = time()
        dtc.fit(X_scaled, epochs = 3000, patience = 0, batch_size = batch_size, tol = 0.001, save_dir = save_dir)
        print('Training time: ', (time() - t0))

    
    if not pretrain_only:
        best_dtc = copy.copy(dtc)
        best_dtc.load_weights(f'{save_dir}/{model_name}_{vessel_pair}.h5')

    output, q = dtc.model.predict(X_scaled)
    cluster_labels = q.argmax(axis=1)

    latent = dtc.encode(X_scaled)

    np.save('latent.npy',latent)
    run['latent'].upload('latent.npy')

    plot_latent(latent, name = 'latent', run = run)
    plot_latent_clusters(latent, cluster_labels, ordered_colors, projection_type= 'tsne', name = 'tsne', run = run)
    plot_latent_clusters(latent, cluster_labels, ordered_colors, projection_type= 'umap', name = 'umap', run = run)

    
    plot_autoencoder_output(X_scaled, output, name = 'autoencoder_output2', run = run, chosen_vessels = chosen_vessels)


    centroids = np.array([latent[cluster_labels == i].mean(axis=0) for i in range(n_clusters)])
    print(f'centroids shape: {centroids.shape}')



    vessel_cluster_dict = {k:int(v) for k, v in zip(patients,cluster_labels)}


    cluster_df = pd.DataFrame(list(vessel_cluster_dict.items()), columns=["patient", "cluster"])

    cluster_df.to_csv('results/cluster_df.csv', index = False)
    run['cluster_df'].upload('results/cluster_df.csv')

    df = pd.read_csv(f'data_{vessel_pair}.csv').merge(cluster_df, on = 'patient')

    plot_average_clusters(cluster_labels, X, 'clusters', chosen_vessels, run, colors)
    analyse_cluster(df, 'est_vo2_pp','exercise', run, ordered_colors)
    analyse_cluster(df, 'cmr_sv_ef','EF',run, ordered_colors)
    plot_cluster_kaplan(clean_df, chosen_vessels, run, ordered_colors, vessel_pair, df)

    # List of names for the different categories
    categories = ['death', 'death_transplant','liver']

    # Loop over the categories
    for name in categories:
        for correction in ['All','Uncorrected']:
            p_df = cox_time_varying_corrected(df, run, name)
            plot_heatmap(p_df, vessel_pair, run, name=name, correction=correction)
        plot_simon_makuch(None, df, run, ordered_colors, name)
  
    if not inference_only:
        s_score = silhouette_score(latent, cluster_labels)
        run['silhouette_score'] = s_score
        print(f'Silhouette Score: {s_score:.2f}')

    run.stop()    



if __name__ == "__main__":
    for n_clusters in [4, 5, 6, 7, 3, 8]:
        for vessel_pair in ['ao']:
            # try:
            main(vessel_pair = vessel_pair, 
                n_clusters = n_clusters, 
                seed = 42, 
                gamma = 0.5,
                cluster_init = 'kmeans', 
                dist_metric = 'eucl', 
                model_name=None,#f'DTC-{model}', 
                pool_size = 3, 
                continue_training = 0, 
                inference_only = 0, 
                pretrain_only=0, 
                load_pretrain = 0,
                ae_model_name = None)
            # except:
            #     pass

