from multiflowdtc_utils import *

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


    curve_points = 30
    vessel_pair_dict = {'pa':['lpa','rpa'], 'vc':['svc','ivc'],'all':['lpa','rpa','svc','ivc'],'ao':['ao']}
    chosen_vessels = vessel_pair_dict[vessel_pair]

    vessel_df = pd.read_csv(f'data_{vessel_pair}.csv')
    curve_df = pd.read_csv('curve_data.csv')
    vessel_curve_df = curve_df.merge(vessel_df, on = 'patient')


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

    status_death = clean_df.index.get_level_values(1).values

    print(X.shape)
    X_mean = np.mean(X)
    X_scaled = (X - X_mean)/np.std(X)

    print(f'Mean = {X_mean:.2f}')


    pretrain_optimizer = 'adam'
    tae_version = 2


    lr = 1e-3
    pt_lr = 1e-3

    # Instantiate model
    dtc = DTC(n_clusters=n_clusters,
                input_dim=X_scaled.shape[-1],
                timesteps=X_scaled.shape[1],
                model_name = model_name,
                tae_version = tae_version,
                vessel_pair = vessel_pair,
                n_filters=50,
                kernel_size=10,
                strides=1,
                pool_size=pool_size,
                n_units=[50,1],
                alpha=1,
                dist_metric=dist_metric,
                cluster_init=cluster_init, # kmeans, hierarchical
                heatmap=False)


    pretrain_epochs = 100
        
    batch_size = 64
    save_dir = 'models'

    dtc.initialize()
    dtc.model.summary()

    if continue_training and not pretrain_only: 
        dtc.load_weights(f'{save_dir}/{model_name}_{vessel_pair}.h5')
    else:
        if load_pretrain:
            dtc.load_ae_weights(f'models/{ae_model_name}_ae.h5')
            dtc.init_cluster_weights(X_scaled)

        else:
            # pretrain
            dtc.pretrain(X=X_scaled, optimizer=pretrain_optimizer, 
                epochs=pretrain_epochs, batch_size=batch_size, learning_rate = pt_lr,
                save_dir=save_dir, model_name = model_name)

            # Initialize clusters
            dtc.init_cluster_weights(X_scaled)


    if not continue_training:
        output, q = dtc.model.predict(X_scaled)
        cluster_labels = q.argmax(axis=1)
        latent = dtc.encode(X_scaled)
        np.save('latent_before.npy',latent)
        plot_latent_clusters(latent, cluster_labels, ordered_colors,  name = 'tsne_before')


    if not inference_only and not pretrain_only:
        # Evaluate
        output, q = dtc.model.predict(X_scaled)
        plot_autoencoder_output(X_scaled, output, name = 'autoencoder_output_before', chosen_vessels = chosen_vessels)

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

    plot_latent_clusters(latent, cluster_labels, ordered_colors, name = 'tsne')
    plot_autoencoder_output(X_scaled, output, name = 'autoencoder_output', chosen_vessels = chosen_vessels)

    vessel_cluster_dict = {k:int(v) for k, v in zip(patients,cluster_labels)}


    cluster_df = pd.DataFrame(list(vessel_cluster_dict.items()), columns=["patient", "cluster"])

    cluster_df.to_csv('results/cluster_df.csv', index = False)

    df = pd.read_csv(f'data_{vessel_pair}.csv').merge(cluster_df, on = 'patient')

    analyse_cluster(df, 'est_vo2_pp','exercise', ordered_colors)
    analyse_cluster(df, 'cmr_sv_ef','EF',ordered_colors)
    plot_cluster_kaplan(clean_df, chosen_vessels, ordered_colors, vessel_pair, df)

    categories = ['death_transplant','liver']

    # Loop over the categories
    for name in categories:
        for correction in ['All','Uncorrected']:
            p_df = cox_time_varying_corrected(df, name)
            plot_heatmap(p_df, vessel_pair, name=name, correction=correction)
        plot_simon_makuch(None, df, ordered_colors, name)
  
    if not inference_only:
        s_score = silhouette_score(latent, cluster_labels)
        print(f'Silhouette Score: {s_score:.2f}')



if __name__ == "__main__":
    for n_clusters in range(3,8):
        for vessel_pair in ['pa','vc']:
            main(vessel_pair = vessel_pair, 
                n_clusters = n_clusters, 
                seed = 42, 
                gamma = 0.5,
                cluster_init = 'kmeans', 
                dist_metric = 'eucl', 
                model_name=f'DTC-{vessel_pair}-{n_clusters}',
                pool_size = 3, 
                continue_training = 0, 
                inference_only = 0, 
                pretrain_only=0, 
                load_pretrain = 0,
                ae_model_name = None)
