from inference_utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def main(type):
    with open('patients5.json', 'r') as json_file:
        patients = json.load(json_file)

    train_patients, val_patients, test_patients = patients['train'],patients['val'],patients['test']
    patients = train_patients +  val_patients + test_patients 
    venc_df = pd.read_csv('venc.csv')
    series_description_df = pd.read_csv('seriesdescription.csv').set_index(['patient','vessel'])


    # data_path = f'resources/clean'
    data_path = f'/workspaces/Flow/data/clean'
    model_names = {'actual':'FLOW-24','empty':'FLOW-21','random':'FLOW-23','vanilla':'FLOW-26'}

    inference_model_name = 'FLOW-3' if type != 'vanilla' else 'FLOW-26'
    model_name = model_names[type]


    csv_file = f'results/segmentation_{model_name}.csv'

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=['patient', 'vessel', 'dice', 'pred_vessel'])

    if type != 'vanilla':
        model = get_model(inference_model_name)
    else:
        model =  tf.keras.models.load_model(f'models/{inference_model_name}.h5', compile = False)


    for patient in tqdm(test_patients[:2]):
        print(patient)
        for vessel in vessels:
            vessel_index = vessels_dict[vessel]
            mag_image, phase_image, true_mask = np.load(f'{data_path}_{image_size}_{image_frames}/{patient}_{vessel}.npy', allow_pickle=True)
            true_mask[true_mask > 0.5] = 1
            true_mask[true_mask <= 0.5] = 0
            true_mask = true_mask.astype('uint8')
            
            y_pred, pred_label = inference(model, mag_image, phase_image, patient, venc_df, vessel, series_description_df, type)

            print(pred_label)
            Path(f'results/{model_name}/masks/').mkdir(parents=True, exist_ok=True)
            np.save(f'results/{model_name}/masks/{patient}_{vessel}.npy', y_pred)

            dice_val = single_dice(true_mask, y_pred[..., vessel_index])

            # Check if the row exists
            row_index = df[(df['patient'] == patient) & (df['vessel'] == vessel)].index

            if not row_index.empty:
                # Update existing row
                df.loc[row_index, ['dice', 'pred_vessel']] = [dice_val, pred_label]
            else:
                # Append new row
                df = pd.concat([df, pd.DataFrame([{
                    'patient': patient, 'vessel': vessel, 'dice': dice_val, 'pred_vessel': pred_label
                }])], ignore_index=True)

            # Save the updated DataFrame
            df.to_csv(csv_file, index=False)

    
    # Load segmentation data
    segmentation_df = (
        pd.read_csv(f'results/segmentation_{model_name}.csv')
        .drop_duplicates(['patient', 'vessel'], keep='last')
        .set_index(['patient', 'vessel'])
    )

    results = []

    # Iterate through test patients
    for patient in tqdm(test_patients[:2]):
        for vessel in vessels:
            # Load predicted and true masks
            pred_mask = np.load(f'results/{model_name}/masks/{patient}_{vessel}.npy')
            _, phase_image, true_mask = np.load(f'{data_path}/{patient}_{vessel}.npy', allow_pickle=True)
            
            # Extract relevant vessel mask
            pred_mask = pred_mask[..., vessels_dict[vessel]]

            # Adjust time dimension scaling
            time_factor = true_mask.shape[-1] / pred_mask.shape[-1]
            pred_mask = zoom(pred_mask, (2, 2, time_factor))

            # Apply thresholding
            pred_mask = (pred_mask >= 0.5).astype(int)

            # Compute Dice coefficient
            dice_val = single_dice(true_mask, pred_mask)

            # Retrieve RR value
            rr = venc_df.loc[(venc_df['patient'] == patient) & (venc_df['vessel'] == vessel), 'rr'].iloc[0]

            # Compute velocity curves
            true_v_curve = interpolate_curve(calculate_curve(true_mask, phase_image, vessel), rr)
            pred_v_curve = interpolate_curve(calculate_curve(pred_mask, phase_image, vessel), rr)

            # Compute area using Simpson's rule
            true_area = simpson(true_v_curve) / 1000
            pred_area = simpson(pred_v_curve) / 1000

            # Retrieve Dice coefficient from segmentation data
            dice_val = segmentation_df.loc[(patient, vessel), 'dice']

            # Store results
            results.append({
                'patient': patient,
                'vessel': vessel,
                'true': true_area,
                'pred': pred_area,
                'dice': dice_val
            })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(f'results/{model_name}_SV.csv', index=False)

if __name__ == "__main__":
    # main(type = 'actual')
    main(type = 'vanilla')
    # main(type = 'random')
    # main(type = 'empty')