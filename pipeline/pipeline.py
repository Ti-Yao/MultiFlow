from utils import *

data_path = '/workspaces/storage/Flow/test_set500'
patients = [pat.split('/')[-1].replace('.gif','') for pat in glob.glob(f'/workspaces/storage/Flow/old_pipeline/segs/FLOW-3/v1 dec_05/*')]

model_name = 'FLOW-3'
version = 'v4_jan_14'
results_path = f'/workspaces/storage/Flow/pipeline'


def read_dicom_header(dicom_files, series):
        '''
        read the information we want from the header and assert that the series has to have pixelarray data
        '''
        dicoms_in_series = dicom_files[series]
        dicom_info = {}
        for dicom_path in dicoms_in_series: # go through dicom in each series
            dcm = pydicom.dcmread(dicom_path, force=True) # read dicom

            try: # if dicom doesn't have an associate pixel array (image), ignore dicom
                image = dcm.pixel_array  
                image_exists = True
                if image.ndim == 3: # ignore dicom if 3d
                    image_exists = False
                try:
                    if dcm.MRAcquisitionType == '3D': # ignore dicom if 3d
                        image_exists = False
                        break
                except:
                    pass

            except Exception as e:
                print('error reading image', e)
                image_exists = False
                break

            if image_exists: # if image exists and is not 3d read all other information
                dicom_info[dicom_path] = {}
                image = dcm.pixel_array.astype('float32')
                try:
                    intercept = dcm.RescaleIntercept
                    slope = dcm.RescaleSlope
                except:
                    try:
                        intercept = list(dcm.RealWorldValueMappingSequence)[0].RealWorldValueIntercept 
                        slope = list(dcm.RealWorldValueMappingSequence)[0].RealWorldValueSlope 
                    except:
                        intercept = 1
                        slope = 1
                image = image * slope + intercept # true pixel values from dicom
                manufacturer = dcm.Manufacturer.lower()
                
                # initialise some variables
                venc = 0
                scale = 1
                vas_flag = 0
                
                if 'siemens' in manufacturer:
                    try:
                        venc = str(dcm[0x0051, 0x1014]._value)
                        numbers = re.findall(r'\d+', venc)
                        venc = float(max(list(map(int, numbers))))
                    except:
                        try:
                            venc = str(dcm[0x0018, 0x0024]._value)
                            venc = float(re.search(r'v(\d+)in', venc).group(1))
                        except:
                            venc = 0
                    image = image.astype('float32')
                    if venc > 0:
                        image = (image * venc)/4096
                        
                if 'ge' in manufacturer:
                    try:
                        venc = dcm[0x0019, 0x10cc].value/10 
                        vas_flag = dcm[0x0043, 0x1032]._value
                        venc_scale = float(dcm[0x0019, 0x10E2]._value)
                        
                        vas_flag = 2 & vas_flag
                        
                        if vas_flag != 0:
                            scale = venc/(venc_scale * np.pi)
                        if vas_flag == 0 and venc >0:
                            image = image/10

                    except:
                        venc = 0
                        
                if 'philips' in manufacturer:
                    try:
                        venc = abs(list(dcm.RealWorldValueMappingSequence)[0].RealWorldValueIntercept)
                    except:
                        try:
                            venc = abs(dcm.RescaleIntercept)
                        except:
                            venc = 0

                dicom_info[dicom_path]['venc'] = float(venc)
                dicom_info[dicom_path]['vas_flag'] = vas_flag
                dicom_info[dicom_path]['scale'] = scale
                dicom_info[dicom_path]['image'] = image
                dicom_info[dicom_path]['uid'] =  '.'.join(dcm.SOPInstanceUID.split('.')[-2:])
                dicom_info[dicom_path]['seriesuid'] =  dcm.SeriesInstanceUID
                dicom_info[dicom_path]['manufacturer'] = manufacturer.lower()
                
                
                rr_ni, rr_hr = 0, 0
                try:
                    rr_ni = round(dcm.NominalInterval,3)
                except Exception as e:
                    rr_ni = 0
                try:
                    rr_hr = round(60000/dcm.HeartRate,3)
                except Exception as e:
                    rr_hr = 0
                rr = np.max([rr_ni, rr_hr])
                rr = rr if (rr > 100) and (rr < 3000) else 0
                dicom_info[dicom_path]['rr'] = rr

                try:
                    dicom_info[dicom_path]['seriesdescription'] = dcm.SeriesDescription.lower()
                except:
                    dicom_info[dicom_path]['seriesdescription'] = ''
                
                try:
                    try:
                        dicom_info[dicom_path]['creationtime'] = convert_time_to_minutes(dcm.InstanceCreationTime)
                    except:
                        dicom_info[dicom_path]['creationtime'] = convert_time_to_minutes(dcm[0x0008, 0x0033].value)
                except:
                    dicom_info[dicom_path]['creationtime'] = 0
                    
                try:
                    dicom_info[dicom_path]['pixelspacing'] = dcm.PixelSpacing
                except:
                    dicom_info[dicom_path]['pixelspacing'] = ''
                try:
                    dicom_info[dicom_path]['triggertime'] = round(dcm.TriggerTime)
                except:
                    dicom_info[dicom_path]['triggertime'] = np.nan
                try:
                    dicom_info[dicom_path]['orientation'] = [round(val,3) for val in dcm.ImageOrientationPatient]
                except:
                    dicom_info[dicom_path]['orientation'] = np.nan
                try:
                    dicom_info[dicom_path]['position'] = [round(val,3) for val in dcm.ImagePositionPatient]
                except:
                    dicom_info[dicom_path]['position'] = np.nan
                try:
                    dicom_info[dicom_path]['slicelocation'] = round(dcm.SliceLocation,3)
                except:
                    dicom_info[dicom_path]['slicelocation'] = np.nan
        return dicom_info


class Flow_Pipeline:
    def __init__(self, patient, data_path, model_path):
        self.patient = patient
        self.data_path = data_path
        self.path = f'{data_path}/{patient}'
        
        self.dicom_info = self.get_dicom_info() # read every dicom file and extract image and metadata
        self.stack_df_list = self.get_stack_df_list(self.dicom_info) # sort into cines
        
        self.model = self.load_models(model_path) # load the models
        
        self.predicted_df = self.get_images(self.stack_df_list) # extract the phase-contract cines
        self.predicted_df, self.num_flows = self.get_predictions(self.predicted_df) # run the MultiFlowSeg model on each image
        self.plot_gif(self.predicted_df) # plot the gifs of all series
        self.predicted_df = self.predicted_df.iloc[:self.num_flows] # choose only the series we want based on criteria
        self.predicted_df = self.calculate_flows(self.predicted_df) # calculate flows on the chosen series
        self.vessel_flows, self.vessel_total_volumes, self.vessel_flow_curves = self.plot_curve(self.predicted_df) # plot flows
        self.predicted_df = self.predicted_df[[col for col in self.predicted_df.columns if 'image' not in col]] # remove images for saving
        
    def load_models(self, model_path):
        ''' load the MultiFlowSeg model'''
        model =  tf.keras.models.load_model(model_path, compile = False) # segmentation part

        # Identify the classification layer
        classification_layer = None
        for layer in model.layers:
            if layer.name == "tf.nn.softmax":  
                classification_layer = layer.output # classification part
                break

        # Create a single model that outputs both segmentation and classification
        model = tf.keras.Model(inputs=model.inputs, outputs=[model.output, classification_layer])
        
        print('model loaded')
        return model
    
    def get_dicom_info(self):
        '''
        puts all the dicom header information for ALL dicoms into a dataframe
        '''
        path = self.path
        series_list = sorted([series for series in glob.glob(f'{path}/*/*') if os.path.isdir(series)] + [series for series in glob.glob(f'{path}/*') if os.path.isdir(series)])
        dicom_files = {}
        new_series_list = []
        for series_path in series_list:
            dicoms_in_series = sorted(glob.glob(f"{series_path}/*.dcm"))
            if len(dicoms_in_series) > min_timesteps and len(dicoms_in_series) < max_timesteps*3: # must have this range of images
                dicom_files[series_path] = dicoms_in_series
                new_series_list.append(series_path)
        series_list = new_series_list
        dicom_info = {}
        for series in series_list:
            dicoms_in_series = read_dicom_header(dicom_files, series)
            dicom_info.update(dicoms_in_series)
        dicom_info = pd.DataFrame.from_dict(dicom_info, orient = 'index').reset_index().rename(columns={'index': 'dicom'}) # put dicom info for all images into a dataframe

        if len(dicom_info) == 0:
            raise ValueError('missing pc dicoms')
        
        dicom_info = dicom_info.drop_duplicates(subset = 'uid') # remove any dicoms that are identicl
        dicom_info = dicom_info.sort_values(by = 'seriesuid')

        self.manufacturer = dicom_info.iloc[0].manufacturer
        dicom_info = dicom_info.drop(columns = ['manufacturer'])
        dicom_info = dicom_info.dropna(subset=['orientation', 'position'])
        dicom_info['orientation'] = dicom_info['orientation'].apply(tuple)
        dicom_info['position'] = dicom_info['position'].apply(tuple)
        return dicom_info
    
    def get_stack_df_list(self, dicom_info):
        # Initialize list to store DataFrames
        stack_df_list = []
        if dicom_info['creationtime'].nunique() > 1: # group based on the time the images were taken and orientation and position 
            time_tolerance = 2
            dicom_info = dicom_info.sort_values(by=['creationtime']).reset_index(drop=True)
            dicom_info['time_diff'] = dicom_info['creationtime'].diff().fillna(0)
            dicom_info['creationtime_label'] = (dicom_info['time_diff'].abs() > time_tolerance).cumsum()
            grouped = dicom_info.groupby(['orientation', 'position', 'creationtime_label'])
            for (orientation, position, creationtime_label), group_df in grouped:
                if len(group_df) > min_timesteps * 2 and (group_df['venc'] != 0).any():
                    sorted_df = group_df.sort_values(['triggertime', 'uid'])
                    stack_df_list.append(sorted_df)
        else: # group based on orientation and position only
            grouped = dicom_info.groupby(['orientation', 'position'])
            for (orientation, position), group_df in grouped:
                if len(group_df) > min_timesteps * 2 and (group_df['venc'] != 0).any():
                    sorted_df = group_df.sort_values(['triggertime', 'uid'])
                    stack_df_list.append(sorted_df)

        if len(stack_df_list) == 0:
            raise ValueError('missing pc series')
        return stack_df_list
    
    def get_images(self, stack_df_list):
        images = []
        vencs = []
        rrs = []
        descriptions = []
        mag_series_uids = []
        phase_series_uids = []
        for stack_df in stack_df_list:
            try:
                if 'ge' in self.manufacturer.lower(): # split phase-contrast into mag and phase
                    mag_df, phase_df = [x for _ , x in stack_df.groupby(stack_df.image.apply(lambda x: x.min()< 0))]
                else:
                    phase_df, mag_df = [x for _ , x in stack_df.groupby(stack_df['venc'] == 0)]
            except:
                continue
            
            mag_df = mag_df.drop_duplicates(subset = ['uid'])
            mag_tt = mag_df.triggertime.unique()
            mag_df = mag_df.loc[mag_df['seriesuid'] == mag_df['seriesuid'].iloc[0]]
            phase_df = phase_df.drop_duplicates(subset = ['uid'])
            phase_df = phase_df[phase_df['triggertime'].isin(mag_tt)]
            phase_df = phase_df.loc[phase_df['venc'] == np.max(phase_df['venc'])]

            if len(phase_df) > len(mag_df):
                selected_seriesuid = phase_df.loc[phase_df['image'].apply(lambda x: np.min(x) < 0), 'seriesuid'].iloc[0]
                phase_df = phase_df.loc[phase_df['seriesuid'] == selected_seriesuid]

            if len(mag_df)!= len(phase_df):
                mag_df = mag_df.groupby('triggertime').apply(assign_group).reset_index(drop=True)
                mag_df = mag_df[mag_df['group'] == 'mag'].drop(columns='group').reset_index(drop=True)
                
            if stack_df.triggertime.nunique() > 0:
                mag_df['image_mean'] = mag_df['image'].apply(np.mean)
                mag_df = mag_df.loc[mag_df.groupby('triggertime')['image_mean'].idxmax()].drop(columns='image_mean')

                phase_df['image_mean'] = phase_df['image'].apply(np.mean)
                phase_df = phase_df.loc[phase_df.groupby('triggertime')['image_mean'].idxmin()].drop(columns='image_mean')

            if len(mag_df)!= len(phase_df):
                continue

            phase_image = np.stack(phase_df['image'], -1)
            mag_image = np.stack(mag_df['image'], -1)

            if 'ge' in self.manufacturer.lower(): # GE needs extra processing for velocity
                vas_flag = phase_df.iloc[0]['vas_flag']
                if vas_flag != 0:
                    scale = phase_df.iloc[0]['scale']
                    velocity = np.divide(phase_image, mag_image, out=np.zeros_like(phase_image, dtype=float), where=mag_image != 0) * scale
                    phase_image = velocity 
                    print('magnitude-weighted')
                    
            ps = float(phase_df.iloc[0].pixelspacing[0])
            mag_image = zoom(mag_image, [ps, ps, 1]) # resize magnitude to 1x1x1mm
            phase_image = zoom(phase_image, [ps, ps, 1]) # resize phase to 1x1x1mm
        
            max_size = 256

            mag_image = tf.image.resize_with_crop_or_pad(mag_image, max_size, max_size) # crop or pad images to 256x256
            phase_image = tf.image.resize_with_crop_or_pad(phase_image, max_size, max_size)  # crop or pad images to 256x256

            image = np.stack([mag_image, phase_image], -1) # combine magnitude and phase together

            # save information to lists
            images.append(image)
            vencs.append(phase_df.venc.unique()[0])
            rrs.append(phase_df.rr.unique()[0])
            descriptions.append(mag_df.seriesdescription.unique()[0])
            mag_series_uids.append(mag_df.seriesuid.unique()[0])
            phase_series_uids.append(phase_df.seriesuid.unique()[0])

        # save information lists to a prediction_df dataframe
        predicted_df = pd.DataFrame()
        predicted_df['image'] = images
        predicted_df['venc'] = vencs
        predicted_df['rr'] = rrs
        predicted_df['description'] = descriptions
        predicted_df['mag_series_uid'] = mag_series_uids
        predicted_df['phase_series_uid'] = phase_series_uids
        predicted_df = predicted_df.loc[predicted_df['rr']> 0]
        return predicted_df
    
    def run_inference(self,mag_image, phase_image, venc, description):
        '''
        Runs model inference on the inputted image
        '''
        frames = mag_image.shape[-1]
        ratio = image_frames/frames
        mag_image = zoom(mag_image, (0.5,0.5,ratio)) # resize images to model input size = 128x128x32
        phase_image = zoom(phase_image, (0.5,0.5,ratio)) # resize images to model input size = 128x128x32

        mag_image[mag_image<1e-10] = 0                
        angles = phase2angle(phase_image, venc) # convert phase to angles
        mag_image = (mag_image - np.min(mag_image))/(np.max(mag_image))
        mag_image[mag_image>=1] = 1

        mag_image = skimage.exposure.equalize_adapthist(mag_image) # CLAHE the magnitude
        complex_image = create_complex_image(mag_image, angles) # create real and imaginary components from the mag and angle
        real_image, imaginary_image = complex_image[...,0],complex_image[...,1] # split components
        mag_image = normalise(mag_image) 
        imaginary_image = normalise(imaginary_image)        

        X = np.stack([mag_image, imaginary_image], -1).astype('float32')[np.newaxis] # input
        y = np.zeros((image_size, image_size, image_frames, 6), dtype='uint8')[np.newaxis] # dummy input
        cgm_input = tf.zeros((6))[np.newaxis] # dummy input 

        if description == '': # if description is empty
            label = ''
            one_hot_input = tf.one_hot(0, 6)[np.newaxis] 

        else:
            description = description.replace('_',' ').replace('.',' ').replace('x','').replace('  ',' ').split(' ')
            print(description)
            labels = is_token_a_substring_in_dictionary(data_dictionary, description) 
            if len(labels) == 0:
                label = 0
            else:
                labels = pd.Series(labels)
                if (labels == 'other').any():
                    label = 'other'
                else:
                    label = labels.value_counts().index[0]
            
            one_hot = vessels_dict[label] if label in vessels_dict.keys() else 0 # tunable input 
            one_hot_input = tf.one_hot(one_hot, 6)[np.newaxis] 
        
        print(label, description, one_hot_input)

        y_pred, probability = self.model.predict({'image_input':X, 'cgm_input': cgm_input,'one_hot_input':one_hot_input,'mask_input': y})
        y_pred = y_pred[-1][0] # get the vessel segmentation
        vessel_index = np.argmax(np.sum(y_pred, axis = (0,1,2))[1:], -1) + 1 # get the vessel index
        vessel = vessels_dict_r[vessel_index] # the predicted vessel

        probability = round(probability[...,vessel_index][0], 2) # get the vessel classification probablity
  
        mask = get_one_hot(np.argmax(y_pred,axis = -1), 6)[...,vessel_index] # binarise mask

        mask = zoom(mask, (2,2,1/ratio)) # reshape mask to original size 
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        mask = clean_mask(mask, vessel) # clean/post-process the mask to remove weries bits
        return mask, vessel, probability, label
        
    
    def select_flows(self,predicted_df):
        '''
        Decide which flows are chosen out of all the ones segmented
        '''
        predicted_df = predicted_df.sort_values('probability')
        keep_indices = []
        available_vessels = predicted_df['vessel'].unique()
        for vessel in vessels:
            if vessel in available_vessels: # loop through vessels that were classified
                vessel_predicted_df = predicted_df.loc[predicted_df['vessel'] == vessel]
                vessel_predicted_df = vessel_predicted_df.loc[vessel_predicted_df['probability']>0.5] # must have greater than 0.5 probability
                vessel_predicted_df = vessel_predicted_df.loc[vessel_predicted_df['mask'].apply(lambda x: np.sum(x) > 0)] # must have non-empty segmentation mask
               
                if len(vessel_predicted_df) > 1: # if more than one series were classified for the given vessel
                    if len(vessel_predicted_df.loc[vessel_predicted_df['dict_label'] == vessel]) > 0:
                        vessel_predicted_df = vessel_predicted_df.loc[vessel_predicted_df['dict_label'] == vessel]
                    vessel_descriptions = list(vessel_predicted_df['description']) # descriptions of each series
                    vessel_lookup = description_match_dict[vessel] 
                    # match the predicted label with the description
                    matches = find_substring_matches(vessel_lookup, vessel_descriptions)
                    if len(matches) > 0: # get the indices of all the matches
                        indices = np.unique(
                                  list(chain.from_iterable(vessel_predicted_df.loc[vessel_predicted_df['description'] == match].index.tolist() 
                                  for match in matches)))
                    else:
                        indices = vessel_predicted_df.index.tolist() # if there are no matches
                    
                    vessel_predicted_df = vessel_predicted_df.loc[indices] # select series with matches unless there are no matches in which case treat all predictions equally
                    vessel_predicted_df = vessel_predicted_df.loc[vessel_predicted_df['probability'] == vessel_predicted_df['probability'].max()] # choose the match with the highest probability
                    vessel_predicted_df = vessel_predicted_df.loc[vessel_predicted_df['num_mask_component'] == vessel_predicted_df['num_mask_component'].max()] # choose the match with the most number of mask components (for double SVCs)
                    vessel_predicted_df = vessel_predicted_df.sort_values('quality_metric', ascending = False) # sort values by segmentation quality
                    max_index = vessel_predicted_df.index[0] # select highest classficiation probablity with best segmentation quality
                    keep_indices.append(max_index)           
                elif len(vessel_predicted_df)==1:
                    keep_indices.append(vessel_predicted_df.index[0])
                else:
                    pass

        new_order = keep_indices+ [i for i in predicted_df.index if i not in keep_indices]
        predicted_df = predicted_df.loc[new_order].reset_index(drop=True) # sort the predicted_df by chosen vessels
        num_flows = len(keep_indices) # number of vessels chosen
        return predicted_df, num_flows
    
    
    def get_predictions(self, predicted_df):
        '''
        Loop through each series and run model inference
        '''
        images = predicted_df['image'].values
        descriptions = predicted_df['description'].values
        vencs = predicted_df['venc'].values
        rrs = predicted_df['rr'].values
        
        masks = []
        vessels = []
        probabilities = []
        dict_labels = []
        quality_metrics = []
        num_mask_components = []
        for rr, venc, image, description in zip(rrs, vencs, images, descriptions):

            mag_image, phase_image = image[...,0], image[...,1]
            
            mask, vessel, probability,label = self.run_inference(mag_image, phase_image, venc, description)
            quality_metric = calculate_segmentation_quality_metric(mask)
            _, num_mask_component = ndimage.label(mask)
            print(num_mask_component)

            masks.append(mask)
            vessels.append(vessel)
            probabilities.append(probability)
            dict_labels.append(label)
            quality_metrics.append(quality_metric)
            num_mask_components.append(num_mask_component)
        predicted_df['vessel'] = vessels
        predicted_df['probability'] = probabilities
        predicted_df['mask'] = masks
        predicted_df['dict_label'] = dict_labels
        predicted_df['quality_metric'] = quality_metrics
        predicted_df['num_mask_component'] = num_mask_components
        
        predicted_df, num_flows = self.select_flows(predicted_df) # select the best flows
        return predicted_df, num_flows

    def calculate_flows(self,predicted_df):
        '''Calculate flows from segmentation masks'''
        images = predicted_df['image'].values
        rrs = predicted_df['rr'].values
        masks = predicted_df['mask'].values
        vessels = predicted_df['vessel'].values
        flows = []
        total_volumes = []
        forward_volumes = []
        backward_volumes = []
        flow_curves = []
        for rr, image,mask, vessel in zip(rrs, images, masks, vessels):
            mag_image, phase_image = image[...,0], image[...,1]
        
            flow_curve = calculate_curve(mask, phase_image, vessel) # flow (cm3/s) by seconds
            flow_curve = interpolate_curve(flow_curve, rr) # interpolate (cm3/s) by milliseconds

            flow = np.mean(flow_curve) * 0.06 # effective flow (L/min)
            total_volume = np.sum(flow_curve)/1000 # total volume (cm3)
            forward_volume = np.sum(flow_curve[flow_curve>0])/1000 # forward volume (cm3)
            backward_volume = np.sum(flow_curve[flow_curve<0])/1000 # backward volume (cm3)

            forward_volume = abs(forward_volume)
            backward_volume = abs(backward_volume)

            # Make sure the forward and backward are in the right direction
            if backward_volume > forward_volume:
                forward_volume, backward_volume = backward_volume, forward_volume
            
            flows.append(flow)
            total_volumes.append(total_volume)
            forward_volumes.append(forward_volume)
            backward_volumes.append(backward_volume)
            flow_curves.append(flow_curve)
            
        # save flows to predicted_df
        predicted_df['flow'] = flows
        predicted_df['total_volume'] = total_volumes
        predicted_df['forward_volume'] = forward_volumes
        predicted_df['backward_volume'] = backward_volumes
        predicted_df['flow_curve'] = flow_curves
        return predicted_df
    
    def plot_gif(self, predicted_df):
        '''Plot GIFs of segmentations for QA'''
        images = predicted_df['image'].values
        masks = predicted_df['mask'].values
        descriptions = predicted_df['description'].values
        probabilities = predicted_df['probability'].values
        dict_labels = predicted_df['dict_label'].values
        
        num_tiles = len(images)

        # Now find rows and columns for this aspect ratio.
        grid_rows = int(np.sqrt(num_tiles) + 0.5)  # Round.
        grid_cols = (num_tiles + grid_rows - 1) // grid_rows     # Ceil.        
        fig, axes = plt.subplots(grid_rows,grid_cols,figsize = (grid_cols*3.9,grid_rows*4.5))
        if len(images) > 1:
            for ax in axes.ravel():
                ax.tick_params(axis='both', which='both', length=0, labelsize=0)
                ax.grid(visible=False)
                axes = axes.ravel()
        else:
            axes = [axes]

        gif_frames = []
        timesteps = np.max([im.shape[-2] for im in images])
        for time in range(0,timesteps):
            ttl = plt.text(0.75, -0.05, f'timestep = {time + 1}/{timesteps}', horizontalalignment='center', verticalalignment='bottom', transform=axes[-1].transAxes, fontsize=16)
            artists = [ttl]
            for pos in range(len(axes)):
                ax = axes[pos]
                if pos < len(predicted_df):
                    vessel = predicted_df.loc[pos].vessel

                    image = images[pos][...,0] # get magnitude
                    mask = masks[pos]
                    
                    description = descriptions[pos]
                    probability = probabilities[pos]
                    dict_label = dict_labels[pos]
                    
                    ax.patch.set_facecolor('white')
                    if time >= image.shape[-1]:
                        time = -1
                    p1 = ax.imshow(image[...,time],cmap = 'gray', vmin = np.min(image), vmax = np.max(image))
                    p2 = ax.imshow(mask[...,time],alpha = mask[...,time] * 0.5, cmap=colormaps[vessel])

                    if pos < self.num_flows: # highlight the chosen series
                        for spine in ax.spines.values():
                            spine.set_edgecolor(colormaps[vessel].colors[0])
                            spine.set_linewidth(5)
                    # show the classification (label), the data dictionary tunable input (dict), the probability (prob) and the series description
                    title = ax.set_title(f'label = {vessel}, dict = {dict_label}, prob = {probability:.2f}\ndescription = {description}')
                    artists.extend([p1,p2, title])
                else:
                    ax.axis('off')
            gif_frames.append(artists)
            legend_patches = [mpatches.Patch(color=plt.cm.get_cmap(colormaps[label])(0.5), label=label) for label in colormaps.keys()]
            fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=15, bbox_to_anchor=(0.5, 0))
            plt.subplots_adjust(top = 0.94, bottom = 0.06, right = 0.99, left = 0.01, hspace = 0.15, wspace = 0)
        ani = animation.ArtistAnimation(fig, gif_frames)
        Path(f'{results_path}/segs/{model_name}/{version}').mkdir(parents=True, exist_ok=True)
        ani.save(f'{results_path}/segs/{model_name}/{version}/{patient}.gif', fps=image.shape[0]/2, writer = 'pillow')
        plt.close()
        

    def plot_curve(self, predicted_df):
        '''Plot the flow curves for each chosen vessel'''
        images = predicted_df['image'].values
        masks = predicted_df['mask'].values
        vessels = predicted_df['vessel'].values
        vencs = predicted_df['venc'].values
        flows = predicted_df['flow'].values
        total_volumes = predicted_df['total_volume'].values
        flow_curves = predicted_df['flow_curve'].values
        
        vessel_flows = {}
        vessel_total_volumes = {}
        vessel_flow_curves = {}

        fig = plt.figure(figsize = (9,6))
        for i, vessel in enumerate(vessels):
            ax = fig.add_subplot(2,3, i + 1)
            mask = masks[i]
            
            phase_image = images[i][...,1]
            vessel = vessels[i]
            venc = vencs[i]
            flow = flows[i]
            total_volume = total_volumes[i]
            flow_curve = flow_curves[i]
            
            vessel_flows[vessel] = flow
            vessel_total_volumes[vessel] = total_volume
            vessel_flow_curves[vessel] = flow_curve
            
            ax.plot(flow_curve, linewidth = 3, label = f'Tot volume, {vessel.upper()}: {total_volume:.1f}'+ 'mL', c = colormaps[vessel].colors[0])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), shadow=True, ncol=2)
            ax.set_ylabel('Flow (mL/s)')
            ax.set_xlabel('Time (ms)')
            
        Path(f'{results_path}/flow_curve_plot/{version}').mkdir(parents=True, exist_ok=True)
        plt.subplots_adjust(wspace = 0.3,hspace = 0.4, top = 0.94, bottom = 0.06, right = 0.99, left = 0.01)
        plt.savefig(f'{results_path}/flow_curve_plot/{version}/{patient}.png', bbox_inches = 'tight')
        plt.close()
        return vessel_flows, vessel_total_volumes, vessel_flow_curves
        
csv_file = f'data_log ({version}).csv'
if os.path.exists(csv_file):
    sv_df = pd.read_csv(csv_file)
else:
    sv_df = pd.DataFrame(columns=['patient','venc', 'rr', 'description', 'mag_series_uid', 'phase_series_uid',
                               'vessel', 'probability', 'dict_label', 'flow',
                               'total_volume', 'forward_volume', 'backward_volume', 'flow_curve'])
    
model_name = 'FLOW-3'
model_path = '../training/models/FLOW-3.h5'


for patient in tqdm(patients[:5]):
    # if patient not in sv_df.patient.unique():
    try:
        print(patient)
        p = Flow_Pipeline(patient,data_path, model_path)

        df = p.predicted_df[['venc', 'rr', 'description', 'mag_series_uid', 'phase_series_uid',
                            'vessel', 'probability', 'dict_label', 'flow',
                            'total_volume', 'forward_volume', 'backward_volume','flow_curve']]
        df['patient'] = patient
        sv_df = pd.concat([sv_df, df])
        sv_df = sv_df.drop_duplicates(subset = ['patient','vessel'], keep = 'last')
        sv_df.to_csv(csv_file, index = False)
        sv_df = pd.read_csv(csv_file)
        del p
    except Exception as e:
        print(patient, e)