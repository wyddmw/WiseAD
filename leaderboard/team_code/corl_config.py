import os

class GlobalConfig:
    """ base architecture configurations """
	# Data
    seq_len = 5 # input timesteps
    # use different seq len for image and lidar
    img_seq_len = 1 
    lidar_seq_len = 1
    pred_len = 6 # future waypoints planned
    scale = 1 # image pre-processing
    img_width = 320 # important this should be consistent with scale, e.g. scale = 1, img_width 320, scale=2, image_width 640
    img_resolution = (336, 336) # image resoluation for evaluation
    start_task = 0 # the start task id for training
    num_tasks = 3 #number of tasks, 1 for joint training
    image_type = 'segmentation'
    # image_type = 'rgb'
    # model_path = '/media/spyder/aa8541c6-90ed-457f-b3e1-f9cbab57a2352/MobileVLM_Drive_models/carla_text_planning/LingoQA_carla_simpleQA_v3/mobilevlm_v2-2.finetune'
    # model_path = '/media/spyder/aa8541c6-90ed-457f-b3e1-f9cbab57a2352/MobileVLM_Drive_models/carla_text_planning/LingoQA_carla_simpleQA/mobilevlm_v2-2.finetune'
    # model_path = '/media/spyder/aa8541c6-90ed-457f-b3e1-f9cbab57a2352/MobileVLM_Drive_models/carla_text_planning/carla_lingoqa_v3_epoch5/checkpoint-22000'    
    # model_path = '/media/spyder/aa8541c6-90ed-457f-b3e1-f9cbab57a2352/MobileVLM_Drive_models/carla_text_planning/carla_lingoqa_joint_epoch2_simpleqa/mobilevlm_v2-2.finetune'
    # model_path = '/media/spyder/aa8541c6-90ed-457f-b3e1-f9cbab57a2352/MobileVLM_Drive_models/carla_text_planning/LingoQA_carla_simpleQA_simpledqa/mobilevlm_v2-2.finetune'
    model_path = '/media/spyder/aa8541c6-90ed-457f-b3e1-f9cbab57a2352/MobileVLM_Drive_models/carla_text_planning/carla_lingoqa_joint_epoch1_simpleqa/mobilevlm_v2-2.finetune'
    # Prompt Parameter
    pool_size = 9
    prompt_name = 'drive' #'l2p' #dual #coda #hippocampus #drive
    prompt_depth = 1 # for l2p, 0 for l2p and 1 for l2p++
    e_prompt_len = 10 #2
    g_prompt_len = 10 #10
    top_k = 1 #3
    ortho = 1 # 1 for true, 0 for false
    attended_p = 1 # 1 for true, 0 for false
    
    # Vision Encoder Parameter
    patch_size = 16 #14 #16
    encoder_embed_dim = 192 #1280 #1024 #1024 #192 #768 #192 for vit_tiny_patch16_224
    depth = 12 #32 #24 #24 #12 for tiny
    num_heads = 3 #16 #12 # 3 for vit_tiny_patch16_224
    ckpt_layer = 0
    drop_path_rate = 0
    
    # Language Decoder Parameter
    state_dim = 6 #10
    act_dim = 9 # overall action space (route decusion)
    embed_len = 3 #4 for foundation; 3 for vlm
    num_lat_classes = 6 # lateral action space
    num_long_classes = 3 # longitudinal aciton space
    num_regressions = 2
    decoder_embed_dim = 1408 #256
    hidden_dim = 1600 #768 #1600 #1280 #768 #192 
    max_emb_size = 4096
    n_layer = 48 #12 #48 #36 #12
    n_head = 25 #12 #25 #20 #12 #4
    dropout = 0.1

    # Architecture Parameter
    encoder = 'ViT-Tiny' #'CLIP-ViT-Large'
    decoder = 'GPT2-xl' #'GPT2-xl'
    mode = 'onboard' #onboard for bfloat16, otherwise float32
    
    ## wandb parameters
    dont_log_wandb = True
    # wandb_entity = "OscarHuang"
    # wandb_entity = "spyderzsy"
    project = "VLMDrive-Dual-CoreLanguage-All"
    wandb_group = "GPT2XL"

    # Domian Randomization
    augment = True
    inv_augment_prob = 0.1 # Probablity that data augmentation is applied is 1.0 - inv_augment_prob
    aug_max_rotation = 20 # degree
    
    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40  # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40  # buffer size

    # Carla
    weather = 'ClearNoon'
    max_throttle = 0.75  # upper limit on throttle signal value in dataset
    brake_speed = 0.1  # desired speed below which brake is triggered
    brake_ratio = 1.1  # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.35  # maximum change in speed input to logitudinal controller

    max_speed = 5
    collision_buffer = [2.5, 1.2]
    momentum = 0
    skip_frames = 2 #2
    detect_threshold = 0.04

    def __init__(self, root_dir='', setting='all', **kwargs):
        self.root_dir = root_dir
        if (setting == 'all'): # All towns used for training no validation data
            self.train_towns = os.listdir(self.root_dir)
            self.val_towns = [self.train_towns[0]]
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
                for file in root_files:
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        self.train_data.append(os.path.join(self.root_dir, town, file))
            for town in self.val_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town))
                for file in root_files:
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        self.val_data.append(os.path.join(self.root_dir, town, file))

        elif (setting == '02_05_withheld'): #Town02 and 05 withheld during training
            print("Skip Town02 and Town05")
            self.train_towns = os.listdir(self.root_dir) #Scenario Folders
            self.val_towns = self.train_towns # Town 02 and 05 get selected automatically below
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
                for file in root_files:
                    if ((file.find('Town02') != -1) or (file.find('Town05') != -1)):  #We don't train on 05 and 02 to reserve them as test towns
                        continue
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        print("Train Folder: ", file)
                        self.train_data.append(os.path.join(self.root_dir, town, file))
            for town in self.val_towns:
                root_files = os.listdir(os.path.join(self.root_dir, town))
                for file in root_files:
                    if ((file.find('Town02') == -1) and (file.find('Town05') == -1)): # Only use Town 02 and 05 for validation
                        continue
                    if not os.path.isfile(os.path.join(self.root_dir, file)):
                        print("Val Folder: ", file)
                    self.val_data.append(os.path.join(self.root_dir, town, file))
        elif (setting == '3-town'):
            print('Train on three different road structures: urban(Town3), highway(Town4), rural(Town7)')
            self.train_towns = os.listdir(self.root_dir)
            self.val_towns = self.train_towns
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                if ((town.find('town03') == -1) and (town.find('town04') == -1) and (town.find('town07') == -1)):  #We don't train on 05 and 02 to reserve them as test towns
                        continue
                root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
                for file in root_files:
                    # if ((file.find('weather-0') == -1) and (file.find('weather-3') == -1) and (file.find('weather-6') == -1)):
                    if file.find('weather-0') == -1:
                        continue
                    subroot_files = os.listdir(os.path.join(self.root_dir, town, file))
                    for subfile in subroot_files:
                        if (subfile.find('data') == -1):
                            continue
                        if not os.path.isfile(os.path.join(self.root_dir, town, file, subfile)):
                            print("Train Folder: ", self.root_dir, town, file, subfile)
                            self.train_data.append(os.path.join(self.root_dir, town, file, subfile))

            for town in self.val_towns:
                if ((town.find('town03') == -1) and (town.find('town04') == -1) and (town.find('town07') == -1)):  #We don't train on 05 and 02 to reserve them as test towns
                        continue
                root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
                for file in root_files:
                    # if ((file.find('weather-0') == -1) and (file.find('weather-3') == -1) and (file.find('weather-6') == -1)):
                    if file.find('weather-0') == -1:
                        continue
                    subroot_files = os.listdir(os.path.join(self.root_dir, town, file))
                    for subfile in subroot_files:
                        if (subfile.find('validation') == -1):
                            continue
                        if not os.path.isfile(os.path.join(self.root_dir, town, file, subfile)):
                            print("Validation Folder: ", self.root_dir, town, file, subfile)
                            self.val_data.append(os.path.join(self.root_dir, town, file, subfile))
                            
        # elif (setting == '3-town'):
        #     print('Train on three different road structures: urban(Town3), highway(Town4), rural(Town7)')
        #     self.train_towns = os.listdir(self.root_dir)
        #     self.val_towns = self.train_towns
        #     self.train_data, self.val_data = [], []
        #     for town in self.train_towns:
        #         if ((town.find('town03') != -1) and (town.find('town04') != -1) and (town.find('town07') != -1)):  #We don't train on 05 and 02 to reserve them as test towns
        #                 continue
        #         root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
        #         for file in root_files:
        #             if ((file.find('weather-0') == -1) and (file.find('weather-3') == -1) and (file.find('weather-6') == -1)):
        #                 continue
        #             if not os.path.isfile(os.path.join(self.root_dir, file)):
        #                 print("Train Folder: ", file)
        #                 self.train_data.append(os.path.join(self.root_dir, town, file))

        #     for town in self.val_towns:
        #         if ((town.find('town03') != -1) and (town.find('town04') != -1) and (town.find('town07') != -1)):  #We don't train on 05 and 02 to reserve them as test towns
        #                 continue
        #         root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
        #         for file in root_files:
        #             if ((file.find('weather-0') == -1) and (file.find('weather-3') == -1) and (file.find('weather-6') == -1)):
        #                 continue
        #             if not os.path.isfile(os.path.join(self.root_dir, file)):
        #                 print("Val Folder: ", file)
        #                 self.val_data.append(os.path.join(self.root_dir, town, file))
        
            print(self.train_data, '\n', self.val_data, '\n')
        
        elif (setting == 'min-3-town'):
            print('Training mode with minimum town dataset')
            self.train_towns = os.listdir(self.root_dir)
            self.val_towns = self.train_towns
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                if ((town.find('town03') == -1) and (town.find('town04') == -1) and (town.find('town07') == -1)):  #We don't train on 05 and 02 to reserve them as test towns
                    continue
                # if ((town.find('town03') == -1)):  #We don't train on 05 and 02 to reserve them as test towns
                #     continue
                root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
                for file in root_files:
                    if (file.find('weather-debug') == -1):
                        continue
                    subroot_files = os.listdir(os.path.join(self.root_dir, town, file))
                    for subfile in subroot_files:
                        if (subfile.find('data') == -1):
                            continue
                        if not os.path.isfile(os.path.join(self.root_dir, town, file, subfile)):
                            print("Train Folder: ", self.root_dir, town, file, subfile)
                            self.train_data.append(os.path.join(self.root_dir, town, file, subfile))
                            
            for town in self.val_towns:
                if ((town.find('town03') == -1) and (town.find('town04') == -1) and (town.find('town07') == -1)):  #We don't train on 05 and 02 to reserve them as test towns
                    continue
                # if ((town.find('town03') == -1)):  #We don't train on 05 and 02 to reserve them as test towns
                #     continue
                root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
                for file in root_files:
                    if (file.find('weather-debug') == -1):
                        continue
                    subroot_files = os.listdir(os.path.join(self.root_dir, town, file))
                    for subfile in subroot_files:
                        if (subfile.find('validation') == -1):
                            continue
                        if not os.path.isfile(os.path.join(self.root_dir, town, file, subfile)):
                            print("Validation Folder: ", self.root_dir, town, file, subfile)
                            self.val_data.append(os.path.join(self.root_dir, town, file, subfile))

        # elif (setting == 'debug-town'):
        #     print('Debug mode with minimum town dataset')
        #     self.train_towns = os.listdir(self.root_dir)
        #     self.val_towns = self.train_towns
        #     self.train_data, self.val_data = [], []
        #     for town in self.train_towns:
        #         if ((town.find('town03') != -1) and (town.find('town04') != -1) and (town.find('town07') != -1)):  #We don't train on 05 and 02 to reserve them as test towns
        #                 continue
        #         root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
        #         for file in root_files:
        #             if ((file.find('weather-debug') == -1)):
        #                 continue
        #             if not os.path.isfile(os.path.join(self.root_dir, file)):
        #                 print("Train Folder: ", file)
        #                 self.train_data.append(os.path.join(self.root_dir, town, file))

        #     for town in self.val_towns:
        #         if ((town.find('town03') != -1) and (town.find('town04') != -1) and (town.find('town07') != -1)):  #We don't train on 05 and 02 to reserve them as test towns
        #                 continue
        #         root_files = os.listdir(os.path.join(self.root_dir, town)) #Town folders
        #         for file in root_files:
        #             if ((file.find('weather-debug') == -1)):
        #                 continue
        #             if not os.path.isfile(os.path.join(self.root_dir, file)):
        #                 print("Val Folder: ", file)
        #                 self.val_data.append(os.path.join(self.root_dir, town, file))
        elif (setting == 'eval'): #No training data needed during evaluation.
            pass
        else:
            print("Error: Selected setting: ", setting, " does not exist.")

        for k,v in kwargs.items():
            setattr(self, k, v)
