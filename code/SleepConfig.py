class SleepConfig:
    RAW_EEG_PATH = '../PSG'
    LABEL_PATH = '../tag'
    PROCESSED_EEG_PATH = '../processed_eeg_data'

    TARGET_CHANNEL = 'EEG Fpz-Cz'
    WINDOW_SEC = 30
    SAMPLING_RATE = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 50
    
    # 模型配置
    MODEL_CONFIG = {
        'input_size': 3000,  # 30秒 ＊ 100Hz
        'n_channels': 1,
        'n_classes': 5
    }
    
    STAGE_WINDOW_SEC = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4,
    }