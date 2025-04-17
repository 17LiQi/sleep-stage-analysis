class SleepConfig:
    RAW_EEG_PATH = '../PSG'
    LABEL_PATH = '../tag'
    PROCESSED_EEG_PATH = '../processed_eeg_data'

    TARGET_CHANNEL = 'EEG Fpz-Cz'
    WINDOW_SEC = 30
    SAMPLING_RATE = 100

    STAGE_WINDOW_SEC = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4,
    }