import mne

# 读取注释文件,后面可以批量读取而不是手动保存了
# 那实际上仅进行数据操作和处理不需要Polyman查看
annotations_file = '../Hypnogram/ST7011JP-Hypnogram.edf'
annotations = mne.read_annotations(annotations_file)

print("Annotations raw data:")
for annot in annotations:
    print(annot)

# 字典格式
# Annotations raw data:
# OrderedDict([('onset', 0.0), ('duration', 1560.0), ('description', 'Sleep stage W'), ('orig_time', None)])
# OrderedDict([('onset', 1560.0), ('duration', 90.0), ('description', 'Sleep stage 1'), ('orig_time', None)])
# OrderedDict([('onset', 1650.0), ('duration', 570.0), ('description', 'Sleep stage 2'), ('orig_time', None)])
# OrderedDict([('onset', 2220.0), ('duration', 120.0), ('description', 'Sleep stage 1'), ('orig_time', None)])
