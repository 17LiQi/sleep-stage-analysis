import pyedflib

def inspect_edf_metadata(edf_path):
    try:
        edf = pyedflib.EdfReader(edf_path)
        print(f"\n文件: {edf_path}")
        print(f"通道数: {edf.signals_in_file}")
        for ch in range(edf.signals_in_file):
            print(f"通道 {ch}: 标签: {edf.getLabel(ch)}, 预滤波: {edf.getPrefilter(ch)}, 采样率: {edf.getSampleFrequency(ch)} Hz")
        edf.close()
    except Exception as e:
        print(f"读取失败: {e}")

inspect_edf_metadata('../../PSG/ST7072J0-PSG.edf')
