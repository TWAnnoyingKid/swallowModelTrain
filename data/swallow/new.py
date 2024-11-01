import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import math
import sys

# 定義位置編碼類別
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)  # [max_len, d_model/2]
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

# 定義改進後的 AudioClassifier 類別
class AudioClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=4, hidden_dim=128, dropout=0.1, max_seq_len=160):
        super(AudioClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # 輸入線性層，將輸入映射到 hidden_dim 維度
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # 位置編碼
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim, max_len=max_seq_len)

        # Transformer 編碼器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2048,  # 確保與訓練時一致
            dropout=dropout,
            activation='relu',
            batch_first=True  # 設置為 True 以匹配輸入維度
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分類線性層
        self.output_fc = nn.Linear(hidden_dim, num_classes)

        # Dropout 層
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            Tensor of shape [batch_size, num_classes]
        """
        # 通過輸入線性層
        x = self.input_fc(x)  # [batch_size, seq_len, hidden_dim]

        # 添加位置編碼
        x = self.positional_encoding(x)  # [batch_size, seq_len, hidden_dim]

        # Transformer 編碼器
        x = self.transformer_encoder(x)  # [batch_size, seq_len, hidden_dim]

        # 池化：取平均
        x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]

        # Dropout
        x = self.dropout(x)  # [batch_size, hidden_dim]

        # 分類層
        x = self.output_fc(x)  # [batch_size, num_classes]

        return x

# 定義梅爾頻譜圖提取函數
def extract_mel_spectrogram(audio_segment, n_mels=128, hop_length=512, n_fft=1024, sample_rate=16000):
    """
    將音頻段轉換為梅爾頻譜圖並正規化。
    
    Args:
        audio_segment (np.ndarray): 音頻段數據。
        n_mels (int): 梅爾頻帶數。
        hop_length (int): Hop 長度。
        n_fft (int): FFT 大小。
        sample_rate (int): 取樣率。
    
    Returns:
        torch.Tensor: 處理後的梅爾頻譜圖張量 [seq_len, n_mels]
    """
    try:
        # 提取梅爾頻譜圖
        mel_spec = librosa.feature.melspectrogram(
            y=audio_segment, 
            sr=sample_rate, 
            n_mels=n_mels, 
            hop_length=hop_length, 
            n_fft=n_fft
        )
        # 轉換為分貝
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # 轉置以匹配模型輸入形狀 [seq_len, n_mels]
        mel_spec_db = mel_spec_db.T
        # 正規化到 [0, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
        return torch.tensor(mel_spec_db, dtype=torch.float32)
    except Exception as e:
        print(f"Error in extract_mel_spectrogram: {e}")
        return None

# 偵測函數實現
def detect_swallow_in_1s_segments(audio_path, model, device, class_names, n_mels=128, hop_length=512, n_fft=1024, sample_rate=16000, segment_duration=1):
    """
    偵測音頻文件中 1 秒段內的吞嚥次數。
    
    Args:
        audio_path (str): 音頻文件路徑。
        model (nn.Module): 已訓練好的模型。
        device (torch.device): 設備（CPU 或 GPU）。
        class_names (list): 類別名稱列表，如 ['normal', 'swallow']。
        n_mels (int): 梅爾頻帶數。
        hop_length (int): Hop 長度。
        n_fft (int): FFT 大小。
        sample_rate (int): 取樣率。
        segment_duration (int): 每段音頻的時長（秒）。
        
    Returns:
        int: 檢測到的吞嚥次數。
    """
    try:
        # 載入整個音頻文件
        audio, sr = librosa.load(audio_path, sr=sample_rate)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return 0

    segment_length = int(sample_rate * segment_duration)  # 1 秒段的樣本數
    total_segments = len(audio) // segment_length
    swallow_count = 0

    for i in range(total_segments):
        start_sample = i * segment_length
        end_sample = start_sample + segment_length
        audio_segment = audio[start_sample:end_sample]

        # 預處理音頻段
        feature = extract_mel_spectrogram(
            audio_segment, 
            n_mels=n_mels, 
            hop_length=hop_length, 
            n_fft=n_fft, 
            sample_rate=sample_rate
        )
        if feature is None:
            print(f"Skipping segment {i+1} due to preprocessing error.")
            continue

        # 增加批次維度並移動到設備
        feature = feature.unsqueeze(0).to(device)  # [1, seq_len, n_mels]

        # 模型推理
        with torch.no_grad():
            outputs = model(feature)  # [1, num_classes]
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted = predicted.item()
            confidence = confidence.item()
            if class_names[predicted] == "swallow":
                swallow_count += 1
                print(f"Swallow detected in segment {i+1} with confidence {confidence:.2f}")

    print(f"檢測到的吞嚥次數：{swallow_count}")
    return swallow_count

# 示例使用
if __name__ == "__main__":
    # 定義常量
    SAMPLE_RATE = 16000  # 與訓練時一致
    SEGMENT_DURATION = 1  # 1秒段
    SEGMENT_LENGTH = int(SAMPLE_RATE * SEGMENT_DURATION)  # 1秒段的樣本數

    # 定義類別名稱
    class_names = ['normal', 'swallow']  # 根據你的數據集調整

    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用設備: {device}')

    # 初始化模型
    input_dim = 1984  # 根據訓練時的設定
    hidden_dim = 128
    num_heads = 4
    num_layers = 4  # 根據訓練時的設定
    num_classes = len(class_names)
    dropout = 0.1
    max_seq_len = 160  # 根據你的梅爾頻譜圖時間步數設定

    model = AudioClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        dropout=dropout,
        max_seq_len=max_seq_len
    ).to(device)

    # 加載模型權重
    model_save_path = 'model'  # 請替換為你的實際路徑
    model_file = os.path.join(model_save_path, 'audioModel.pth')

    if os.path.exists(model_file):
        try:
            # 建議設置 weights_only=True 以符合未來的 torch.load 預設
            state_dict = torch.load(model_file, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"成功載入模型權重：{model_file}")
        except Exception as e:
            print(f"載入模型權重時出錯：{e}")
            sys.exit(1)
    else:
        print(f"模型權重文件不存在：{model_file}")
        sys.exit(1)

    # 偵測函數使用示例
    audio_file = '音檔/比較輕吞.wav'  # 請根據實際情況調整路徑
    if os.path.exists(audio_file):
        swallow_count = detect_swallow_in_1s_segments(
            audio_path=audio_file,
            model=model,
            device=device,
            class_names=class_names,
            n_mels=128,
            hop_length=512,
            n_fft=1024,
            sample_rate=16000,
            segment_duration=1
        )
        print(f"總吞嚥次數：{swallow_count}")
    else:
        print(f"音頻文件不存在：{audio_file}")
