import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep # mplhep 라이브러리 import

def plot_time_and_frequency(file_path):
    """
    오디오 파일의 시간 영역 파형과 주파수 영역 히스토그램을 시각화합니다.

    Args:
        file_path (str): 분석할 오디오 파일의 경로.
    """
    try:
        # 1. mplhep 스타일 적용 (ROOT 스타일로 변경)
        plt.style.use(hep.style.ROOT)
        print("mplhep ROOT 스타일을 적용했습니다.")

        # 2. 오디오 파일 로드
        y, sr = librosa.load(file_path, sr=None)
        print(f"오디오를 성공적으로 불러왔습니다.")
        print(f"샘플링 속도: {sr} Hz, 오디오 길이: {len(y)/sr:.2f}초")

        # 3. 푸리에 변환 (FFT) 수행
        fft_result = np.abs(np.fft.fft(y))
        
        # 4. FFT 결과에 해당하는 주파수 축 생성
        freqs = np.fft.fftfreq(len(y), d=1/sr)

        # 5. 결과 시각화 (2개의 서브플롯 생성)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 2]})
        
        # --- 첫 번째 그래프: 시간 영역 파형 ---
        times = librosa.times_like(y, sr=sr)
        ax1.plot(times, y, label="Audio Waveform", color='royalblue')
        ax1.set_title('Time-Domain Waveform')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        ax1.legend()

        # --- 두 번째 그래프: 주파수 영역 히스토그램 ---
        positive_freq_indices = np.where(freqs >= 0)
        ax2.plot(freqs[positive_freq_indices], fft_result[positive_freq_indices], label="Frequency Magnitude", color='crimson')
        ax2.set_title('Frequency Histogram (FFT)')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        # 6. 그래프 파일로 저장
        plt.savefig('time_and_frequency_analysis.png')
        print("시간 및 주파수 분석 그래프를 'time_and_frequency_analysis.png' 파일로 저장했습니다.")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

# --- 코드 실행 ---
if __name__ == '__main__':
    # mplhep 라이브러리가 없는 경우 설치 안내
    try:
        import mplhep
    except ImportError:
        print("mplhep 라이브러리가 설치되지 않았습니다. 터미널에 'pip install mplhep'를 입력하여 설치해주세요.")
    else:
        audio_file = '10(down).m4a'
        plot_time_and_frequency(audio_file)
