import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import scipy.signal

def analyze_audio_events(file_path):
    """
    오디오 파일의 시간 및 주파수 영역에서 특정 이벤트의 발생 횟수를 분석하고 시각화합니다.

    Args:
        file_path (str): 분석할 오디오 파일의 경로.
    """
    try:
        # 1. 스타일 적용 및 오디오 파일 로드
        plt.style.use(hep.style.ROOT)
        y, sr = librosa.load(file_path, sr=None)
        print(f"오디오 파일 로드 완료 (샘플링 속도: {sr} Hz, 길이: {len(y)/sr:.2f}초)")

        # --- 사용자 조정 필요 구간 ---
        # 시간 영역: 진폭 임계값 (0과 1 사이의 값)
        AMP_THRESHOLD = 0.15
        # 주파수 영역: 피크 감지 민감도 (0~1 사이의 값, 높을수록 뚜렷한 소리만 감지)
        PEAK_HEIGHT_THRESHOLD = 0.2
        # --- 사용자 조정 필요 구간 끝 ---

        # 2. 시간 영역 분석: 임계값 초과 횟수 계산
        y_normalized = np.abs(y) / np.max(np.abs(y))
        crossings = np.where(np.diff(np.sign(y_normalized - AMP_THRESHOLD)) > 0)[0]
        min_dist = int(0.1 * sr)
        if len(crossings) > 0:
            time_event_indices = [crossings[0]]
            for i in range(1, len(crossings)):
                if crossings[i] - time_event_indices[-1] > min_dist:
                    time_event_indices.append(crossings[i])
            time_event_count = len(time_event_indices)
        else:
            time_event_count = 0
        
        print(f"\n--- 분석 결과 ---")
        print(f"시간 영역: 진폭이 {AMP_THRESHOLD}를 초과한 횟수: {time_event_count}회")

        # 3. 주파수 영역 분석 (STFT 및 FFT 기반)
        D = np.abs(librosa.stft(y))
        
        # 전체 주파수 대역의 총 에너지 계산
        total_energy = np.sum(D, axis=0)
        total_energy_normalized = (total_energy - np.min(total_energy)) / (np.max(total_energy) - np.min(total_energy))
        
        # 피크(이벤트) 탐지
        peaks, _ = scipy.signal.find_peaks(total_energy_normalized, height=PEAK_HEIGHT_THRESHOLD, distance=int(0.5 * sr / (D.shape[1] / (len(y)/sr))))
        freq_event_count = len(peaks)
        print(f"주파수 영역 (전체): 소리 이벤트 발생 횟수: {freq_event_count}회")

        # 전체 주파수 히스토그램 계산
        fft_result = np.abs(np.fft.fft(y))
        freqs_fft = np.fft.fftfreq(len(y), d=1/sr)

        # 4. 결과 시각화 (4개의 서브플롯)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [1, 2, 1.5, 2]})
        
        # 시간 영역 파형
        times = librosa.times_like(y, sr=sr)
        ax1.plot(times, y, label="Audio Waveform", color='royalblue', alpha=0.7)
        ax1.axhline(y=AMP_THRESHOLD * np.max(np.abs(y)), color='r', linestyle='--', label=f'Amplitude Threshold ({AMP_THRESHOLD})')
        ax1.axhline(y=-AMP_THRESHOLD * np.max(np.abs(y)), color='r', linestyle='--')
        ax1.set_title(f'Time-Domain Waveform (Threshold Crossings: {time_event_count})')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.legend()

        # 스펙트로그램
        S_db = librosa.amplitude_to_db(D, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax2)
        ax2.set_title('Spectrogram')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        
        # 주파수 영역 히스토그램 (FFT)
        positive_freq_indices = np.where(freqs_fft >= 0)
        ax3.plot(freqs_fft[positive_freq_indices], fft_result[positive_freq_indices], label="Frequency Magnitude", color='crimson')
        ax3.set_title('Frequency Histogram (FFT)')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Magnitude')
        ax3.grid(True)
        ax3.legend()

        # 전체 주파수 대역 에너지
        stft_times = librosa.times_like(D[0], sr=sr)
        ax4.plot(stft_times, total_energy_normalized, label='Total Spectral Energy', color='darkorange', alpha=0.7)
        ax4.plot(stft_times[peaks], total_energy_normalized[peaks], "x", markersize=10, color='crimson', label=f'Detected Events ({freq_event_count})')
        ax4.set_title('Total Spectral Energy Over Time')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Normalized Energy')
        ax4.legend()

        plt.tight_layout()
        plt.savefig('full_audio_analysis_total_freq.png')
        print("\n전체 분석 그래프를 'full_audio_analysis_total_freq.png' 파일로 저장했습니다.")

    except ImportError:
        print("필수 라이브러리가 설치되지 않았습니다. 'pip install librosa matplotlib numpy scipy mplhep'를 실행해주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

# --- 코드 실행 ---
if __name__ == '__main__':
    audio_file = '30bg.m4a'
    analyze_audio_events(audio_file)
