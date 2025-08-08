# ===================================================================
# 1. 필수 라이브러리 불러오기 (Import)
# ===================================================================
# 오디오 분석 및 처리를 위한 핵심 라이브러리
import librosa
import librosa.display

# 수치 계산 및 배열 관리를 위한 필수 라이브러리
import numpy as np

# 그래프 시각화를 위한 라이브러리
import matplotlib.pyplot as plt

# Matplotlib 그래프에 고에너지물리(HEP) 스타일을 적용하여 미려하게 만듭니다 (선택 사항).
import mplhep as hep

# 신호 처리, 특히 피크(peak)를 찾는 데 사용됩니다.
import scipy.signal

# 불필요한 경고 메시지를 숨겨서 출력을 깔끔하게 합니다.
import warnings
import os

# -------------------------------------------------------------------
# 경고 메시지 무시 설정
warnings.filterwarnings("ignore", category=UserWarning)

# ===================================================================
# 2. 메인 분석 함수 정의
# ===================================================================
def analyze_geiger_clicks(file_path):
    """
    가이거 계수기 오디오 파일(.m4a 등)을 분석하여 클릭(이벤트) 횟수를 계산하고,
    4개의 상세 그래프(파형, FFT, 스펙트로그램, Onset 검출)로 시각화합니다.

    Args:
        file_path (str): 분석할 오디오 파일의 경로
    """
    try:
        # --- 스타일 적용 및 파일 로드 ---
        try:
            plt.style.use(hep.style.ROOT)
            print("mplhep 스타일 적용 완료.")
        except:
            print("mplhep 스타일 적용 실패. 기본 스타일을 사용합니다.")

        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            print(f"오류: 파일을 찾을 수 없습니다: {file_path}")
            return

        # librosa.load가 특정 오디오 포맷(e.g., .m4a)에서 불안정할 수 있으므로,
        # 이 코드에서는 이전 버전의 pydub 로딩 방식을 주석으로 남겨두고,
        # 현재는 librosa.load를 직접 사용합니다. (FFmpeg이 잘 설치된 환경에서는 잘 동작함)
        y, sr = librosa.load(file_path, sr=None)
        duration_minutes = len(y) / sr / 60
        print(f"오디오 파일 로드 완료 (샘플링 속도: {sr} Hz, 길이: {duration_minutes:.2f}분)")

        # 오디오 데이터가 비어 있는지 확인
        if len(y) == 0:
            print("오디오 파일이 비어 있습니다.")
            return

        # ===========================================================
        # 3. 사용자 조정 핵심 파라미터 (튜닝 변수)
        # 이 값들을 조절하여 탐지 정확도를 높일 수 있습니다.
        # ===========================================================

        # (1) PRE_AMP: 사전 증폭 (기본값: 1.0)
        # 녹음된 소리가 너무 작을 경우, 이 값을 1.0보다 크게 설정하여 전체 신호를 증폭시킬 수 있습니다.
        PRE_AMP = 1.0

        # (2) N_FFT: STFT(단시간 푸리에 변환) 윈도우 크기 (주파수 해상도)
        # 소리를 시간 단위로 잘라 분석할 때, 한 조각의 크기.
        # 값이 클수록 주파수 해상도는 높아지지만, 시간 해상도는 낮아집니다.
        # '틱' 소리처럼 짧은 소리를 분석할 때는 1024나 512 정도의 낮은 값이 유리합니다.
        N_FFT = 1024

        # (3) HOP_LENGTH: STFT 윈도우 이동 간격 (시간 해상도)
        # 분석창을 얼마나 촘촘하게 이동시킬지 결정. 값이 작을수록 더 정밀하게 시간 변화를 봅니다.
        # N_FFT의 1/4 또는 1/2 값을 주로 사용합니다.
        HOP_LENGTH = 256

        # (4) PEAK_HEIGHT_THRESHOLD: 피크 검출 민감도 (0.0 ~ 1.0)
        # '클릭'으로 인정할 최소한의 세기(강도). 가장 중요한 튜닝 파라미터입니다.
        # - 값을 높이면: 더 크고 명확한 클릭만 감지합니다 (둔감, 잡음 제거에 효과적).
        # - 값을 낮추면: 더 작은 소리 변화도 클릭으로 감지합니다 (민감, 신호 누락 방지).
        PEAK_HEIGHT_THRESHOLD = 0.6

        # (5) MIN_PEAK_DISTANCE_SECONDS: 최소 이벤트 간격 (초)
        # 하나의 '틱' 소리가 여러 번으로 잘못 카운트되는 것을 방지합니다.
        # 여기에 설정된 시간보다 짧은 간격으로 발생한 피크들은 하나로 처리됩니다.
        MIN_PEAK_DISTANCE_SECONDS = 0.05

        # --- 파라미터 적용 및 신호 처리 ---
        # 사전 증폭 적용
        y = y * PRE_AMP

        # ===========================================================
        # 4. 핵심 분석: 스펙트럼 플럭스 계산 및 피크 탐지
        # ===========================================================

        # (1) Onset Strength(시작점 강도) 계산
        # librosa.onset.onset_strength는 스펙트럼 플럭스(Spectral Flux)를 계산합니다.
        # 스펙트럼 플럭스란, 주파수 스펙트럼이 시간에 따라 얼마나 급격하게 변하는지를 나타내는 척도입니다.
        # '틱' 소리가 나는 순간, 소리의 주파수 구성이 급변하므로 이 값이 급증합니다.
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)

        # 계산된 Onset 강도를 0과 1 사이로 정규화하여, PEAK_HEIGHT_THRESHOLD를 직관적으로 적용할 수 있게 합니다.
        if np.max(onset_env) > 0:
            onset_env_normalized = onset_env / np.max(onset_env)
        else:
            onset_env_normalized = onset_env

        # (2) 피크(클릭 이벤트) 찾기
        # 최소 이벤트 간격(초)을 STFT 프레임 수로 변환합니다.
        frame_rate = sr / HOP_LENGTH
        distance_in_frames = max(1, int(MIN_PEAK_DISTANCE_SECONDS * frame_rate))

        # scipy.signal.find_peaks 함수를 사용하여 정규화된 Onset 강도 곡선에서 피크를 찾습니다.
        # - height: PEAK_HEIGHT_THRESHOLD보다 높은 피크만 찾습니다.
        # - distance: distance_in_frames 이상 떨어진 피크만 찾습니다.
        peaks, _ = scipy.signal.find_peaks(onset_env_normalized, height=PEAK_HEIGHT_THRESHOLD, distance=distance_in_frames)

        # 피크가 감지된 프레임 위치를 시간(초)으로 변환합니다.
        onset_times = librosa.frames_to_time(peaks, sr=sr, hop_length=HOP_LENGTH)
        # 최종 클릭 수와 분당 클릭 수(CPM)를 계산합니다.
        click_count = len(onset_times)
        cpm = click_count / duration_minutes if duration_minutes > 0 else 0

        # --- 터미널에 분석 결과 출력 ---
        print("\n--- 분석 결과 ---")
        print(f"총 감지된 클릭 수: {click_count}회")
        print(f"분당 클릭 수 (CPM): {cpm:.2f} 회/분")
        print(f"  - 사용된 파라미터: 민감도={PEAK_HEIGHT_THRESHOLD}, 최소간격={MIN_PEAK_DISTANCE_SECONDS}초")
        print(f"  - FFT 설정: N_FFT={N_FFT}, HOP_LENGTH={HOP_LENGTH}")

        # ===========================================================
        # 5. 시각화: 4개의 그래프로 결과 보여주기
        # ===========================================================

        # (1) 4행 1열의 그래프 캔버스(figure)와 각 그래프(axes)를 생성합니다.
        # figsize로 전체 크기를 지정하고, sharex 옵션은 사용하지 않아 축 충돌을 방지합니다.
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(18, 20))

        # --- 1행: 오디오 파형 및 최종 검출된 클릭 ---
        librosa.display.waveshow(y, sr=sr, ax=ax1, color='royalblue', alpha=0.8)
        ax1.vlines(onset_times, ymin=y.min(), ymax=y.max(), color='crimson', linestyle='--', linewidth=1.5, label=f'Detected Clicks ({click_count})')
        ax1.set_title(f'Audio Waveform & Detected Clicks (Total: {click_count}, CPM: {cpm:.2f})')
        ax1.set_ylabel('Amplitude')
        ax1.legend(loc='upper right')

        # --- 2행: 주파수 히스토그램 (FFT) ---
        # 이 그래프는 전체 오디오 파일에 어떤 주파수 성분이 강하게 분포하는지 보여줍니다.
        fft_result = np.fft.fft(y)
        freqs_fft = np.fft.fftfreq(len(y), d=1/sr)
        positive_freq_indices = np.where(freqs_fft >= 0) # 양수 주파수만 사용
        ax2.plot(freqs_fft[positive_freq_indices], np.abs(fft_result[positive_freq_indices]), color='darkviolet', linewidth=1.0)
        ax2.set_title('Frequency Histogram (FFT of Entire Audio)')
        ax2.set_xlabel('Frequency (Hz)') # 이 그래프의 x축은 주파수입니다.
        ax2.set_ylabel('Magnitude')
        ax2.set_xlim(0, sr / 2) # 가청 주파수 영역 위주로 표시
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        # --- 3행: 스펙트로그램 ---
        # 시간에 따른 주파수의 변화를 시각화합니다. 색이 밝을수록 해당 시간/주파수의 에너지가 강함을 의미합니다.
        S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='log', ax=ax3)
        ax3.set_title(f'Spectrogram (N_FFT={N_FFT})')
        ax3.set_ylabel('Frequency (Hz)')

        # --- 4행: 스펙트럼 플럭스 (Onset Strength) 및 피크 검출 ---
        # 분석의 핵심 과정을 보여주는 그래프입니다.
        # 주황색 선이 '소리 변화의 강도'이고, 빨간 X가 설정된 임계값을 넘어 '클릭'으로 판정된 지점입니다.
        stft_times = librosa.times_like(onset_env, sr=sr, hop_length=HOP_LENGTH)
        ax4.plot(stft_times, onset_env_normalized, label='Onset Strength (Spectral Flux)', color='darkorange', linewidth=1.5)
        ax4.plot(onset_times, onset_env_normalized[peaks], 'x', markersize=10, markeredgewidth=2, color='crimson', label=f'Detected Peaks ({click_count})')
        ax4.axhline(y=PEAK_HEIGHT_THRESHOLD, color='gray', linestyle=':', label=f'Peak Threshold ({PEAK_HEIGHT_THRESHOLD})')
        ax4.set_title('Onset Strength Envelope & Detected Peaks')
        ax4.set_xlabel('Time (s)') # 맨 아래 그래프에만 시간 축 라벨을 표시합니다.
        ax4.set_ylabel('Normalized Strength')
        ax4.legend(loc='upper right')
        ax4.set_ylim(0, 1.05)

        # --- X축 동기화 및 정리 ---
        # 시각화 충돌을 피하기 위해, 수동으로 시간 축 그래프들의 범위를 동기화합니다.
        # 1. 마지막 시간 그래프(ax4)의 x축 범위를 기준으로 삼습니다.
        final_xlims = ax4.get_xlim()
        # 2. 다른 시간 그래프(ax1, ax3)들의 x축 범위를 강제로 동일하게 설정합니다.
        ax1.set_xlim(final_xlims)
        ax3.set_xlim(final_xlims)
        # 3. 위쪽 그래프들의 x축 눈금 숫자(라벨)를 숨겨서 그래프를 깔끔하게 만듭니다.
        ax1.tick_params(axis='x', labelbottom=False)
        ax3.tick_params(axis='x', labelbottom=False)

        # 그래프들이 겹치지 않도록 전체 레이아웃을 자동으로 조정합니다.
        plt.tight_layout(pad=2.0)

        # 최종 그래프를 이미지 파일로 저장합니다.
        output_filename = 'geiger_click_analysis_final_commented.png'
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"\n상세 주석 버전의 분석 그래프를 '{output_filename}' 파일로 저장했습니다.")
        # 로컬 컴퓨터에서 바로 그래프를 보고 싶다면 아래 줄의 주석을 해제하세요.
        # plt.show()

    except Exception as e:
        # 코드 실행 중 문제가 발생하면 오류 메시지를 출력합니다.
        print(f"분석 중 오류 발생: {e}")

# ===================================================================
# 6. 코드 실행 지점
# ===================================================================
# 이 스크립트가 직접 실행될 때만 아래 코드가 동작하도록 합니다.
if __name__ == '__main__':
    # 분석할 오디오 파일의 경로를 여기에 지정합니다.
    audio_file = '10(down).m4a'
    # 위에서 정의한 메인 분석 함수를 호출합니다.
    analyze_geiger_clicks(audio_file)
