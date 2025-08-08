-----

# Geiger Counter Audio Analyzer

[](https://www.python.org/)
[](https://opensource.org/licenses/MIT)

Geiger-Müller 계수기(가이거 계수기)에서 발생하는 '클릭' 소리를 녹음한 오디오 파일을 분석하여, 방사선 이벤트의 발생 횟수와 분당 계수(CPM)를 자동으로 계산하는 Python 스크립트입니다.

이 프로젝트는 단순한 소리 크기(amplitude)가 아닌, **스펙트럼 플럭스(Spectral Flux)** 기반의 **시작점 감지(Onset Detection)** 알고리즘을 사용하여 배경 소음 속에서도 '틱' 소리를 효과적으로 식별합니다.

## 주요 기능

  * **이벤트 카운팅**: 오디오 파일에서 가이거 '클릭' 소리의 총횟수를 정확하게 계수합니다.
  * **CPM 계산**: 분당 계수(Counts Per Minute)를 자동으로 계산하여 방사선 수준을 정량화합니다.
  * **강력한 분석 기법**: 소리의 급격한 변화를 감지하는 스펙트럼 플럭스 기법을 사용하여 잡음 환경에서도 안정적으로 동작합니다.
  * **상세한 시각화**: 분석 결과를 4개의 상세한 그래프(파형, FFT, 스펙트로그램, Onset 검출)로 구성된 이미지 파일로 저장하여 분석 과정을 직관적으로 이해할 수 있도록 돕습니다.
  * **파라미터 튜닝**: 사용자가 직접 분석 파라미터를 조절하여 다양한 녹음 환경과 조건에 맞게 탐지 정확도를 최적화할 수 있습니다.

-----

## 요구 사항

스크립트를 실행하기 위해 다음 환경과 라이브러리가 필요합니다.

1.  **Python 3.7+**

2.  **FFmpeg**: `.m4a` 등 다양한 오디오 포맷을 처리하기 위해 반드시 필요합니다.

      * **Windows**: [FFmpeg 다운로드 페이지](https://www.gyan.dev/ffmpeg/builds/)에서 `ffmpeg-release-essentials.zip`을 받아 압축 해제 후, `bin` 폴더를 시스템 환경 변수 `Path`에 추가합니다.
      * **macOS**: `brew install ffmpeg`
      * **Linux (Debian/Ubuntu)**: `sudo apt install ffmpeg`

3.  **Python 라이브러리**:

    ```
    librosa
    numpy
    matplotlib
    scipy
    mplhep
    ```

-----

## 설치 방법

1.  **프로젝트 저장소 복제(Clone)**

    ```bash
    git clone https://github.com/opercjy/60Co_proj.git
    cd 60Co_proj
    ```

2.  **FFmpeg 설치**

    위의 "요구 사항" 섹션을 참고하여 자신의 운영체제에 맞게 FFmpeg을 설치합니다.

3.  **Python 라이브러리 설치**

    아래 명령어를 사용하여 필요한 모든 라이브러리를 한 번에 설치합니다.

    ```bash
    pip install librosa numpy matplotlib scipy mplhep
    ```

-----

## 사용 방법

1.  분석하고 싶은 오디오 파일(예: `my_recording.m4a`)을 스크립트가 있는 폴더에 넣습니다.

2.  스크립트 파일(`analyze_audio_events.py`)을 열어 맨 아래 `if __name__ == '__main__':` 부분을 수정합니다.

    ```python
    if __name__ == '__main__':
        # 분석할 오디오 파일의 경로를 여기에 지정합니다.
        audio_file = '10(down).m4a'  # <-- 이 부분을 실제 파일명으로 변경
        analyze_geiger_clicks(audio_file)
    ```

3.  터미널에서 스크립트를 실행합니다.

    ```bash
    python analyze_audio_events.py
    ```

4.  **결과 확인**

      * **터미널**: 총 클릭 수와 CPM이 텍스트로 출력됩니다.
      * **이미지 파일**: `geiger_click_analysis_visualization_fixed.png` 라는 이름으로 4개의 상세 그래프가 포함된 이미지 파일이 생성됩니다.

-----

## 파라미터 튜닝으로 정확도 높이기

`analyze_audio_events.py` 스크립트 상단의 `사용자 조정 핵심 파라미터` 섹션에서 다음 값들을 조절하여 탐지 정확도를 최적화할 수 있습니다.

  * `PEAK_HEIGHT_THRESHOLD`: **가장 중요한 파라미터**로, 클릭으로 인정할 최소 강도(민감도)를 결정합니다. (0.0 \~ 1.0)

      * **값을 높이면**: 잡음이나 약한 신호를 무시하고, 더 크고 명확한 클릭만 감지합니다. (계수 ↓)
      * **값을 낮추면**: 작은 신호도 감지하여 민감도가 높아지지만, 잡음을 클릭으로 잘못 인식할 수 있습니다. (계수 ↑)

  * `MIN_PEAK_DISTANCE_SECONDS`: 하나의 클릭이 여러 번으로 나뉘어 감지되는 것을 방지합니다. 일반적으로 `0.05` 정도면 충분합니다.

  * `N_FFT`, `HOP_LENGTH`: 오디오 분석의 시간/주파수 해상도를 결정하는 고급 파라미터입니다. '틱' 소리와 같이 짧은 이벤트를 분석할 때는 현재 설정된 값(`1024`, `256`)이 대부분의 경우에 잘 동작합니다.

-----

## 예시 출력 결과

스크립트를 성공적으로 실행하면 아래와 같은 상세 분석 그래프를 얻을 수 있습니다.
<img width="1675" height="1875" alt="geiger_click_analysis_visualization_fixed" src="https://github.com/user-attachments/assets/0a9cdb5f-ec2e-4cb5-9053-c813c1422c69" />
-----

## 라이선스

이 프로젝트는 [MIT 라이선스](https://www.google.com/search?q=LICENSE)를 따릅니다.
