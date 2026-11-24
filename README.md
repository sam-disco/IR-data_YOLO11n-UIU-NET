# IR-compare — 야간 드론 열영상 사람 검출: YOLO11n · YOLO11n + 제안 필터 · UIU-Net · UIU-Net + 제안 필터

## 1. 준비

```
conda env create -f environment.yml
conda activate ircompare
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip uninstall -y albumentations
```

## 2. 데이터 받기

데이터(HIT-UAV v1.2.1, 815 MB)는 이 저장소에 들어 있지 않다. 아래 둘 중 하나로 준비한다.

**방법 A — 코드가 받게 둔다(따로 할 일 없음)**

`python run_all.py`만 실행하면 처음에 데이터 zip을 `data/hituav/`에 받는다. 인터넷이 되는 기계면 이 방법이 가장 쉽다.

**방법 B — 직접 받아서 넘긴다(받다가 끊기거나 인터넷이 막힌 기계일 때)**

1. 아래 주소에서 zip을 받는다. 파일 이름을 `HIT-UAV-Infrared-Thermal-Dataset-v1.2.1.zip`으로 저장한다.
   - https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset/archive/refs/tags/v1.2.1.zip
   - 또는 Zenodo: https://doi.org/10.5281/zenodo.7633134 (v1.2.1)
2. 실행할 때 zip 경로를 준다. **압축은 풀지 않는다**(코드가 필요한 폴더만 푼다).

```
python run_all.py --hituav-zip <받은 zip 경로>
```

어느 방법이든 zip의 sha256을 확인한다(`5fed15d2ff895afb16c0ca328ea19353f416d7543e6f39ba54e44ca26c538555`, 814,751,578 B). 다르면 「HIT-UAV 원본을 준비하지 못했다」로 멈추니 다시 받는다.

데이터 출처: Suo 외, "HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object detection," Scientific Data 10:227 (2023), DOI 10.1038/s41597-023-02066-6.

## 3. 실행

```
python run_all.py
python run_all.py --hituav-zip <HIT-UAV zip 경로>
```

HIT-UAV가 나눠 준 **배포 분할**의 야간 영상만 쓴다: train 1,390장으로 학습하고 test 396장 · val 195장으로 평가한다. 학습은 모두 6회다.

| 단계 | 하는 일 | 코드 |
|---|---|---|
| `prepare` | zip 풀기 → 야간 영상 목록 → 입력 영상 만들기(야간 1,981장): `[I, I, I]`(원영상) · `[I, P, I]`(가운데 채널 = 제안 필터 출력) → 화소 대조 | `prepare.py` · `common/hituav.py` · `common/filters.py` |
| `filter` | YOLO11n 학습 · 평가 2회 = 원영상 · 제안 필터 | `filter/` |
| `uiunet` | UIU-Net 학습 · 평가 4회 = (원영상 · 제안 필터) × 정답 마스크 둘(사람 상자 안 0.5배 타원 · 상자 채움). 정답 마스크는 학습 중에 라벨에서 그린다 | `uiunet/` |
| `report` | `results/REPORT.md` | `report.py` · `common/metrics.py`(채점) |

끝난 학습은 건너뛴다. **끊기면 같은 명령을 다시 실행한다**(UIU-Net은 끊긴 epoch부터, YOLO11n은 끊긴 학습만 처음부터 — 반쯤 남은 폴더는 지우지 않고 이름만 바꿔 둔다).
옵션: `--device 1`(GPU 번호, 기본 0) · `--only prepare`(또는 `filter` · `uiunet` · `report` — 한 단계만).

## 4. 결과 위치 · 보고서

- 결과: `results/REPORT.md`(보고서) · `results/REPORT.json` · `results/logs/run_all.log`(화면 기록) · `results/filter/` · `results/uiunet/`(가중치 · 평가 파일). 만든 입력 영상은 `data/build/`에 있다 — `baseline`과 `proposed`의 같은 이름 PNG를 나란히 열면 필터가 무엇을 남기는지 보인다.
- 보고서: 표 1은 네 모형의 Person AP50(UIU-Net은 마스크 방식마다 한 줄씩 — test · val), 표 2는 제안 필터 전후의 차, 표 3은 UIU-Net − YOLO11n, 표 4는 타원 − 상자다. 모형마다 seed 하나라서 몇 점 안쪽의 차이는 우열로 읽지 않는다. 배포 분할은 같은 비행의 영상이 train과 test에 나뉘어 있어서 값이 높게 나온다. 맨 위 문장 그대로, 이 값으로 필터나 모형의 우열을 말하지 않는다.


## 5. 서버에서 돌릴 때 - `IR-compare.zip`을 PC의 Downloads 폴더에 받음.

**PC의 PowerShell에서** (HIT-UAV zip을 받아 둔 것이 없으면 둘째 줄은 건너뛴다 — 서버가 직접 받는다)

```
scp $HOME\Downloads\IR-compare.zip <계정>@<서버 주소>:~/
scp $HOME\Downloads\HIT-UAV-Infrared-Thermal-Dataset-v1.2.1.zip <계정>@<서버 주소>:~/
ssh <계정>@<서버 주소>
```

**서버의 bash에서** (한 줄씩. 1절의 설치는 넷째 줄 자리에서 처음 한 번만 한다)

```
mkdir -p ~/IR_compare
python -m zipfile -e ~/IR-compare.zip ~/IR_compare
cd ~/IR_compare/IR-compare
conda activate <환경 이름>
nohup python run_all.py --hituav-zip ~/HIT-UAV-Infrared-Thermal-Dataset-v1.2.1.zip > ~/IR_compare/logs_run.txt 2>&1 &
tail -f ~/IR_compare/logs_run.txt
```

- HIT-UAV zip을 보내지 않았으면 다섯째 줄에서 `--hituav-zip ~/HIT-UAV-Infrared-Thermal-Dataset-v1.2.1.zip`을 뺀다.
- `nohup … &`로 띄우므로 접속이 끊겨도 계속 돈다. `tail -f`는 Ctrl+C로 나와도 실행은 멈추지 않는다. 다시 보려면 여섯째 줄만 다시 친다.
- 몇 분 안에 `입력 영상: 3962행 … MANIFEST와 같음`이 보이면 데이터 준비가 끝난 것이다. 도중에 멈췄으면 셋째 ~ 다섯째 줄을 다시 실행한다(이어 간다).
- GPU가 여러 개면 다섯째 줄의 `run_all.py` 뒤에 `--device <번호>`를 붙인다.
- 이전 판(날짜 접기판)을 `~/IR_compare`에 풀어 돌린 적이 있으면 폴더 이름을 다르게 한다(예: `~/IR_compare_official` — 위 여섯 줄의 `~/IR_compare`를 모두 바꾼다).

**끝난 뒤 결과를 PC로 — 서버 창이 아니라 PC에서 새로 연 PowerShell에서** (프롬프트가 `PS C:\…>`인 창)

```
scp <계정>@<서버 주소>:~/IR_compare/IR-compare/results/REPORT.md $HOME\Downloads\
scp <계정>@<서버 주소>:~/IR_compare/IR-compare/results/logs/run_all.log $HOME\Downloads\
```
