# PPO-Mujoco


## Table of Contents

- [소개](#소개)
- [요구 사항](#요구-사항)

## 소개

이 프로젝트는 Proximal Policy Optimization (PPO) 알고리즘을 사용하여 OpenAI Gym의 MuJoCo 환경에서 에이전트를 훈련하는 코드입니다.

## 요구 사항

- Python 3.6 이상
- PyTorch
- gym==0.24.1
- mujoco==2.2.0
- imageio>=2.1.2

## 파일 설명

- **`main.py`**: 메인 실행 스크립트로, 환경 설정 및 에이전트 훈련을 담당합니다.
- **`Agents/`**: PPO 에이전트의 구현이 포함된 디렉토리입니다.
- **`Networks/`**: 신경망 구조를 정의한 코드가 포함되어 있습니다.
- **`parse_params/`**: 명령줄 인자 파싱을 위한 스크립트가 포함되어 있습니다.
- **`utills/`**: 유틸리티 함수들이 포함된 디렉토리입니다.
