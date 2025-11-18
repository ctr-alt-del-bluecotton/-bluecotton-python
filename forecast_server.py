from datetime import date, timedelta
from pathlib import Path
from typing import List

import xgboost as xgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ==== 경로 설정 ====
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "xgb_revenue.json"

# ==== FastAPI 앱 ====
app = FastAPI(title="Revenue Forecast API")

# ==== Pydantic 모델 ====


class ForecastPoint(BaseModel):
    date: date
    revenue: int  # 지금은 안 쓰지만, 확장 대비해서 남겨둠


class ForecastRequest(BaseModel):
    series: List[ForecastPoint]
    horizon: int


class ForecastResponsePoint(BaseModel):
    date: str
    predictRevenue: int


class ForecastResponse(BaseModel):
    data: List[ForecastResponsePoint]


# ==== 전역 모델 객체 ====
model = None


def load_model():
    global model
    if MODEL_PATH.exists():
        model = xgb.XGBRegressor()
        model.load_model(str(MODEL_PATH))
        print(f"✅ XGBoost 모델 로드 성공: {MODEL_PATH}")
    else:
        print(f"⚠️ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")


@app.on_event("startup")
def on_startup():
    load_model()


def make_feature_frame(start_date: date, horizon: int) -> pd.DataFrame:
    """
    마지막 날짜(start_date) 다음 날부터 horizon일 만큼의
    특징(dayofyear, month, weekday)을 생성해서 DataFrame으로 반환.
    """
    rows = []
    for i in range(1, horizon + 1):
        d = start_date + timedelta(days=i)
        rows.append(
            {
                "dayofyear": d.timetuple().tm_yday,
                "month": d.month,
                "weekday": d.weekday(),  # 0=월, 6=일
            }
        )

    df = pd.DataFrame(rows)
    return df[["dayofyear", "month", "weekday"]]


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    if not request.series:
        raise HTTPException(status_code=400, detail="series is empty")

    try:
        # 요청 로그용
        print(
            f"[Forecast] series_len={len(request.series)}, "
            f"horizon={request.horizon}, "
            f"last_date={request.series[-1].date}"
        )

        # 마지막 관측 날짜 기준으로 이후 horizon일 예측
        last_date = request.series[-1].date

        # 입력 특징 DataFrame 생성 (⚠️ train_xgb_model과 동일한 컬럼)
        X_future = make_feature_frame(last_date, request.horizon)

        # 예측
        preds = model.predict(X_future)

        # 응답 포맷으로 변환
        results = []
        for i, p in enumerate(preds, start=1):
            d = last_date + timedelta(days=i)
            results.append(
                ForecastResponsePoint(
                    date=d.isoformat(),
                    predictRevenue=int(round(float(p))),
                )
            )

        return ForecastResponse(data=results)

    except Exception as e:
        # 디버깅을 위해 내부 에러 그대로 detail에 실어 보냄
        print(f"[ERROR] Prediction failed: {repr(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e!r}",
        )


# uvicorn forecast_server:app --reload --port 8001 로 실행 중이면 아래는 안 써도 됨.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("forecast_server:app", host="127.0.0.1", port=8001, reload=True)
