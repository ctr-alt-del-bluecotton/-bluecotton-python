import pandas as pd
from pathlib import Path
import xgboost as xgb

# ==== 경로 설정 ====
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "daily_revenue.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "xgb_revenue.json"


def load_data(path: Path) -> pd.DataFrame:
    print(f"📂 데이터 로드: {path}")
    df = pd.read_csv(path)

    # date 컬럼을 datetime으로 변환
    df["date"] = pd.to_datetime(df["date"])

    # 특징 컬럼 생성 (⚠️ forecast_server와 반드시 동일해야 함)
    df["dayofyear"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday  # 0=월, 6=일

    print(f"✅ 데이터 행 수: {len(df)}")
    return df


def train_and_save_model():
    # 데이터 로드
    df = load_data(DATA_PATH)

    # 입력 특징 / 타깃 분리
    feature_cols = ["dayofyear", "month", "weekday"]
    X = df[feature_cols]
    y = df["revenue"]

    # 모델 디렉터리 없으면 생성
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("🚂 모델 학습 중...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )

    model.fit(X, y)
    print("✅ 학습 완료")

    # 모델 저장
    model.save_model(str(MODEL_PATH))
    print(f"✅ 모델 저장 완료: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
