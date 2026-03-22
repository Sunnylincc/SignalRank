import numpy as np
import pandas as pd

from signalrank.common.types import UserContext
from signalrank.orchestration.pipeline import build_assets, recommend


def test_recommend_smoke() -> None:
    items = pd.DataFrame(
        {
            "item_id": [1, 2, 3, 4],
            "category": ["electronics", "travel", "fashion", "sports"],
            "price_bucket": ["mid", "high", "low", "mid"],
            "language": ["en", "en", "en", "en"],
            "title": ["Headphones", "Trip", "Shoes", "Bike"],
            "description": ["electronics audio", "travel guide", "fashion style", "sports gear"],
        }
    )
    interactions = pd.DataFrame(
        {
            "user_id": [11, 11, 12, 12],
            "item_id": [1, 2, 1, 3],
            "event_type": ["click", "impression", "conversion", "click"],
            "surface": ["home", "home", "search", "search"],
        }
    )

    assets = build_assets(items, interactions)
    recs = recommend(
        assets,
        np.array([1.0] * 64, dtype=np.float32),
        user_ctx=UserContext(user_id=11, surface="home"),
        top_k=2,
    )
    assert len(recs) == 2
    assert recs[0].final_score >= recs[1].final_score
