"""Analysis settings shared by command-line and optional dashboard entry points."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    foot: Literal['left', 'right'] = 'right'
    shot_type: Literal['penalty', 'free_kick', 'shot'] = 'shot'
    camera_view: Literal['side', 'oblique', 'front'] = 'side'
    camera_motion: Literal['fixed', 'moving'] = 'fixed'
    direction: Literal['auto', 'left', 'right'] = 'auto'
    contact_frame: int | None = Field(default=None, ge=0, strict=True)
    scale_m_per_px: float | None = Field(default=None, gt=0, le=10, strict=True)
