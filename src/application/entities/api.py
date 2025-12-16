from typing import Optional

from pydantic import BaseModel, Field, conlist


class ImageEntry(BaseModel):
    image_name: str = Field(..., description="image name")
    image_base64: str = Field(..., description="image base64")


class ImageRequestEntry(BaseModel):
    images: conlist(ImageEntry, min_length=2, max_length=2)
    area: Optional[int] = Field(description="image area", default=4)
    registry_threshold: Optional[int] = Field(
        description="registry threshold", default=1000
    )
    enableColor: Optional[bool] = Field(default=True, description="enable color")
