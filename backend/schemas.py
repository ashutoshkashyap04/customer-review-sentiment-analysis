from pydantic import BaseModel, Field

class ReviewSentiment(BaseModel):
    review : str = Field(
        ...,
        min_length= 1
    )