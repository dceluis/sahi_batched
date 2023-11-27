## Usage

```
from sahi_batched.models import Yolov8DetectionModel
from sahi_batched import get_sliced_prediction_batched
```

Replace

```
AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="./yolov8n.pt",
    confidence_threshold=0.3,
    device="cpu",  # or 'cuda:0',
    ...
)

get_sliced_prediction(...)
```

with 

```
Yolov8DetectionModel(
    # model_type="yolov8", # comment this out
    model_path="./yolov8n.pt",
    confidence_threshold=0.3,
    device="cpu",  # or 'cuda:0',
    ...
)


get_sliced_prediction_batched(...)
```
