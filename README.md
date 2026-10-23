## Sign Speech

An initial, runnable sign-language classification pipeline built around numeric
feature vectors. The feature contract is intentionally simple: a CSV file has
one label column (default: `label`) and one or more numeric feature columns.

### Layout

- `data/`: dataset loading and validation
- `model/`: PyTorch model definition and checkpoint helpers
- `train/`: command-line training entry point
- `inference/`: FastAPI prediction service

### Install

```bash
python -m pip install -r requirements.txt
```

### CSV format

```csv
label,feature_0,feature_1
hello,0.12,-0.45
thanks,0.77,0.08
```

### Download a tabular dataset from Hugging Face

```bash
venv/bin/python -m data.download_huggingface \
	--dataset ORGANIZATION/DATASET \
	--split train \
	--label-column class \
	--output data/signs.csv
```

This adapter supports datasets with numeric feature columns. Image- or
video-only datasets need a landmark extraction step before they can be used by
the current classifier.

### Train

```bash
python -m train.train --csv data/signs.csv --output checkpoints/sign_speech.pt
```

The command writes a checkpoint containing the model configuration and label
mapping, so inference does not need a separate classes file.

### Serve predictions

```bash
uvicorn inference.api:app --reload
```

Then send `POST /predict` with JSON such as:

```json
{"features": [0.12, -0.45]}
```

Set `SIGN_SPEECH_CHECKPOINT` to point the API at a trained checkpoint.

### Video pipeline

The video pipeline is separate from the CSV pipeline. It reads each video
frame, detects up to two hands with MediaPipe, stores 21 `(x, y, z)` points per
hand, pads or truncates every example to 32 frames, and classifies the resulting
`[32, 126]` sequence with a GRU.

Install the additional video dependencies:

```bash
venv/bin/python -m pip install -r requirements.txt
```

Train directly from the Hugging Face video dataset:

```bash
venv/bin/python -m train.train_video \
	--dataset asthalochan/American_Sign_Language \
	--split train \
	--epochs 1 \
	--output checkpoints/sign_speech_video.pt
```

For the archive-only American Sign Language repository, fetch selected videos
without downloading the 3.3 GB ZIP in full:

```bash
HF_TOKEN="your_token" venv/bin/python -m train.train_video \
	--remote-zip asthalochan/American_Sign_Language \
	--split train --max-examples 16 --epochs 1 --batch-size 2 \
	--output checkpoints/sign_speech_video.pt
```

This reads the ZIP directory remotely, downloads only the selected compressed
MP4 members, extracts them into temporary files, and removes those files after
each example. Set `HF_TOKEN` in the shell rather than placing it in a source
file or command history.

Use `--max-examples 16 --epochs 1 --batch-size 2` for the first manual test.
When `--max-examples` is supplied, the command streams only those examples and
does not download the full video archive. Video decoding and landmark detection
are CPU-intensive, so a complete run can take a while. If MediaPipe installation
is interrupted, rerun the install and wait for the wheel download to finish
before starting training. The `asthalochan/American_Sign_Language` repository
currently contains a single large ZIP archive rather than row-level videos, so
it is not compatible with this streaming command; use a row-level Hugging Face
video dataset or extract that archive into labeled directories first.

Test a local video after training:

```bash
venv/bin/python -m inference.video_predict path/to/my_sign.mp4 \
	--checkpoint checkpoints/sign_speech_video.pt \
	--top-k 3
```

The output is one label and confidence per line. The test video should show one
of the dataset's 47 signs, with the hands visible and reasonably well lit.
