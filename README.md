# Review Reality Check — ML backend + Chrome extension

This starter project follows the attached synopsis: text features (TF-IDF n-grams and linguistic signals), optional reviewer/product/time metadata, a supervised baseline classifier, evaluation on a held-out split, and a per-review score surfaced in a Chrome extension.

## Important limitations

- A model trained on one dataset may not generalize to Amazon, Yelp, Google Maps, etc. Website DOMs also vary; the included generic content script is a starter and will need site-specific selectors for reliable review detection.
- The displayed 0–10 score is the model's estimated probability of the dataset's chosen `human` class multiplied by 10. It is **not a verified truth score**, and logistic-regression probabilities may need calibration.
- The extension currently sends review text to the local API at `127.0.0.1:8000`. Do not deploy it to users without privacy, security, CORS, hosting, and data-handling review.

## 1. Setup

Use Python 3.10+.

```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## 2. Train using your CSV

Example only; replace path and column names with those in your CSV:

```bash
python train.py --csv "C:\\data\\reviews.csv" --text-col review_text --label-col label --human-label genuine --reviewer-col reviewer_id --product-col product_id --rating-col rating --timestamp-col timestamp
```

For the classic Ott dataset, you may need to combine its separate files and set the correct labels/columns first. The `--human-label` value must match the genuine label in your CSV exactly (case-insensitive); every other label is treated as non-human/deceptive for this binary baseline.

The script:
1. Drops rows without labels.
2. Creates a stratified 80/20 train/test split by default.
3. Fits TF-IDF and numeric preprocessing **on the training split only**.
4. Trains Logistic Regression with class weighting.
5. Prints accuracy, precision/recall/F1, ROC-AUC, and confusion matrix.
6. Saves `backend/artifacts/review_model.joblib` and `metrics.json`.

### Behavioural-feature caution
Reviewer/product counts are computed from the provided dataset. For strict evaluation on unseen reviewers/products, use a group-based split by reviewer/product rather than a random split. For deployment, reviewer history must be available at inference time; otherwise these features are missing and imputed. The extension currently submits text and optional rating only.

## 3. Start API

From `backend/`:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to test the API.
- `GET /health`
- `POST /predict` with `{"text":"The room was clean and staff were helpful.","rating":4}`
- `POST /predict/batch` with `{"reviews":[{"text":"..."}]}`

## 4. Load the Chrome extension

1. Start the API and train a model first.
2. Open `chrome://extensions`.
3. Enable **Developer mode**.
4. Choose **Load unpacked** and select the `extension/` folder.
5. Open a review page. Click the extension icon to turn scoring on/off.

The content script uses broad generic selectors as a demonstration. If a site is not detected correctly, edit `SELECTORS` in `extension/content.js` for that website's review-card DOM. The toggle hides badges when disabled; it does not disable the browser's ability to read the page.

## Score bands

- 8.5–10.0: Most Likely Human
- 7.0–8.4: Probably Human
- 5.0–6.9: Uncertain
- 3.0–4.9: Probably Fake
- 0.0–2.9: Most Likely Fake

These are product-language bands, not calibrated certainty thresholds. Tune them using validation data and clearly communicate uncertainty.

## Project structure

```text
deceptive_review_extension/
  backend/
    train.py
    main.py
    requirements.txt
    artifacts/            # generated after training
  extension/
    manifest.json
    popup.html
    popup.css
    popup.js
    content.js
    content.css
```
