\# PhishGuard-X: Adaptive Multimodal Phishing Detection Under Evolving Web Threats



PhishGuard-X is a research prototype for phishing detection using URL lexical features and static webpage HTML features. It combines both modalities and investigates temporal generalization, distribution drift, and model retraining strategies.



\## Research Questions



\- \*\*RQ1:\*\* Does combining URL and HTML features improve detection compared with individual modalities?

\- \*\*RQ2:\*\* How do random and chronological evaluation differ?

\- \*\*RQ3:\*\* How do Population Stability Index (PSI) and Kolmogorov–Smirnov (KS) tests compare for drift detection?

\- \*\*RQ4:\*\* How does drift-triggered retraining compare with static and monthly retraining?

\- \*\*RQ5:\*\* What trade-offs exist between detection performance and inference latency?



\## Implemented Features



\- URL lexical feature extraction.

\- Static HTML feature extraction.

\- URL-only, HTML-only, and multimodal fusion experiments.

\- Chronological holdout evaluation.

\- PSI- and KS-based drift diagnostics.

\- Static, monthly, and drift-triggered retraining comparisons.

\- Flask prediction API.

\- Experiment result tables and graphs.



\## Dataset



The current experiments use 4,000 labeled webpage records from the PhreshPhish dataset: 2,245 legitimate and 1,755 phishing records.



The dataset is a limited sample and is not necessarily representative of real-world phishing traffic. Results should therefore be interpreted as preliminary.



\## Setup



Create and activate a Python virtual environment, then install dependencies:



```powershell

pip install -r requirements.txt

pip install scipy

```



\## Run the Experiments



Use the existing experiment scripts and results in the project. The current key outputs include:



\- `results/REAL\_PHRESH\_mm\_modality.csv`

\- `results/REAL\_PHRESH\_retrain\_v4\_summary.csv`

\- `results/REAL\_PHRESH\_retrain\_v4\_two\_class\_summary.csv`

\- `results/psi\_vs\_ks\_diagnostic.csv`

\- `results/psi\_ks\_retraining\_two\_class\_summary.csv`

\- `results/chronological\_holdout.csv`

\- `results/PHISHGUARD\_X\_FINAL\_RESULTS.md`



Some scripts may need to be run from the project root.



\## Run the Prediction API



Start the Flask server:



```powershell

python -m flask --app app.server run

```



The API is available at `http://127.0.0.1:5000`.



Example PowerShell request:



```powershell

Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body '{"url":"https://example.com"}'

```



The endpoint accepts a URL and optionally webpage HTML. Its response includes a phishing probability, risk category, modality scores, rule-based reasons, and inference latency.



\*\*Important:\*\* The current model has produced high phishing scores for well-known legitimate domains in manual testing. Do not rely on the output for security decisions until this false-positive issue has been investigated and independently validated.



\## Current Preliminary Results



On seven evaluated months containing both classes:



| Strategy | Mean F1 | Mean Precision | Mean Recall | Mean FPR | Mean AUC | Retraining Events |

|---|---:|---:|---:|---:|---:|---:|

| Static | 0.7347 | 0.8981 | 0.6252 | 0.0464 | 0.9197 | 0 |

| Monthly retraining | 0.8398 | 0.9007 | 0.7894 | 0.0538 | 0.9470 | 7 |

| Drift-triggered retraining | 0.8229 | 0.8976 | 0.7616 | 0.0331 | 0.9390 | 5 |



These are preliminary results from the current sample. The PSI threshold has not been independently validated, and single-class months are excluded from this comparison. The results do not establish real-world generalization.



\## Limitations



\- The dataset has substantial monthly class imbalance.

\- The current sample may contain temporal and collection biases.

\- Static HTML analysis does not execute JavaScript.

\- WHOIS and screenshot-based features are not currently part of the implemented pipeline.

\- UI explanations are rule-based rather than SHAP-based explanations.

\- The model's high scores for legitimate test domains require further investigation.

\- Drift thresholds and retraining policies need validation on independent data.



\## Future Work



\- Validate on a larger, representative, independently collected dataset.

\- Investigate false positives and calibrate probabilities using validation data.

\- Compare PSI and KS using a predefined evaluation protocol.

\- Add statistical tests and confidence intervals.

\- Add SHAP-based explanations.

\- Explore screenshot-based features as an additional modality.



\## Safety



Analyze saved webpage HTML as static data. Do not open suspicious pages in a normal browser or visit potentially malicious URLs outside an isolated, controlled environment.



