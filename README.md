Ethical Chat Guard system by introducing **three additional unethical pattern detection modules** beyond coercive language:

- Hallucination Detection  
- Bias Detection  
- Unsafe Persuasion Detection  

These modules allow the system to perform **multi-dimensional ethical risk assessment**, enabling deeper analysis of LLM responses and improving real-time safety auditing.

While the original system primarily focused on coercive language, this extension broadens the scope of analysis to cover additional high-impact ethical risks commonly observed in modern LLM deployments.

##  Objectives

The main objectives of this contribution are:

- Detect hallucinated or inconsistent content in LLM responses  
- Identify explicit social and demographic bias  
- Detect manipulative or pressuring persuasion language  
- Present independent risk scores for each category  
- Provide clear color-coded visualization inside the UI  
- Maintain lightweight, offline-capable operation  

##  Added Detection Modules

### 1. Hallucination Detection  
File: `services/hallucination_detector.py`

Detects whether an LLM response may contain hallucinated or internally inconsistent information.

### 2. Bias Detection  
File: `services/bias_detector.py`

Detects biased statements related to protected demographic categories.

### 3. Unsafe Persuasion Detection  
File: `services/persuasion_detector.py`

Detects manipulative or pressuring persuasion patterns.

##  System Architecture Extension

User Prompt
↓
Local LLM Generator
↓
| Coercion Detector (existing) |
| Hallucination Detector (new) |
| Bias Detector (new) |
| Persuasion Detector (new) |
↓
Risk Fusion Engine
↓
UI Risk Panel (Color-Coded)


Each detector runs independently and outputs a probability score between 0 and 1.


##  Hallucination Detection

### Purpose

LLMs sometimes generate content that is:

- Factually incorrect  
- Internally contradictory  
- Logically inconsistent  

This module detects hallucination risk using **semantic consistency analysis**.

### Method

1. Split response into sentences  
2. Generate embeddings using Sentence-BERT  
3. Compute cosine similarity between sentence embeddings  
4. Extract statistical features  
5. Apply Logistic Regression classifier  

### Feature Vector

[mean_similarity, minimum_similarity, sentence_count]


### Model

- Sentence Transformer: all-MiniLM-L6-v2  
- Classifier: Logistic Regression  
- Stored Model File: `models/hallucination_model.pkl`  


### Output

0.0 → Low hallucination risk
1.0 → High hallucination risk


### Strengths

- Fully offline  
- Lightweight  
- Fast inference  

### Limitations

- Cannot verify factual correctness  
- Measures only internal consistency  


##  Bias Detection

### Purpose

Detects potential bias targeting:

- Gender  
- Race  
- Religion  
- Nationality  


### Approach

Rule-based keyword matching using predefined phrase groups.

### Example Patterns

women are
men are
muslims are
indians are
black people are


### Scoring Strategy

| Matches Found | Output Score |
|-------------|--------------|
| 0 | 0.0 |
| 1 | 0.4 |
| 2 | 0.7 |
| ≥3 | 0.9 |


### Output

Probability value between 0 and 1.


### Strengths

- Interpretable  
- Transparent  
- Easy to extend  

### Limitations

- Cannot detect implicit bias  
- English-only patterns  


##  Unsafe Persuasion Detection

### Purpose

Detects manipulative persuasive language encouraging immediate or pressured actions.

### Pattern Categories

- Urgency  
- Pressure  
- Reward baiting  
- Authority claims  
- Commitment pressure  

### Example Patterns

buy now
act fast
limited time
you must
only today
sign up now


### Scoring Formula

score = min(1.0, hits / 5)

### Output

Probability between 0 and 1.

### Strengths

- Fast  
- Simple  
- Low computational cost  

### Limitations

- Rule-based  
- No semantic reasoning  

##  UI Risk Panel Integration

The Risk Panel now displays:

- Risk Score (0–100)  
- Hallucination Probability  
- Bias Probability  
- Persuasion Probability  

Each value is shown with a colored indicator.

##  Color Coding

| Probability | Label | Color |
|------------|------|------|
| < 0.30 | LOW | GREEN |
| 0.30 – 0.69 | MEDIUM | YELLOW |
| ≥ 0.70 | HIGH | RED |

##  Example Output

Input:

Buy now! Limited time offer! You must invest today!

Output:

Hallucination: 0.4 (YELLOW)
Bias: 0.0 (GREEN)
Persuasion: 1.0 (RED)
Risk Score: 75/100

##  Fusion with Core Risk Engine

The additional scores are passed into:

services/detector.py → assess()


They are stored in the Assessment object and displayed by the UI.

##  Why This Extension Matters

Single-metric safety systems fail to capture the full risk profile of LLM outputs.

This contribution enables:

- Multi-axis ethical auditing  
- Greater transparency  
- Better debugging  
- Improved user trust  

##  Performance

- Runs in real time  
- CPU-only execution  
- No network dependency  

##  Extensibility

Future detectors can be added using the same pattern:

- Hate Speech  
- Self-Harm  
- Extremism  
- Misinformation  

##  Known Limitations

- Rule-based bias and persuasion detection  
- No multilingual support  
- Hallucination detection is proxy-based  

##  Future Improvements

- Transformer-based persuasion classifier  
- Multilingual bias detection  
- External fact-check integration  
- Ensemble hallucination models  

##  Summary

This contribution transforms Ethical Chat Guard from a single-focus coercion detector into a **multi-dimensional ethical risk analysis platform**, enabling deeper inspection of LLM behavior and supporting safer AI deployment.
