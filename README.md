# Ethical Chat Guard

A real-time AI safety tool for detecting and mitigating coercive language patterns in Large Language Model (LLM) responses. This Streamlit-based application combines rule-based detection with machine learning to identify potentially harmful or manipulative text in chatbot interactions.

##  Project Overview

Ethical Chat Guard helps developers, researchers, and content moderators analyze LLM outputs for coercive language patterns. The system provides:

- **Real-time risk assessment** of chatbot responses
- **Multi-layered detection** combining rules and ML models
- **Interactive chat interface** with live auditing
- **Batch analysis capabilities** for CSV datasets
- **Safe rewrite suggestions** for high-risk responses
- **Detailed session reports** with downloadable audit trails

##  Features

### Main Chat Interface (`EthicsBot.py`)
- Interactive chatbot with real-time coercion detection
- Live risk scoring (0-100) with color-coded alerts
- Highlighted detection of coercive phrases in responses
- Three sensitivity modes: Conservative, Balanced, Aggressive
- Safe rewrite feature to automatically generate non-coercive alternatives
- Session summary reports with downloadable CSV exports

### Quick Risk Checker (`Quick_Risk_Checker.py`)
- Single text analysis for standalone assessment
- Batch CSV processing for bulk evaluation
- Configurable risk sensitivity settings
- Exportable results for documentation and compliance

### Detection Categories

The system detects seven categories of coercive language:

1. **Urgency** - Time pressure tactics ("do it now", "immediately")
2. **Inevitability** - Removal of choice ("no other option", "must")
3. **Emotional Pressure** - Guilt and emotional manipulation ("you will regret")
4. **Authority Pressure** - Appeals to expertise ("trust me", "experts agree")
5. **Dismissal of Alternatives** - Suppressing critical thinking ("don't overthink")
6. **Fear-Based Pressure** - Threat-based persuasion ("something bad will happen")
7. **Reward Baiting** - Over-promising benefits ("this will guarantee success")

## Architecture

### Components

**Services Layer:**
- `detector.py` - Core rule-based detection engine with category markers
- `sbert_lr.py` - Machine learning model (Sentence-BERT + Logistic Regression)
- `llm_openai.py` - OpenAI GPT integration for chat and rewrites
- `rewrite.py` - Safe response rewriting logic
- `storage.py` - Session data persistence

**Models:**
- `lr_coercion.joblib` - Trained logistic regression classifier
- `threshold.json` - Dynamic detection thresholds
- `coercion_threshold.json` - Model-specific thresholds

**Data:**
- `hh_coercion_weak_labels.csv` - Training dataset (595KB)
- `coercion_dataset_500_v1.csv` - Evaluation dataset

##  Requirements

### Core Dependencies

```
Python 3.8+
streamlit==1.53.1
openai==2.15.0
sentence-transformers==5.2.0
scikit-learn==1.8.0
pandas==2.3.3
torch==2.10.0
joblib==1.5.3
python-dotenv==1.2.1
```

See `requirements.txt` for complete dependency list.

##  Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Ethical_Chat_Guard_Project
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Verify model files

Ensure the following files exist:
- `models/lr_coercion.joblib`
- `models/threshold.json`
- `models/coercion_threshold.json`

##  Usage

### Running the Main Application

```bash
streamlit run EthicsBot.py
```

The application will start on `http://localhost:8501`

### Running Quick Risk Checker

```bash
streamlit run pages/Quick_Risk_Checker.py
```

Access directly at `http://localhost:8501/Quick_Risk_Checker`

### Modes Explained

**Conservative Mode:**
- Higher detection threshold
- Reduces false positives
- Best for production environments where accuracy is critical

**Balanced Mode (Default):**
- Standard detection sensitivity
- Balanced between precision and recall
- Recommended for general use

**Aggressive Mode:**
- Lower detection threshold
- Catches more potential issues but may have false positives
- Best for research and thorough content auditing

##  Model Training

### Training the Detection Model

```bash
python models/train_lr_hh.py
```

This script:
1. Loads the training dataset (`hh_coercion_weak_labels.csv`)
2. Generates embeddings using Sentence-BERT (all-MiniLM-L6-v2)
3. Trains a Logistic Regression classifier
4. Saves the model to `models/lr_coercion.joblib`

### Tuning Detection Thresholds

```bash
python models/tune_threshold.py
```

Optimizes the classification threshold based on validation performance.

##  Project Structure

```
Ethical_Chat_Guard_Project/
├── EthicsBot.py                 # Main chat application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .env                         # API keys (create this)
│
├── pages/
│   └── Quick_Risk_Checker.py    # Standalone risk analysis tool
│
├── services/
│   ├── detector.py              # Rule-based detection engine
│   ├── sbert_lr.py              # ML model inference
│   ├── llm_openai.py            # OpenAI API integration
│   ├── rewrite.py               # Safe rewrite logic
│   └── storage.py               # Data persistence
│
├── models/
│   ├── lr_coercion.joblib       # Trained ML model
│   ├── threshold.json           # Detection thresholds
│   ├── coercion_threshold.json  # Model thresholds
│   ├── train_lr_hh.py           # Model training script
│   └── tune_threshold.py        # Threshold optimization
│
├── data/
│   ├── hh_coercion_weak_labels.csv      # Training data
│   ├── coercion_dataset_500_v1.csv      # Evaluation data
│   ├── build_hh_coercion_dataset.py     # Dataset builder
│   └── tune_threshold.py                # Data-level threshold tuning
│
└── utils/
    ├── config.py                # Configuration loading
    ├── helpers.py               # Utility functions
    └── __init__.py
```

##  Technical Details

### Detection Algorithm

The system uses a three-component fusion approach:

1. **Rule Score (45-55%)** - Pattern matching against 85+ known coercive phrases
2. **Model Score (30-40%)** - Semantic similarity using SBERT embeddings + Logistic Regression
3. **Context Score (15%)** - Adjusts for user-requested coercion (e.g., "be strict with me")

Final Risk Score = `(W_rule × R_score) + (W_model × M_score) + (W_context × C_score)`

### Highlighting Logic

Coercive phrases are highlighted in the chat interface when:
- Model confidence exceeds the highlight gate (varies by mode)
- Multiple overlapping detections are deduplicated, preferring longer matches

### Safe Rewrite Feature

When a high-risk response is detected:
1. User clicks "Safe rewrite"
2. System sends response to OpenAI with rewriting prompt
3. Generated alternative maintains informational content while removing coercive language
4. User can preview, edit, or add rewrite to conversation
5. Rewrite is automatically audited before addition

##  Performance Metrics

The machine learning model achieves (based on training script):
- Training set: 80% of labeled data
- Test set: 20% holdout
- Model: Logistic Regression with max_iter=4000, C=1.0
- Features: 384-dimensional SBERT embeddings (all-MiniLM-L6-v2)

##  Safety Considerations

**What This Tool Does:**
- Detects potential coercive language patterns
- Provides risk scores and explanations
- Suggests safer alternatives

**What This Tool Doesn't Do:**
- Guarantee complete detection of all harmful content
- Replace human judgment in critical decisions
- Provide legal or professional advice

**Recommended Use:**
- As one layer in a multi-faceted content moderation system
- For research and analysis of LLM behavior
- To improve prompt engineering and response quality
- In conjunction with human review for high-stakes applications

## Contributing

Contributions are welcome! Areas for improvement:

- Additional coercion categories and patterns
- Multi-language support
- Integration with other LLM providers
- Enhanced ML models (transformers, ensemble methods)
- Real-time streaming detection
- API endpoint development


##  Research Background

This project is based on research in ethical AI and LLM safety, focusing on detection of unethical reasoning patterns in large language models. See the included research document for detailed methodology and findings.

##  Known Issues

- Large batch processing (>2000 rows) may be slow
- OpenAI API rate limits apply to chat and rewrite features
- Model file size requires ~5MB storage

##  Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Contact the development team
- Review the included research documentation

##  Acknowledgments

- Anthropic's Helpful and Harmless (HH) dataset for training data
- Sentence-BERT for semantic embeddings
- Streamlit for the web interface framework
- OpenAI for GPT model access
