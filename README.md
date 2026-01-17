# Codebook-Injected Segmentation

This repository contains the code and data for injecting expert codebooks (taxonomies) into dialogue segmentation models. It allows for both LLM-based segmentation and training of neural segmentation models (MoveRAG, Dial-Start).

## Project Structure

```
Codebook-Injected_Segmentation/
├── data/                       # Dataset directory (place your CSVs here)
│   ├── Upchieve_CLASS/         # CLASS dataset files
│   └── TalkMoves/              # TalkMoves dataset files
├── src/                        # Source code
│   ├── eval/                   # Evaluation metrics scripts
│   ├── seg_LLM/                # LLM-based segmentation (Gemini/GPT4)
│   └── seg_Dial_start/         # Neural model training and inference
├── results/                    # Output directory for segmentation results
├── .env.template               # Template for API keys (COPY to .env)
├── .gitignore                  # Git ignore rules
└── requirements.txt            # Python dependencies
```

## Setup & installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Codebook-Injected_Segmentation
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables:**
    To use LLM-based components, you need to set up your API keys.
    Copy the template file:
    ```bash
    cp .env.template .env
    ```
    Then edit `.env` and add your **OpenAI** or **Cornell AI Gateway** keys.

## Data Preparation

Ensure your data is placed in the `data/` directory.
- **CLASS Dataset**: Should be in `data/Upchieve_CLASS/`
- **TalkMoves Dataset**: Should be in `data/TalkMoves/`

To preprocess the raw data for training:

```bash
cd src/seg_Dial_start
# For CLASS dataset
python data_preprocess_class_tax.py --save_name _class_processed
```

## Usage

### 1. LLM-Based Segmentation
To run zero-shot or few-shot segmentation using an LLM (e.g., Gemini Pro):

```bash
# From the project root
source .env  # Load your API keys
bash src/run_segmentation.sh
```
*Note: You may need to adjust the `DATA_DIR` and `OUT_JSON` variables inside `src/run_segmentation.sh` to match your desired input/output.*

### 2. Training Neural Models
**Attribution**: The `Dial-Start` model is adapted from [AlibabaResearch/DAMO-ConvAI](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/dial-start). We have added a variant using Retrieval Augmentation.
To train the segmentation models:

```bash
cd src/seg_Dial_start

# Train on CLASS dataset
python train_v1.py --dataset CLASS_all --save_model_name CLASS_Model --batch_size 8 --epoch 10

# Train with Taxonomy Injection (MoveRAG)
python train_max_MoveRAG.py --dataset CLASS_all --save_model_name CLASS_MoveRAG
```

### 3. Evaluation
To evaluate the results against ground truth:

```bash
# Example evaluation command
python src/eval/evaluation_segmention.py \
  --file1 results/llm_results_class.json \
  --file2 results/model_output_class.json \
  --csv data/Upchieve_CLASS/CLASS_preprocessed.csv
```

## Security Note
**API Keys**: Never commit your `.env` file or hardcode keys into scripts. This repository uses environment variables to handle credentials securely.
**Data**: Raw data files are excluded from version control via `.gitignore`.

