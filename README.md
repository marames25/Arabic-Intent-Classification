# Arabic Intent Classification

Intent classification for Egyptian Arabic telecom call-centre transcripts.

## Approaches

### Track A — BERT Fine-Tuning

Fine-tuned an Arabic transformer model for 27-class intent classification.

### Track C — Few-Shot Prompting

Used Qwen2.5-7B-Instruct with a constrained few-shot prompt and no model training.

## Intents

The system classifies customer utterances into 27 predefined intents.

## Dataset

The dataset was created specifically for this project and is not included
in this repository. The notebooks expect the dataset to be available locally
or on Google Drive.

## Results

See:

- `evaluation/evaluation_summary_trackA.txt`
- `evaluation/evaluation_summary_trackC.txt`
- `evaluation/track_comparison.txt`

## Notebooks

- `notebooks/Arabic_Intent_Classification_BERT.ipynb`
- `notebooks/Arabic_Intent_Classification_FewShot.ipynb`

## Prompt

The Track C prompt is available in:
`prompts/prompt_template.txt`
