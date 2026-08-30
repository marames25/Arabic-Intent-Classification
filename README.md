# Arabic Intent Classification

Intent classification for Egyptian Arabic telecom call-centre transcripts.

The system classifies a customer's utterance into one of **27 predefined intents**. Two approaches are implemented and evaluated:

- **Track A — Fine-tuned AraBERT**
- **Track C — Few-Shot Prompting with Qwen2.5-7B-Instruct**

The project is designed for Egyptian Arabic, particularly customer-support utterances from telecom call-centre conversations.

---

## Approaches

### Track A — AraBERT Fine-tuning

Track A fine-tunes an Arabic BERT-based encoder for supervised intent classification.

**Base model:**

```text
aubmindlab/bert-base-arabertv02-twitter
```

The model is fine-tuned on the training split and evaluated on validation and held-out test data.

The notebook also includes:

- Arabic preprocessing using the AraBERT preprocessor
- Label encoding
- Tokenization
- Supervised fine-tuning
- Accuracy, precision, recall, and macro F1
- Confusion matrix
- Per-intent error analysis
- Latency benchmarking
- Held-out test evaluation
- Single-utterance inference
- Blind-test prediction generation
- Saving the final fine-tuned model and tokenizer

**Notebook:**

```text
notebooks/Arabic_Intent_Classification_BERT.ipynb
```

---

### Track C — Few-Shot Prompting

Track C uses a pretrained instruction-following language model without additional model training.

**Model:**

```text
Qwen/Qwen2.5-7B-Instruct
```

The model is loaded in 4-bit quantization.

The classifier uses a structured prompt containing:

1. The 27 intent definitions
2. Few-shot examples
3. Explicit rules for ambiguous/overlapping intents
4. A strict output constraint requiring exactly one valid intent ID

The notebook also includes:

- Few-shot prompt construction
- Prompt refinement using the validation set
- Constrained output validation
- Fuzzy-match fallback for malformed model outputs
- Accuracy, precision, recall, and macro F1
- Confusion matrix
- Per-intent error analysis
- Latency benchmarking
- Held-out test evaluation
- Blind-test prediction generation
- Saving the final prompt template

**Notebook:**

```text
notebooks/Arabic_Intent_Classification_FewShot.ipynb
```

---

## Dataset

The dataset contains **598 Egyptian Arabic customer utterances** across 27 intents.

The split used by both tracks is:

| Split         | Size | Purpose                                    |
| ------------- | ---: | ------------------------------------------ |
| Train         |  418 | Model training / few-shot example sourcing |
| Validation    |   90 | Model selection / prompt refinement        |
| Held-out Test |   90 | Final evaluation                           |

The split is stratified across the 27 intents and uses a fixed random seed (`42`) for reproducibility.

The dataset is **not included in this repository**.

The notebooks expect:

```text
dataset.jsonl
```

to be available locally or in the execution environment.

Each JSONL record contains at least:

```json
{
  "customer_text": "...",
  "intent": "..."
}
```

and may also contain:

```json
{
  "agent_text": "..."
}
```

---

## Intents

The system supports the following 27 intent IDs:

```text
activate_line
add_data_package
add_family_line
app_navigation_assistance
ask_price
bill_inquiry
booking_reservation
cancel_addon
change_credit_limit
confirm_agree
convert_to_postpaid
decline_reject
downgrade_package
greeting
identify_self
inquiry_installment
package_renewal_inquiry
points_redemption
provide_number
request_agent_action
request_info
roaming_package
system_issue_complaint
thank_close
transfer_data_between_lines
transfer_ownership
update_account_info
```

### Important Intent Boundaries

Several intents have similar surface forms. The project explicitly handles these overlaps.

#### `identify_self` vs `provide_number`

- Full customer name → `identify_self`
- Phone/ID digits → `provide_number`

#### `app_navigation_assistance` vs `system_issue_complaint`

- Asking where/how to find something → `app_navigation_assistance`
- Reporting that something is broken, crashing, or not working → `system_issue_complaint`

#### `bill_inquiry` vs `ask_price`

- A charge/bill that already happened → `bill_inquiry`
- Asking about a cost before committing → `ask_price`

#### `inquiry_installment`

If the utterance is about device installments (`تقسيط`, `قسط`, etc.), the installment intent takes priority over general price or bill questions.

#### `ask_price` vs `roaming_package`

- Asking for a flat roaming cost/rate → `ask_price`
- Asking about roaming plans/packages or choosing between them → `roaming_package`

#### `package_renewal_inquiry` vs `roaming_package`

If the package is specifically a roaming package, the roaming intent takes priority.

#### `cancel_addon` vs `package_renewal_inquiry`

- Cancelling an add-on service → `cancel_addon`
- Cancelling automatic renewal → `package_renewal_inquiry`

#### `update_account_info` vs `change_credit_limit`

- Personal/contact/account information → `update_account_info`
- Credit/usage limit (`الليميت` / حد الائتمان) → `change_credit_limit`

#### `add_family_line` vs `transfer_ownership`

- Changing membership of a family/shared plan → `add_family_line`
- Changing the owner of a line → `transfer_ownership`

#### `request_info`

`request_info` is intended as the fallback intent when none of the more specific intents applies.

---

## Evaluation

### Track A — AraBERT

Validation performance:

```text
Macro F1: 0.849
Accuracy: 0.856
```

Held-out test performance:

```text
Macro F1: 0.7495
Accuracy: 0.7667
```

The Track A notebook also reports a CPU batch-size-1 median latency of approximately:

```text
178.95 ms
```

The three lowest-F1 validation intents were:

```text
ask_price
confirm_agree
decline_reject
```

---

### Track C — Few-Shot Qwen

Validation performance after iterative prompt refinement:

```text
Macro F1: 0.934
Accuracy: 0.933
```

Held-out test performance:

```text
Macro F1: 0.8402
Accuracy: 0.8556
```

The held-out test score is the primary estimate of generalization because the validation set was used during prompt refinement.

Track C reported batch-size-1 median latency of approximately:

```text
2929.4 ms
```

on CUDA.

The three lowest-F1 held-out test intents were:

```text
transfer_ownership
confirm_agree
transfer_data_between_lines
```

The constrained-output fuzzy fallback was triggered on approximately:

```text
0.4%
```

of cumulative validation + test calls, with no fully unparseable outputs.

---

## Track Comparison

| Metric                 | Track A — AraBERT | Track C — Qwen Few-Shot |
| ---------------------- | ----------------: | ----------------------: |
| Training               |       Fine-tuning |             No training |
| Model                  |           AraBERT |     Qwen2.5-7B-Instruct |
| Validation Macro F1    |             0.849 |                   0.934 |
| Held-out Test Macro F1 |            0.7495 |              **0.8402** |
| Held-out Test Accuracy |            0.7667 |              **0.8556** |
| Quantization           |                 — |                   4-bit |
| Median latency         |       ~179 ms CPU |           ~2929 ms CUDA |
| Model artifact         |  Fine-tuned model |         Prompt template |

### Interpretation

On the held-out test set, the few-shot Qwen approach achieved higher macro F1 and accuracy than the fine-tuned AraBERT approach.

However, the two approaches have different deployment trade-offs:

- **AraBERT** is much lighter and substantially faster for single-utterance inference.
- **Qwen Few-Shot** achieved better held-out test performance but has significantly higher inference latency and requires a larger language model.
- Track C also requires maintaining the prompt, intent definitions, few-shot examples, and output-validation rules.

---

## Usage

### Track A

Open:

```text
notebooks/Arabic_Intent_Classification_BERT.ipynb
```

Install the required packages:

```bash
pip install transformers datasets evaluate accelerate arabert scikit-learn pandas matplotlib seaborn
```

Make sure `dataset.jsonl` is available in the notebook working directory.

The notebook trains the model and saves the resulting model and tokenizer to:

```text
intent_model_final/
```

For single-utterance inference:

```python
predict_intent(
    customer_text,
    agent_text=""
)
```

The function returns one of the 27 predefined intent IDs.

---

### Track C

Open:

```text
notebooks/Arabic_Intent_Classification_FewShot.ipynb
```

Install the required packages:

```bash
pip install transformers accelerate bitsandbytes scikit-learn pandas matplotlib seaborn
```

A GPU is recommended because the notebook uses the 7B Qwen model.

The inference function is:

```python
predict_intent_llm(
    customer_text,
    agent_text=""
)
```

It returns:

```text
(predicted_intent, raw_model_output)
```

The final prompt is saved as:

```text
prompt_template.txt
```

---

## Blind Test

Both approaches support inference on a blind test JSONL file.

Expected input:

```text
blind_test.jsonl
```

Each record should contain:

```json
{
  "customer_text": "...",
  "agent_text": "..."
}
```

`agent_text` is optional.

### Track A Output

```text
predictions.txt
```

### Track C Output

```text
predictions_trackC.txt
```

Each output file contains one predicted intent ID per line, preserving the order of the input records.

---

## Reproducibility

Both tracks use:

```text
SEED = 42
```

The dataset split is performed from the single source file `dataset.jsonl`, rather than storing separate train/validation/test files.

This keeps the split reproducible when the same dataset and random seed are used.

---

## Outputs

Evaluation artifacts are stored under:

```text
evaluation/
```

Recommended files:

```text
evaluation/
├── evaluation_summary_trackA.txt
├── evaluation_summary_trackC.txt
└── track_comparison.txt
```

The Track C prompt is stored under:

```text
prompts/
└── prompt_template.txt
```

---

## Notes

- The dataset is not included in this repository.
- Model checkpoints are not required to reproduce the notebooks from scratch.
- GPU execution is recommended for Track C.
- Track C's validation score should not be interpreted as its final generalization performance because the validation set was used for iterative prompt refinement.
- The held-out test set is the primary comparison point between the two approaches.

---
