# Data layout

Both datasets are parquet files in the verl format, with the columns
`data_source`, `prompt`, `ability`, `reward_model` and `extra_info`.
`prompt` is a list of chat messages, and `reward_model` is a dict with
`style` and `ground_truth`.

| file | content | rows |
|---|---|---|
| `data/dapo17k/train.parquet` | DAPO-Math-17k training prompts (`data_source = math_dapo`) | 17,398 |
| `data/aime/test.parquet` | AIME 2022, 2023 and 2024 (I and II), one problem per row | 90 |

`data/dapo17k/train.parquet` can be built from `BytedTsinghua-SIA/DAPO-Math-17k`
with the standard verl preprocessing. `data/aime/test.parquet` holds the problem
statement as the user message and the integer answer as `ground_truth`.
Answers are graded on the last `\boxed{}` expression with
`verl.utils.reward_score.default_compute_score`.
