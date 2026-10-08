# The Thinking Leakage in Reinforcement Learning for LLMs

Code for the experiments in *The Thinking Leakage in Reinforcement Learning for LLMs*.

Hybrid reasoning models such as Qwen3 expose two modes under the same weights: THINK mode,
where the template opens a `<think>` block, and NO-THINK mode, where the template appends an
empty block `<think>\n\n</think>\n\n`. We train Qwen3-1.7B, 4B and 8B with GRPO in NO-THINK
mode and show that the training moves the model toward its own THINK mode: the responses
become longer and more reflective, the original model judges them as more THINK-like, and
the activations move along the direction that separates the two modes. Interventions on
weights, activations and attention heads show that this shift accounts for most of the
accuracy gain, and the GRPO advantages at the start of training already favour THINK-like
responses.

## Layout

```
train/      GRPO training launcher (verl), LoRA extraction, scaled weight updates
scripts/    generation, measurements and interventions
jobs/       one shell script per experiment, in the order of the paper
data/       expected dataset layout (see data/README.md)
```

All commands run from the repository root. Outputs go to `analysis_out/` and adapters to
`checkpoints/`; both are ignored by git.

## Setup

```
pip install -r requirements.txt
```

Training and answer grading use [verl](https://github.com/verl-project/verl) (we used commit
`bcb63864`). Generation uses vLLM with hot-swapped LoRA adapters, and the measurements use
`transformers` with hooks on the residual stream. Every experiment in the paper ran on a
single H200 GPU.

Set the base models with `BASE_8B`, `BASE_4B` and `BASE_17B` (Hugging Face ids or local
paths; defaults are `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B` and `Qwen/Qwen3-1.7B`).

## Training

```
MODEL_PATH=Qwen/Qwen3-8B   EXPERIMENT_NAME=qwen3_8b_grpo_nothink   MAX_RESPONSE_LENGTH=8192  bash train/train_grpo_nothink.sh
MODEL_PATH=Qwen/Qwen3-4B   EXPERIMENT_NAME=qwen3_4b_grpo_nothink   MAX_RESPONSE_LENGTH=16384 bash train/train_grpo_nothink.sh
MODEL_PATH=Qwen/Qwen3-1.7B EXPERIMENT_NAME=qwen3_1p7b_grpo_nothink MAX_RESPONSE_LENGTH=16384 bash train/train_grpo_nothink.sh
```

GRPO on DAPO-Math-17k with the NO-THINK template (`enable_thinking=False`): 256 prompts and 5
responses per prompt at temperature 1, four minibatches of 64 prompts per step, clip ratio
0.2, token-mean loss, KL penalty 0.001 to the original model, rank-64 LoRA (alpha 64) on every
linear layer, AdamW with learning rate 1e-5, weight decay 0.01 and gradient clipping 1.0.
One epoch is 67 steps; a checkpoint is written every 5 steps.

The checkpoints are converted to standalone PEFT adapters, one per saved step:

```
MODEL=8B CKPT_ROOT=checkpoints/thinking_leakage/qwen3_8b_grpo_nothink bash jobs/00_extract_adapters.sh
```

## Experiments

Each job takes `MODEL=8B`, `MODEL=4B` or `MODEL=1.7B` where the experiment was run at several
sizes. The jobs are written for a single GPU and skip work whose output already exists.

| job | paper | what it does |
|---|---|---|
| `01_trajectory.sh` | Fig. 1 | NO-THINK generation on AIME at every checkpoint (2 samples per problem), the THINK-mode reference, the think direction and the thinking activation shift α |
| `02_continue_32k.sh` | Fig. 1, TWR | continues truncated responses to the 32k budget, dumps per-token LLR for the think win rate, recomputes α on the full responses |
| `03_scaled_updates.sh` | §5.1, Fig. 2 | builds `W0 + cΔW` adapters from the step-15 and step-67 updates and evaluates them |
| `04_think_subspace.sh` | §5.2 | estimates the 8-dimensional think subspace, the PCA and random controls, per layer |
| `05_patching.sh` | §5.2, Fig. 3 | cross-model patching of the subspace during generation, both directions, all controls, continued to 32k |
| `06_steering.sh` | App. | steers the original model along its think direction at layer 18 |
| `07_heads.sh` | §5.3, Tables 1–2 | attention to the empty think block, attention knockout, single-head patching and the head summary |
| `08_write_decomp.sh` | §5.3 | decomposes the shift along the think direction into the part written by the LoRA update and the part produced by the pretrained weights |
| `09_grpo_signal.sh` | §6, Fig. 4, Tables 3–4 | samples groups from the original model, scores their think-likeness, relates it to the GRPO advantages, and takes one gradient step |
| `10_think_position.sh` | App. | where `</think>` appears in the responses of the RL-trained models |
| `11_think_reference.sh` | §4 | THINK-mode accuracy of the original models with 8 samples per problem at the 32k budget |
| `12_adapter_surgery.sh` | App. | removes the think-subspace component from the LoRA update |

Run order: `00` and `01` for every model size, `02` for every model size, then `03` to `12` in
any order (`05`, `06`, `08`, `09` and `12` need `04`; `09` and `08` use the vectors written by `01`
and `02`).

## Scripts

| script | measurement |
|---|---|
| `think_leak_gen.py` | vLLM generation for a list of adapters under the NO-THINK template, plus the THINK reference; accuracy, length, reflection words, literal `<think>`/`</think>` tags |
| `continue_gen_dir.py`, `think_continue_long.py` | resume responses that hit a budget, so that a longer budget does not require regenerating |
| `think_vector.py` | think direction `v` (mean activation difference between the two prefixes on fixed text), thinking activation shift α with cross-fitting, cosine with `v`, optional steering |
| `llr_dump.py` | per-response log-likelihood ratio of the original model under the THINK and NO-THINK prefixes from which the think win rate is computed |
| `think_subspace.py` | mean difference plus the leading directions of the per-token differences on 128 training prompts; PCA and random controls |
| `think_lesion.py`, `think_lesion_continue.py`, `think_lesion_report.py` | cross-model patching of a subspace at every layer during generation; paired bootstrap over problems |
| `steer_continue.py` | adds `a·v` at one layer during generation |
| `circuit_gate.py`, `circuit_gate_attn.py` | attention of every head to the four tokens of the empty think block; knockout of that attention; α and the relative KL reduction `R_KL` |
| `circuit_route.py`, `circuit_route_summary.py` | patches the output of one head from the RL-trained model into the original model; patching effect per head and the hypergeometric test against the heads that attend to the block |
| `think_write_decomp.py` | LoRA-written versus pretrained-weight contributions to the shift along `v` |
| `think_position.py` | position of `</think>` inside the responses |
| `think_dynamics_sample.py`, `think_dynamics_score.py`, `think_dynamics_report.py` | groups of five responses from the original model on training prompts, GRPO advantages, think-likeness of correct and wrong responses within and across groups |
| `think_proj_full.py` | α of every sampled response, used to stratify by the number of correct responses in the group |
| `grad_probe.py` | one GRPO gradient step on a fresh LoRA adapter with the full signal, one sign of the advantages only, shuffled advantages or uniform weights |
| `adapter_surgery.py` | edits a LoRA adapter inside or outside the think subspace |
| `passk_eval.py`, `paired_boot.py` | pass@k evaluation and the problem-level paired bootstrap |

## Output layout

The jobs write their results to the following locations.

| path | content |
|---|---|
| `analysis_out/think_leak/qwen_s2`, `analysis_out/think_leak4b/nothink`, `analysis_out/think_leak4b/nothink17` | 8B, 4B and 1.7B trajectories; `*_32k` holds the continued responses and the LLR dumps |
| `analysis_out/think_leak/qwen_amp_s2` | scaled updates |
| `analysis_out/think_leak/surgery/subspace.pt` | think subspace of the 8B model |
| `analysis_out/think_leak/lesion`, `analysis_out/think_leak/lesion32k` | patching at the 8k and 32k budgets |
| `analysis_out/think_leak/steer32k` | steering |
| `analysis_out/think_circuit` | attention heads |
| `analysis_out/think_leak/decomp` | write decomposition |
| `analysis_out/think_dyn/q8s2`, `analysis_out/think_leak/grad_probe.json` | GRPO signal and gradient probe |
| `analysis_out/think_leak/position` | `</think>` positions |
| `analysis_out/think_leak/ceiling`, `.../nothink/ceiling`, `.../nothink17/ceiling` | THINK-mode references at 8 samples per problem |
| `checkpoints/walk_variants/*` | extracted, scaled and edited adapters |

## Citation

```
@article{zhu2026thinkingleakage,
  title   = {The Thinking Leakage in Reinforcement Learning for LLMs},
  author  = {Zhu, Wenxuan and Liu, Shuming and An, Bang and Aliev, Aznaur and Li, Bing and Yang, Yibo and Ghanem, Bernard},
  year    = {2026}
}
```
