# Full-scale reproduction (paper grid)

For reproducing the paper's full evaluation grid (4 paper classifiers
× 4 K-sweep generative imputers + references × 3 mechanisms × 3 rates
× 6 K values × cohort × target × seed). Requires installing
third-party model implementations and either GPU compute locally or
the RunPod Serverless deployment described below.

The harness ships **scaffold stubs** for each paper method; running
them out of the box returns
``status: "error: NotImplementedError: ..."``. To reproduce paper-level
numbers, replace the stubs with the third-party reference
implementations listed in the next section.

## Methods

| Method | Type | Code | Public reference |
|---|---|---|---|
| **TabPFN-v2**  | classifier | scaffold stub | `pip install tabpfn` ; weights via HuggingFace (e.g. `Prior-Labs/TabPFN-v2-clf`) — **HuggingFace ToS acceptance required** |
| **TabICL-v2**  | classifier | scaffold stub | `pip install tabicl` ; see Qu et al. 2025 for the public release URL + checkpoints |
| **TabDPT**     | classifier | scaffold stub | `pip install tabdpt` ; see Ma et al. 2025 for the public release URL + checkpoints |
| **MaskMLP**    | classifier | scaffold stub | The Shangguan et al. 2024 reference is **not publicly released**; the recipe is described in the paper's Method |
| **MIRI**       | imputer    | scaffold stub | See Yu et al. 2025 for the public release |
| **TabCSDI**    | imputer    | scaffold stub | See Zheng et al. 2022 for the public release |
| **DiffPuter**  | imputer    | scaffold stub | See Zhang et al. 2025 for the public release |
| **CFMI**       | imputer    | scaffold stub | See Simkus et al. 2025 for the public release |
| **HGB**, **LogReg** | classifier | concrete (sklearn stand-ins) | Not in paper; quick-start runnable demo only |
| **MICE**, **MissForest** | imputer | concrete | sklearn `IterativeImputer` and `miceforest` |
| **NATIVE**, **MEAN**, **ZEROS** | imputer | concrete | passthrough / sklearn `SimpleImputer` |

Each stub raises `NotImplementedError` with the citation and recipe in
the message text; `src/n26/classifiers/tabpfn.py` shows the expected
lazy-import pattern. To wire one in:

1. `pip install <package>` (and accept any HuggingFace ToS).
2. Replace the body of the corresponding `fit` / `fit_transform`
   method with a call to the reference implementation.

### A note on HuggingFace gating

TabPFN-v2 (and typically TabICL-v2 / TabDPT) download model weights
from HuggingFace on first use. Some checkpoints are **gated**: the
download URL only resolves after you accept the model card's terms of
use on huggingface.co. After accepting, set `HF_TOKEN` (or run
`huggingface-cli login`) before launching cells that use those
classifiers. For RunPod deployment, set `HF_TOKEN` in the endpoint's
environment variables.

## Build the image

`handler.py` is the per-job entry point: receives
`{"input": {"experiment_id": int, "run_name": str}}`, runs the
matching manifest cell via `scripts/run_cell.py`, writes the per-fold
JSON to the Network Volume, returns a small status payload.

```bash
DOCKER_USER="<your-dockerhub-username>"
TAG="$(git rev-parse --short HEAD 2>/dev/null || echo latest)"

docker build -t "$DOCKER_USER/reallit-harness:$TAG" \
             -t "$DOCKER_USER/reallit-harness:latest" \
             -f deploy/runpod/Dockerfile .

bash deploy/runpod/test_image_locally.sh  
docker push "$DOCKER_USER/reallit-harness:$TAG"
docker push "$DOCKER_USER/reallit-harness:latest"
```

Point a RunPod Serverless endpoint at the pushed tag.

## Submit a run

```bash
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...
export HF_TOKEN=... 

python deploy/runpod/submit_runpod.py \
    --name myrun \
    --datasets D1,D2,D3-G1-2,D3-G3,D4 \
    --imputers NATIVE,MEAN,ZEROS,MICE,MissForest,MIRI,TabCSDI,DiffPuter,CFMI \
    --classifiers MaskMLP,TabPFN-v2,TabICL-v2,TabDPT \
    --regimes MCAR,MAR,MNAR \
    --rates 10,20,30 \
    --seeds 5

python deploy/runpod/collect_results.py --name myrun
```

Cells whose method is still a stub return `status: "error:
NotImplementedError: ..."`; cells whose dataset is `requires_external_data`
(D1, D3-G1-2, D3-G3) and whose data file is missing return
`status: "error: FileNotFoundError: ..."`.

## Aggregate

```bash
python scripts/aggregate.py \
    --results-dir results --run-name myrun \
    --manifest docs/ablation_matrix.csv \
    --merge-into results/myrun/manifest_filled.csv
```

## License

CC BY 4.0. See [LICENSE](../../LICENSE).
