# Industrial Citation Policy

## Scope

This project uses a constrained literature policy for the industrial dataset and benchmark writing tasks.

All benchmark rationale, dataset-design justification, metric selection, and baseline framing should be derived from:

1. papers already included in this repository
2. internal wiki notes and topic summaries built from those papers
3. references explicitly cited by those papers

No claim should rely on an unrelated external source unless it is first connected back to this repository's paper graph.

## Why This Constraint Exists

This constraint keeps the benchmark design defensible.

The goal is not to assemble a generic survey from arbitrary sources. The goal is to build an industrial benchmark narrative that is traceable to the paper set you already curated. That makes later thesis writing, paper writing, and method justification cleaner, because every design item can be traced back to your own collection.

## Allowed Source Layers

### Layer 1: Direct Repository Papers

These are the strongest sources. They include:

- extracted markdown papers under `extracted_markdown`
- structured wiki source notes under `wiki-kb/wiki/sources`
- topic notes under `wiki-kb/wiki/topics`

These should be used first whenever possible.

### Layer 2: Repository-Derived Aggregates

These are allowed as navigation aids and summary anchors, not as stand-alone evidence:

- `slam_index.md`
- `robotics_index.md`
- `general_index.md`
- `general_subindexes/*`
- master reports and dashboards

They help identify representative papers, but the final paper text should cite underlying papers rather than only citing summary tables.

### Layer 3: References Cited By Repository Papers

These are allowed when:

- the concept is important but only indirectly covered in the repository
- a benchmark metric or protocol clearly originates in an earlier cited work
- you need a historical anchor for a commonly used SLAM or reconstruction metric

However, these external references should still be justified through a repository paper that points to them.

## Not Allowed As Primary Justification

The following should not be used as primary support for the industrial benchmark drafts unless later added into the repository literature chain:

- random web articles
- unrelated blog posts
- unsourced benchmark conventions
- metrics copied from papers outside the repository graph without traceability
- generic statements such as "industry usually does X" without repository-backed evidence

## Practical Writing Rule

For every important design choice in the paper draft, ask:

1. Which repository paper supports this?
2. If the repository paper is not the original source, which reference does it point to?
3. Can the claim be written in a narrower way if the evidence is only partial?

If these questions cannot be answered, the sentence should be softened, scoped down, or removed.

## Recommended Citation Workflow

### Step 1: Find the local anchor

Start from one of these:

- `INDUSTRIAL_DATASET_LITERATURE_BASIS.md`
- `slam_index.md`
- `robotics_index.md`
- `general_index.md`
- `wiki-kb/wiki/topics/3DGS-SLAM综述.md`

### Step 2: Move to representative papers

Prefer direct papers that match one of these industrial benchmark dimensions:

- textureless or repetitive environments
- low-light or photometric degradation
- dynamic environments
- tracking robustness and relocalization
- dense reconstruction quality
- robotics localization and active reconstruction

### Step 3: Expand only through explicit references

If a classical benchmark, metric, or older baseline is needed, expand only through the reference list of those representative papers.

### Step 4: Keep claims traceable

When writing the thesis or paper section, each paragraph should be traceable to a small set of concrete papers, not to a vague topic bucket.

## Immediate Implication For Current Drafts

The following draft files should be interpreted under this policy:

- `INDUSTRIAL_DATASET_SPEC.md`
- `INDUSTRIAL_EVAL_SPEC.md`
- `INDUSTRIAL_DATASET_LITERATURE_BASIS.md`
- `INDUSTRIAL_BENCHMARK_SECTION_DRAFT.md`
- `INDUSTRIAL_PROBLEM_FORMULATION_DRAFT.md`
- `INDUSTRIAL_EXPERIMENTS_SECTION_DRAFT.md`

These drafts are writing skeletons. Before final paper submission, the placeholder literature claims in them should be backed by citations drawn from the repository paper set and, where necessary, from explicit references cited by those papers.
