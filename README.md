# Thesis

This repository accompanies my Master's thesis:
Selective Key-Value Cache Updates for Textbook Knowledge
University of Bayreuth — 2026
The thesis investigates how previously computed KV caches can be selectively reused when structured knowledge sources are modified.

# Selective KV-Cache Updates for Textbook Knowledge

This project implements and evaluates a selective Key-Value (KV) cache
update strategy for large language models when a previously cached
knowledge document is modified.

The main idea is simple: when only one part of a document changes,
there is no need to recompute the complete KV cache. We reuse the
cached prefix before the first changed segment and recompute only the
changed segment and its causally dependent suffix.

This project was developed as part of my Master's thesis at the
University of Bayreuth.

---

## Approach

The document is divided into segments (chapters in the current
implementation).

When a segment changes:

1. Identify the first changed segment.
2. Reuse the KV cache corresponding to the unchanged prefix.
3. Recompute the changed segment together with all following segments.
4. Combine the reused prefix cache and the newly computed tail.
5. Use the updated cache for question answering.

---

## Our Implementation

<img width="449" height="862" alt="image" src="https://github.com/user-attachments/assets/a0692f68-89e9-48f4-965c-5aeb0f905462" />

---

## Architecture

[experimental_procedure.drawio.pdf](https://github.com/user-attachments/files/32224594/experimental_procedure.drawio.pdf)

---

## What I Implemented

The project contains the main components required for selective KV-cache updates:
•	Cache construction for the complete knowledge source 
•	Chapter/segment indexing for mapping document regions to token positions 
•	Cache storage for persisting generated KV caches 
•	Cache cropping for extracting the reusable prefix 
•	Selective recomputation using the prefix cache as past_key_values 
•	Updated cache construction for the modified document 
•	Question answering evaluation using the updated cache 
•	Runtime and speedup evaluation against full KV-cache recomputation 
The current prototype uses chapter-level segmentation, while the underlying approach is applicable more generally to segmented knowledge sources.

---

## Experimental Design

Four document modification scenarios were evaluated.
The changed chapter was placed at different positions in the knowledge source in order to measure how the location of a modification affects cache reuse and update efficiency.

[runtime_comparison.pdf](https://github.com/user-attachments/files/32225224/runtime_comparison.pdf)
[Reusable Prefix Ratio and Selective-Update Speedup.pdf](https://github.com/user-attachments/files/32225223/Reusable.Prefix.Ratio.and.Selective-Update.Speedup.pdf)
[experimental_procedure.drawio.pdf](https://github.com/user-attachments/files/32225222/experimental_procedure.drawio.pdf)
[cache_implementation.drawio.pdf](https://github.com/user-attachments/files/32225221/cache_implementation.drawio.pdf)
[squad_preparation.drawio.pdf](https://github.com/user-attachments/files/32225219/squad_preparation.drawio.pdf)


This setup makes it possible to evaluate the method under different modification positions rather than measuring only a single update case.

---

## Results

<img width="713" height="442" alt="image" src="https://github.com/user-attachments/assets/d206bd48-3776-4053-91e2-762b45de683b" />

The selective update strategy was compared with rebuilding the complete KV cache from the updated document.

<img width="596" height="388" alt="image" src="https://github.com/user-attachments/assets/4fd3883a-3538-4a6d-85bc-12b9b65c21b5" />

---

## [Selective Update KV Cache.pptx](https://github.com/user-attachments/files/32225411/Selective.Update.KV.Cache.pptx)
Response Quality 

In addition to runtime, the generated answers were evaluated using BERTScore against the corresponding reference answers.
For the evaluated scenarios, the selective update produced the same measured BERTScore as the full-cache baseline:

<img width="598" height="365" alt="image" src="https://github.com/user-attachments/assets/2607f05a-0503-4cab-a11c-2f8bfdd554c3" />

No measured response-quality degradation was observed under the tested conditions.

---

## Key Results

The experiments show that:
•	A complete KV-cache rebuild is not always necessary after a document update. 
•	The unchanged prefix can be reused instead of being recomputed. 
•	Because of causal dependencies, the modified segment and its suffix must be recomputed. 
•	The amount of reusable cache depends on the position of the modification. 
•	The best evaluated case reused 85.67% of the updated document tokens. 
•	The best measured update achieved a 5.62× speedup over full cache recomputation. 
•	The selective approach showed no measured degradation in response quality in the evaluated QA experiments. 

---

## Installation

Requirements
The project was developed using:
•	Python 
•	PyTorch 
•	Hugging Face Transformers 
•	CUDA 
•	NVIDIA GPU 
The experiments were run on an NVIDIA RTX 4090 with 24 GB VRAM.
Install the Python dependencies using the project's dependency file:
pip install -r requirements.txt
If a CUDA-enabled PyTorch installation is required, use the installation command appropriate for your CUDA environment.

---

## Model

The final experiments use:
meta-llama/Llama-3.1-8B-Instruct
The model must be available through Hugging Face before running the experiments.
Depending on your environment, authentication may be required to access the model.

---

## Running the Project

The repository contains the scripts used to:
1.	Build the initial KV cache. 
2.	Create the token-to-chapter index. 
3.	Store the generated cache. 
4.	Apply a document modification. 
5.	Reuse the valid prefix cache. 
6.	Recompute the affected tail. 
7.	Evaluate update time and speedup. 
8.	Run question answering using the updated cache. 
Refer to the experiment scripts in the repository for the exact configuration used in the thesis.

---

## Evaluation Metrics
The implementation evaluates four main aspects:
1. KV-Cache Update Time
Measures the time required to construct the updated cache.
2. Reusable Prefix Ratio
Measures how much of the updated document can reuse the previous cache:
Reuse Ratio = Reused Tokens / Updated Document Tokens
3. Speedup
Compares selective cache updates with full KV-cache recomputation:
Speedup = Full Rebuild Time / Selective Update Time
4. Response Quality
Evaluates the generated answers against reference answers using BERTScore.

---

## Limitations

The current implementation and evaluation have several limitations.
•	The experimental knowledge source is relatively small. 
•	The evaluation uses a single language model. 
•	Experiments were conducted on one GPU configuration. 
•	The QA evaluation uses 10 question-answer pairs. 
•	The current prototype uses chapter-level segmentation. 
•	The evaluated modification scenarios use predefined changed chapters rather than a complete automatic document change-detection pipeline. 
These limitations define the scope of the current thesis implementation rather than the general applicability of the underlying idea.

---

## Future Work

Several extensions could make the approach more general and practical:
•	Automatic detection of changed document segments 
•	Longer knowledge sources 
•	Repeated document updates 
•	Adaptive segmentation strategies 
•	Multiple simultaneous modifications 
•	Integration with Cache-Augmented Generation systems 
•	Evaluation of memory and cache-transfer costs 
•	Evaluation under concurrent workloads 
•	Larger and more diverse QA benchmarks 

---

## Thesis

This repository accompanies my Master's thesis:
Selective Key-Value Cache Updates for Textbook Knowledge
University of Bayreuth — 2026
The thesis investigates how previously computed KV caches can be selectively reused when structured knowledge sources are modified.

---

## Citation

If you use this work, please cite my thesis:
Shubham Chheta.
Selective Key-Value Cache Updates for Textbook Knowledge.
Master's Thesis, University of Bayreuth, 2026.
