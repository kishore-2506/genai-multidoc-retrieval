## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
Researchers must synthesize findings across many papers quickly. Manually scanning PDFs is slow and error-prone. Build an agentic retrieval system that (1) indexes multiple PDF papers as tool objects, (2) uses a retriever to select the most relevant tools for a query, (3) invokes per-document tools (vector retrieval and summaries) and synthesizes concise, accurate answers, and (4) provides measurable evaluation signals for retrieval and synthesis quality.

### DESIGN STEPS:


### STEP 1: Ingest papers and create per-document tools

Convert each PDF into chunks and embeddings.

Build two tools per document:

vector_tool: performs similarity retrieval over chunks.

summary_tool: returns a compact summary or metadata of the paper.

Ensure tools expose metadata (title, authors, paper id, tool_type) to support retrieval and selection.


### STEP 2: Build a tool-object index and retriever

Collect all tool objects into a single list.

Create an ObjectIndex (backed by a vector index) over the tool objects.

Expose the index as a retriever with similarity_top_k to return the top-k tool objects for a query.


### STEP 3: Instantiate and run the function-calling agent

Create a function-calling agent worker using FunctionCallingAgentWorker.from_tools, passing the tool retriever and an LLM instance.

Provide a system prompt that instructs the agent to always use tools and not rely on external prior knowledge.

Run queries through AgentRunner(agent_worker) to let the agent retrieve tools, call them, and synthesize answers.

Log tool-selection decisions, function calls, and LLM responses for debugging and evaluation.

### PROGRAM:
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()

from pathlib import Path
from utils import get_doc_tools

papers = [
    "25642_SAVIOR_Sample_efficient_.pdf",
    "25645_Quantum_Inspired_Image_E.pdf",
    "25649_Improving_Developer_Emot.pdf",
]

paper_to_tools_dict = {}
for paper in papers:
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo")

from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
agent_worker = FunctionCallingAgentWorker.from_tools(initial_tools, llm=llm, verbose=True)
agent = AgentRunner(agent_worker)

resp1 = agent.query("Summarize the SAVIOR methodology and explain how it improves sample efficiency for OCR fine-tuning.")
print(str(resp1))

resp2 = agent.query("Describe the quantum-inspired image encoding framework (Q-GAF, Q-RP, Q-MTF) and explain how it improves financial time-series forecasting accuracy.")
print(str(resp2))

resp3 = agent.query("Explain how the CommiTune framework combines LLaMA-based augmentation with CodeBERT fine-tuning to improve developer emotion classification.")
print(str(resp3))
```
### OUTPUT:
```
Added user message to memory: Summarize the SAVIOR methodology and explain how it improves sample efficiency for OCR fine-tuning.
=== Calling Function ===
Calling function: vector_tool_25642_SAVIOR_Sample_efficient_ with args: {"query": "SAVIOR methodology"}
=== Function Output ===
SAVIOR methodology is a targeted data curation approach designed to align Vision-Language Models (VLMs) with the structural and semantic requirements of enterprise OCR. It focuses on maximizing coverage of high-impact failure modes with minimal samples by curating training data that preserves semantic meaning, layout structure, and hierarchical relationships needed by downstream LLMs in business document workflows. The methodology consists of three stages: Failure Mode Identification, Targeted Data Curation, and Balanced Dataset Construction. By addressing critical failure scenarios like vertical text orientation, fine print omission, multi-column layouts, stylized text, degraded image quality, mixed content types, and hierarchical structure loss, SAVIOR aims to improve OCR performance and enhance enterprise workflows.
=== LLM Response ===
The SAVIOR methodology is a targeted data curation approach that aims to align Vision-Language Models (VLMs) with the structural and semantic requirements of enterprise OCR. It focuses on maximizing coverage of high-impact failure modes with minimal samples by curating training data that preserves semantic meaning, layout structure, and hierarchical relationships needed by downstream LLMs in business document workflows. The methodology consists of three stages: Failure Mode Identification, Targeted Data Curation, and Balanced Dataset Construction. By addressing critical failure scenarios such as vertical text orientation, fine print omission, multi-column layouts, stylized text, degraded image quality, mixed content types, and hierarchical structure loss, SAVIOR aims to improve OCR performance and enhance enterprise workflows.
assistant: The SAVIOR methodology is a targeted data curation approach that aims to align Vision-Language Models (VLMs) with the structural and semantic requirements of enterprise OCR. It focuses on maximizing coverage of high-impact failure modes with minimal samples by curating training data that preserves semantic meaning, layout structure, and hierarchical relationships needed by downstream LLMs in business document workflows. The methodology consists of three stages: Failure Mode Identification, Targeted Data Curation, and Balanced Dataset Construction. By addressing critical failure scenarios such as vertical text orientation, fine print omission, multi-column layouts, stylized text, degraded image quality, mixed content types, and hierarchical structure loss, SAVIOR aims to improve OCR performance and enhance enterprise workflows.
Added user message to memory: Describe the quantum-inspired image encoding framework (Q-GAF, Q-RP, Q-MTF) and explain how it improves financial time-series forecasting accuracy.
=== Calling Function ===
Calling function: summary_tool_25645_Quantum_Inspired_Image_E with args: {"input": "Describe the quantum-inspired image encoding framework (Q-GAF, Q-RP, Q-MTF)"}
=== Function Output ===
The quantum-inspired image encoding framework, consisting of Quantum-GAF (Q-GAF), Quantum-RP (Q-RP), and Quantum-MTF (Q-MTF), introduces a methodology that transforms time-series data into image representations for prediction. Q-GAF encodes temporal correlations by combining both the amplitude and phase of quantum states, capturing richer dynamics compared to classical methods. Quantum-RP measures recurrence structures based on quantum state similarity using complex-valued amplitudes and phases, providing a more informative recurrence map than classical RP. Quantum-MTF discretizes quantum states via clustering and encodes their transition probabilities across time, reflecting both amplitude and phase evolution. These quantum analogues extend classical image representations like GAF, RP, and MTF, capturing complex amplitude-phase structures and enabling the representation of latent temporal dynamics and market phase patterns that are often undetected by classical models.
=== Calling Function ===
Calling function: summary_tool_25645_Quantum_Inspired_Image_E with args: {"input": "Explain how the quantum-inspired image encoding framework improves financial time-series forecasting accuracy."}
=== Function Output ===
The quantum-inspired image encoding framework enhances financial time-series forecasting accuracy by incorporating complex-domain representations that capture both probabilistic amplitudes and temporal dynamics. This approach expands the modeling capabilities beyond traditional methods by utilizing Gaussian soft encoding and phase-function encoding to embed richer temporal dynamics. Empirical studies using high-frequency data from market indices show that the quantum-inspired encodings achieve an average accuracy improvement of 2.6% over classical benchmarks and a 32.9% higher win rate when compared to identical baselines. The quantum variants outperform classical methods, showcasing a practical advantage in forecasting financial trends.
=== LLM Response ===
The quantum-inspired image encoding framework, which includes Quantum-GAF (Q-GAF), Quantum-RP (Q-RP), and Quantum-MTF (Q-MTF), transforms time-series data into image representations for prediction. Q-GAF captures temporal correlations by combining quantum states' amplitude and phase, providing richer dynamics than classical methods. Quantum-RP measures recurrence structures based on quantum state similarity using complex-valued amplitudes and phases, offering more informative recurrence maps. Quantum-MTF discretizes quantum states via clustering and encodes their transition probabilities across time, reflecting amplitude and phase evolution. These quantum analogues extend classical image representations, capturing complex amplitude-phase structures and latent temporal dynamics.

The quantum-inspired image encoding framework improves financial time-series forecasting accuracy by incorporating complex-domain representations that capture probabilistic amplitudes and temporal dynamics. It expands modeling capabilities beyond traditional methods by utilizing Gaussian soft encoding and phase-function encoding to embed richer temporal dynamics. Empirical studies using high-frequency data from market indices demonstrate that the quantum-inspired encodings achieve an average accuracy improvement of 2.6% over classical benchmarks and a 32.9% higher win rate compared to identical baselines. The quantum variants outperform classical methods, showcasing a practical advantage in forecasting financial trends.
assistant: The quantum-inspired image encoding framework, which includes Quantum-GAF (Q-GAF), Quantum-RP (Q-RP), and Quantum-MTF (Q-MTF), transforms time-series data into image representations for prediction. Q-GAF captures temporal correlations by combining quantum states' amplitude and phase, providing richer dynamics than classical methods. Quantum-RP measures recurrence structures based on quantum state similarity using complex-valued amplitudes and phases, offering more informative recurrence maps. Quantum-MTF discretizes quantum states via clustering and encodes their transition probabilities across time, reflecting amplitude and phase evolution. These quantum analogues extend classical image representations, capturing complex amplitude-phase structures and latent temporal dynamics.

The quantum-inspired image encoding framework improves financial time-series forecasting accuracy by incorporating complex-domain representations that capture probabilistic amplitudes and temporal dynamics. It expands modeling capabilities beyond traditional methods by utilizing Gaussian soft encoding and phase-function encoding to embed richer temporal dynamics. Empirical studies using high-frequency data from market indices demonstrate that the quantum-inspired encodings achieve an average accuracy improvement of 2.6% over classical benchmarks and a 32.9% higher win rate compared to identical baselines. The quantum variants outperform classical methods, showcasing a practical advantage in forecasting financial trends.
Added user message to memory: Explain how the CommiTune framework combines LLaMA-based augmentation with CodeBERT fine-tuning to improve developer emotion classification.
=== Calling Function ===
Calling function: vector_tool_25649_Improving_Developer_Emot with args: {"query": "Explain how the CommiTune framework combines LLaMA-based augmentation with CodeBERT fine-tuning to improve developer emotion classification."}
=== Function Output ===
The CommiTune framework combines LLaMA-based augmentation by fine-tuning a LLaMA model on a manually labeled dataset to generate augmented training examples. These augmented examples are then used to fine-tune CodeBERT, a code-aware encoder, on the expanded corpus. This hybrid approach leverages the strengths of generative models for diverse paraphrasing and domain-specific encoders for stable code-aware representations, resulting in improved developer emotion classification.
=== Calling Function ===
Calling function: summary_tool_25649_Improving_Developer_Emot with args: {"input": "Explain how the CommiTune framework combines LLaMA-based augmentation with CodeBERT fine-tuning to improve developer emotion classification."}
=== Function Output ===
The CommiTune framework enhances developer emotion classification by first fine-tuning a generative model on manually labeled training commits to align with a four-label taxonomy. Then, generative data augmentation is used to create two paraphrases for each training commit, resulting in an augmented dataset. Finally, CodeBERT is retrained on this augmented dataset, leading to significant improvements in performance metrics across all emotion categories, particularly in resolving ambiguities and achieving a state-of-the-art Macro F1-score of 0.82. This combined approach effectively bridges representational gaps in the baseline model, resulting in more accurate and nuanced classification of developer emotions in technical commit messages.
=== LLM Response ===
The CommiTune framework combines LLaMA-based augmentation by fine-tuning a LLaMA model on a manually labeled dataset to generate augmented training examples. These augmented examples are then used to fine-tune CodeBERT, a code-aware encoder, on the expanded corpus. This hybrid approach leverages the strengths of generative models for diverse paraphrasing and domain-specific encoders for stable code-aware representations, resulting in improved developer emotion classification.

In summary, the CommiTune framework enhances developer emotion classification by fine-tuning a generative model on manually labeled training commits to align with a four-label taxonomy. It then uses generative data augmentation to create two paraphrases for each training commit, resulting in an augmented dataset. Finally, CodeBERT is retrained on this augmented dataset, leading to significant improvements in performance metrics across all emotion categories, particularly in resolving ambiguities and achieving a state-of-the-art Macro F1-score of 0.82. This combined approach effectively bridges representational gaps in the baseline model, resulting in more accurate and nuanced classification of developer emotions in technical commit messages.
assistant: The CommiTune framework combines LLaMA-based augmentation by fine-tuning a LLaMA model on a manually labeled dataset to generate augmented training examples. These augmented examples are then used to fine-tune CodeBERT, a code-aware encoder, on the expanded corpus. This hybrid approach leverages the strengths of generative models for diverse paraphrasing and domain-specific encoders for stable code-aware representations, resulting in improved developer emotion classification.

In summary, the CommiTune framework enhances developer emotion classification by fine-tuning a generative model on manually labeled training commits to align with a four-label taxonomy. It then uses generative data augmentation to create two paraphrases for each training commit, resulting in an augmented dataset. Finally, CodeBERT is retrained on this augmented dataset, leading to significant improvements in performance metrics across all emotion categories, particularly in resolving ambiguities and achieving a state-of-the-art Macro F1-score of 0.82. This combined approach effectively bridges representational gaps in the baseline model, resulting in more accurate and nuanced classification of developer emotions in technical commit messages.
```

### RESULT:

The multidocument retrieval agent was successfully developed using LlamaIndex, efficiently retrieving, summarizing, and synthesizing information from multiple research papers to produce accurate and concise responses.
