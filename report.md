<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="github-markdown.css">
<style>
	.markdown-body {
		box-sizing: border-box;
		min-width: 200px;
		max-width: 980px;
		margin: 0 auto;
		/* padding: 10px; */
	}

	@media (max-width: 767px) {
		.markdown-body {
			padding: 15px;
		}
	}
</style>

<article class="markdown-body">

# CS598JH Assignment Report

Shengjie Ma sm138@illinois.edu

In this assignment, I explored 3 improvement strategies provided in the instructions, and proposed my own improvement - controlled generation.

## 1. Experimental Result

| Strategy                        | Accuracy (%) |
| ------------------------------- | ------------ |
| baseline                        | 76.14        |
| JSONlization                    | 80.72        |
| domain knowledge                | 77.78        |
| JSONlization + domain knowledge | 80.07        |
| controlled generation           | 84.97        |

Table 1. Experimental results on the MCQ dataset, showing accuracy (%) achieved by various strategies.

## 2. Improvement Strategies Discussion

### 2.1 JSONlization

**Motivation.** This strategy converts the retrieved context into JSON format. For example, text
```text
Disease psoriasis associates Gene SLC29A3. Disease psoriasis associates Gene BCL11B.
```
becomes 
```json
{
    "Diseases": {
        "Psoriasis": {
            "Genetic Associations": [
                {"Gene": "SLC29A3"},
                {"Gene": "BCL11B"}
            ]
        }
    }
}
```
The motivation of this approach is that 1) JSON is a structured format, so the model can infer relationship between diseases and its attributes more easily than plain text. In addition, the JSON format above consolidates statements with the same subject (Disease Psoriasis) and the same predicate (associates), thus may save some tokens.

**Implementation.** I propose gemini-1.5-flash to convert text to JSON, using the following prompt.
```text
Convert text into JSON format according to the structure demonstrated in the example.
EXAMPLE TEXT (omitted)

EXAMPLE JSON (omitted)

TEXT
{text}

JSON
```

**Discussion.** The accuracy improved by more than 4 percent, demonstrating the effectiveness of this improvement.

### 2.2 Domain Knowledge

**Motivation.** We can provide the model some prior knowledge to guide it make better use of the context. Firstly, none of the MCQ questions are related to symptoms, so symptoms information are useless. Secondly, for some questions we do not have the exact disease in the knowledge graph, but the retriever can possibly retrieve similar diseases. Therefore, we can remind our model that similar diseases tend to have the same gene associations, encouraging the model to make a reasonable guess.

**Implementation.** I simply append the domain knowledge to the retrieved context. In the assignment document contains another domain knowledge that provenance data is useless, but I doesn't include it because the baseline method does not retrieve provenance data anyway.

**Discussion.** The accuracy improved around 1.6% compared to the baseline. Although the improvement is small, by inspecting the model output, we can confirm the domain knowledge is helpful. For example, in this question,

> Out of the given list, which Gene is associated with triple-receptor negative breast cancer and idiopathic pulmonary fibrosis. Given list is: CRTC1, TERT,  ATG5,  WNT10A,  ULK4

The retrieved context does not have information about triple-receptor negative breast cancer, but it contains information about idiopathic pulmonary fibrosis.

> Disease idiopathic pulmonary fibrosis associates Gene TERT.

The baseline KG_RAG answers with:

> None of the provided genes (CRTC1, TERT, ATG5, WNT10A, ULK4) are associated with both triple-receptor negative breast cancer and idiopathic pulmonary fibrosis in the given context.

On the other hand, with domain knowledge, the model knows that similar diseases tend to have similar gene associations. So it is able to make a reasonable guess, yielding the correct answer: TERT.

### 2.3 Combining JSONlization and Domain Knowledge

**Motivation:** Simply combine the two optimizations to explore whether they can work together.

**Discussion.** The result is a little bit lower than solely using JSONlization. This indicates that although domain knowledge can improve the performance of the baseline KG_RAG, it cannot improve KG_RAG with JSONlization any further. For unknown reasons, when the model is prompted with JSON context, it acquires a stronger tendency to make a guess based on imperfect information. For example, JSONlization can pass the example in section 2.2 without domain knowledge, thus the domain knowledge become less needed.

### 2.4 Controlled Generation

**Motivation.** A common error is that the model answers a MCQ question with "I don't know the answer". In some tasks, it is important that the model knows what it doesn't know. However, for this specific MCQ task, the model should always choose one of the given options. We can use controlled generation to force the model to choose one option. In controlled generation, we define the expected model output format using a JSON schema, and Google will ensure the output meets the schema. Under the hood, when the LLM inference engine samples the next token, it only sample from a subset of tokens that meets the constraint [[1]](#references).

**Implementation.** Google's gemini API supports controlled generation, as documented [here](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output). I provide a JSON schema to specify the output format
```json
{
  "type": "object",
  "properties": {
    "answer": {
      "type": "string",
      "enum": ["option1", "option2", "option3"]
    }
  },
  "required": ["answer"]
}
```

**Discussion.** The accuracy improved by more than 8 percent, making Controlled Generation the most effective improvement in this report. In the baseline KG_RAG, although we have instructed the model to choose an answer in the system prompt, in some cases it does not follow the instruction and reply with "I don't know." Controlled Generation ensures the model always output an option, thus increasing the accuracy.

## References

[1] Koo, Terry, Frederick Liu, and Luheng He. "Automata-based constraints for language model decoding." arXiv preprint arXiv:2407.08103 (2024).

</article>