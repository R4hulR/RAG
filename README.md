# Retrieval Augmented Generation (RAG) for PDF Question Answering

![Workflow Image](flowchart.png)

This repository implements a fully functional Retrieval Augmented Generation (RAG) system for answering questions based on PDF documents. The project leverages powerful language models and embedding techniques to enhance the accuracy and efficiency of question answering.

## Key Features

* **Document Processing:** Extracts text from PDFs, performs cleaning and normalization, and divides the content into manageable chunks.
* **Embedding Generation:** Utilizes pre-trained SentenceTransformer models (e.g., all-MiniLM-L6-v2) to generate embeddings for each text chunk.
* **Similarity Search:** Employs efficient similarity search techniques (e.g., dot product) to retrieve relevant text chunks based on user queries.
* **Question Answering:** Integrates retrieved text chunks with a Large Language Model (LLM) to generate comprehensive and contextually relevant answers.

## Technical Details

* **PDF Processing:** PyMuPDF is used for extracting text from PDF documents.
* **Text Processing:** spaCy is used for sentence segmentation.
* **Tokenization:** Hugging Face's Transformers library is used for tokenization, specifically the BERT base uncased tokenizer.
* **Embedding Models:** SentenceTransformers library provides pre-trained models for generating text embeddings.
* **LLM Integration:** Successfully implemented using Google's Gemma models on Kaggle's T4 GPU.
* **Hardware Acceleration:** Leverages GPU acceleration for faster embedding generation, similarity search, and LLM inference.

## Implementation Details

### GPU Memory Management
The system automatically detects available GPU memory and selects an appropriate model:
- < 5.1 GB: Warns about insufficient memory
- 5.1 - 8 GB: Uses Gemma 2B with 4-bit precision
- 8 - 19 GB: Uses Gemma 2B in float16 or Gemma 7B in 4-bit precision
- > 19 GB: Recommends Gemma 7B in 4-bit or float16 precision

### Model Loading
- Utilizes the Hugging Face Transformers library to load the Gemma model.
- Implements Flash Attention 2 if available for improved performance.
- Uses BitsAndBytes for quantization when necessary.

### RAG Pipeline
1. **Context Retrieval:** Uses a custom `retrieve_relevant_resources` function to find relevant text chunks based on the query.
2. **Prompt Formatting:** Employs a `prompt_formatter` function to create a structured prompt with context and query.
3. **Answer Generation:** Utilizes the loaded LLM to generate answers based on the formatted prompt.

### User Interface
The `ask` function provides a simple interface for users to query the system:
- Accepts a query and optional parameters (temperature, max tokens, etc.)
- Returns a formatted answer, with options to include context items and raw model output.

## Usage

1. **Set Up Environment:**
   - Use a Kaggle notebook with GPU acceleration enabled.
   - Install required libraries: `transformers`, `sentence-transformers`, `PyMuPDF`, `spacy`, `torch`, `flash-attn`, `bitsandbytes`.

2. **Prepare Data:**
   - Upload your PDF document to your Kaggle environment.
   - Process the document to extract text and generate embeddings.

3. **Run Queries:**
   Use the `ask` function to generate answers:
   ```python
   query = "How would you describe the Time Tombs to someone who has not read Hyperion?"
   answer = ask(query, temperature=0.7, max_new_tokens=512)
   print(answer)

## Future Enhancements

* **Vector Database Integration:** Explore the use of vector databases (e.g., Faiss) for efficient storage and retrieval of embeddings, especially for large datasets.
* **Fine-tuning Embedding Models:** Fine-tune embedding models on domain-specific data to improve retrieval accuracy.
* **Dynamic Chunk Sizing:** Implement adaptive chunk sizing based on content complexity and embedding model limitations.
* **User Interface:** Develop a user-friendly interface for interacting with the RAG system.
* **Multi-Platform Support:** Extend support for running the system on various cloud platforms and local setups with GPU acceleration.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.


## License

This project is licensed under the [MIT License](LICENSE).
