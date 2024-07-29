# Retrieval Augmented Generation (RAG) for PDF Question Answering
![Workflow Image](workflow.png)
This repository implements a Retrieval Augmented Generation (RAG) system for answering questions based on PDF documents. The project leverages powerful language models and embedding techniques within Google Colab to enhance the accuracy and efficiency of question answering.

## Key Features

* **Document Processing:** Extracts text from PDFs, performs cleaning and normalization, and divides the content into manageable chunks.
* **Embedding Generation:** Utilizes pre-trained SentenceTransformer models (e.g., all-MiniLM-L6-v2) to generate embeddings for each text chunk.
* **Similarity Search:** Employs efficient similarity search techniques (e.g., dot product) to retrieve relevant text chunks based on user queries.
* **Question Answering:** Integrates retrieved text chunks with Large Language Models (LLMs) to generate comprehensive and contextually relevant answers.

## Technical Details

* **PDF Processing:** PyMuPDF is used for extracting text from PDF documents.
* **Text Processing:** spaCy is used for sentence segmentation and tokenization.
* **Embedding Models:** SentenceTransformers library provides pre-trained models for generating text embeddings.
* **Hardware Acceleration:** Leverages GPU acceleration (if available) for faster embedding generation and similarity search within Google Colab.

## Usage

1. **Open the Notebook in Google Colab:**
   Access the provided Jupyter Notebook through Google Colab.

2. **Prepare PDF Document:**
   Upload your PDF document to your Google Colab environment.

3. **Run the Notebook:**
   Execute the notebook cells sequentially to process the document, generate embeddings, and perform question answering.

## Future Enhancements

* **Vector Database Integration:** Explore the use of vector databases (e.g., Faiss) for efficient storage and retrieval of embeddings, especially for large datasets.
* **Fine-tuning Embedding Models:** Fine-tune embedding models on domain-specific data to improve retrieval accuracy.
* **Dynamic Chunk Sizing:** Implement adaptive chunk sizing based on content complexity and embedding model limitations.
* **User Interface:** Develop a user-friendly interface for interacting with the RAG system.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
