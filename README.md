# DeepScale AI: A Safe and Strong Local RAG System 🚀

Welcome to **DeepScale AI**, a powerful and secure local Retrieval-Augmented Generation (RAG) system designed to help you interact with your documents in a conversational manner. Built with cutting-edge AI technologies, DeepScale AI allows you to process PDF and text documents, ask questions, and get accurate answers with references to the source material. Whether you're a researcher, student, or professional, DeepScale AI is here to make your document interaction seamless and efficient. 📚🤖

---

## Features 🌟

- **Document Processing**: Upload and process multiple PDF and text documents with ease. 📄
- **Conversational AI**: Engage in natural language conversations with your documents. 💬
- **Source References**: Get detailed references to the source material for every answer. 🔍
- **Local Processing**: All processing is done locally, ensuring data privacy and security. 🔒
- **Customizable Parameters**: Fine-tune the model's behavior with advanced parameters like temperature, response length, and top-k sampling. 🛠️
- **GPU Support**: Optimized for GPU acceleration for faster processing.

---

## Installation 🛠️

To get started with DeepScale AI, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Adarsh3315/DeepScaleAIV1.git
   cd DeepScaleAIV1
   ```

2. **Install Dependencies**:
   Make sure you have Python 3.8 or higher installed. Then, install the required dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   Start the DeepScale AI application by running:
   ```bash
   python app.py
   ```

4. **Access the Interface**:
   Open your web browser and navigate to `http://localhost:7860` to access the DeepScale AI interface.

---

## Usage Guide 📚

### Step 1: Document Processing 📄
1. **Upload Documents**: Click on the "Upload" button to select and upload your PDF or text documents.
2. **Process Documents**: Click the "Process Documents" button to index the documents and prepare them for querying.

### Step 2: Model Configuration ⚙️
1. **Initialize Model**: After processing the documents, click the "Initialize Model" button to load the AI model.
2. **Adjust Parameters**: Use the sliders to adjust the temperature, response length, and top-k sampling to customize the model's behavior.

### Step 3: Start Chatting 💬
1. **Ask Questions**: Type your question in the input box and press "Submit" or hit Enter.
2. **View Responses**: The AI will generate a response based on the content of your documents. You can also view the source references for each answer.

---

## Advanced Configuration 🛠️

### GPU Acceleration 🚀
If you have a GPU, you can enable GPU acceleration by modifying the `requirements.txt` file. Uncomment the GPU-enabled PyTorch line and comment out the CPU-only lines:

```plaintext
# 🔹 GPU Users: Uncomment the line below & comment the three torch lines above
torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Model Customization 🧩
You can customize the AI model by adjusting the following parameters:
- **Temperature**: Controls the randomness of the model's responses. Lower values make the model more deterministic.
- **Response Length**: Sets the maximum number of tokens in the generated response.
- **Top-K Sampling**: Limits the model's sampling to the top-k most likely tokens.

---

## Troubleshooting 🛑

- **File Loading Issues**: Ensure that the uploaded files are in PDF or text format and are not corrupted.
- **Model Initialization Errors**: Check if your system has sufficient memory and that all dependencies are correctly installed.
- **Performance Issues**: If the application is slow, consider enabling GPU acceleration or reducing the document size.

---

## Contributing 🤝

We welcome contributions from the community! If you have any suggestions, bug reports, or feature requests, please open an issue on our GitHub repository. For code contributions, feel free to submit a pull request.

---

## License 📝

DeepScale AI is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments 🙏

- **LangChain**: For providing the framework for building conversational AI applications.
- **Hugging Face**: For the pre-trained models and transformers library.
- **PyTorch**: For the deep learning framework that powers the AI model.

---

## Contact 📧

For any questions or support, feel free to reach out to us at [support@deepscale.ai](mailto:support@deepscale.ai).

---

## GitHub Repository and Version Information 📂

- **Current GitHub Repository**: [https://github.com/Adarsh3315/DeepScaleAIV1](https://github.com/Adarsh3315/DeepScaleAIV1)
- **Current Version**: **DeepScale AI V1**
- **YouTube Demo Video**: [https://youtu.be/TGgFPURzaUQ?feature=shared](https://youtu.be/TGgFPURzaUQ?feature=shared)

As we release new versions of DeepScale AI, we will continue to add them to separate repositories. Stay tuned for updates and new features! 🚀

---

Thank you for choosing DeepScale AI! We hope it enhances your document interaction experience. Happy querying! 🎉
