import gradio as gr
import os
import time
from typing import List, Tuple, Optional
from pathlib import Path
from threading import Thread
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch

EMBEDDING_MODEL = "BAAI/bge-m3"
MODEL_NAME = "agentica-org/DeepScaleR-1.5B-Preview"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONTEXT_LENGTH = 8192

bnb_config = (
    BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    if DEVICE == "cuda"
    else None
)


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [0]
        return input_ids[0][-1] in stop_ids


def validate_file_paths(file_paths: List[str]) -> List[str]:
    valid_paths = []
    for path in file_paths:
        try:
            if Path(path).exists() and Path(path).suffix.lower() in [".pdf", ".txt"]:
                valid_paths.append(path)
        except (OSError, PermissionError) as e:
            print(f"File validation error: {str(e)}")
    return valid_paths


def load_documents(file_paths: List[str]) -> List[Document]:
    documents = []
    valid_paths = validate_file_paths(file_paths)

    if not valid_paths:
        raise ValueError("No valid PDF/TXT files found!")

    for path in valid_paths:
        try:
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.endswith(".txt"):
                loader = TextLoader(path)
            docs = loader.load()
            if docs:
                documents.extend(docs)
        except Exception as e:
            print(f"Error loading {Path(path).name}: {str(e)}")

    if not documents:
        raise ValueError("All documents failed to load.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", "。", " ", ""],
    )
    return text_splitter.split_documents(documents)


def create_vector_store(documents: List[Document]) -> FAISS:
    if not documents:
        raise ValueError("No documents to index.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    return FAISS.from_documents(documents, embeddings)


def initialize_deepseek_model(
    vector_store: FAISS,
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
    top_k: int = 50,
) -> ConversationalRetrievalChain:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, use_fast=True, trust_remote_code=True
        )

        torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto" if DEVICE == "cuda" else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            repetition_penalty=1.1,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
            batch_size=1,
            return_full_text=False,
        )

        llm = HuggingFacePipeline(
            pipeline=text_pipeline, model_kwargs={"temperature": temperature}
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question",
        )

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}
            ),
            memory=memory,
            chain_type="stuff",
            return_source_documents=True,
            verbose=False,
            max_tokens_limit=MAX_CONTEXT_LENGTH,
        )

    except Exception as e:
        raise RuntimeError(f"Model initialization failed: {str(e)}")


def format_sources(source_docs: List[Document]) -> List[Tuple[str, int]]:
    sources = []
    try:
        for doc in source_docs[:3]:
            content = doc.page_content.strip()[:500] + "..."
            page = doc.metadata.get("page", 0) + 1
            sources.append((content, page))
        while len(sources) < 3:
            sources.append(("No source found", 0))
    except Exception:
        return [("Source processing error", 0)] * 3
    return sources


def handle_conversation(
    qa_chain: Optional[ConversationalRetrievalChain],
    message: str,
    history: List[Tuple[str, str]],
) -> Tuple:
    start_time = time.time()

    if not qa_chain:
        return None, "", history, *[("System Error", 0)] * 3

    try:
        response = qa_chain.invoke({"question": message, "chat_history": history})
        answer = response["answer"].strip()
        sources = format_sources(response.get("source_documents", []))

        new_history = history + [(message, answer)]
        elapsed = f"{(time.time() - start_time):.2f}s"
        print(f"Response generated in {elapsed}")

        return (
            qa_chain,
            "",
            new_history,
            *[item for sublist in sources for item in sublist],
        )
    except Exception as e:
        error_msg = f"⚠️ Error: {str(e)}"
        return qa_chain, "", history + [(message, error_msg)], *[("Error", 0)] * 3


def create_interface() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Default()) as interface:
        qa_chain = gr.State()
        vector_store = gr.State()

        gr.Markdown(
            """
            <h1 style="text-align:center; color: #ooffff;">
              DeepScale R1
            </h1>
            <p style="text-align:center; color: #008080;">
              A Safe and Strong Local RAG System by Adarsh Pandey !!
            </p>
            """,
            elem_id="header-section",
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Step 1: Document Processing")
                file_input = gr.Files(
                    file_types=[".pdf", ".txt"], file_count="multiple"
                )
                process_btn = gr.Button("Process Documents", variant="primary")
                process_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### Step 2: Model Configuration")
                with gr.Accordion("Advanced Parameters", open=False):
                    temp_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                    token_slider = gr.Slider(
                        minimum=256,
                        maximum=4096,
                        value=1024,
                        step=128,
                        label="Response Length",
                    )
                    topk_slider = gr.Slider(
                        minimum=1, maximum=100, value=50, step=5, label="Top-K Sampling"
                    )
                init_btn = gr.Button("Initialize Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", interactive=False)

            with gr.Column(scale=1, min_width=500):
                chatbot = gr.Chatbot(
                    label="Conversation History",
                    height=450,
                    avatar_images=["2.png", "3.png"],
                )
                msg_input = gr.Textbox(
                    label="Your Query",
                    placeholder="Ask a question about your documents...",
                )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.ClearButton([msg_input, chatbot], value="Clear Chat")

                with gr.Accordion("Source References", open=True):
                    for i in range(3):
                        with gr.Row():
                            gr.Textbox(
                                label=f"Reference {i+1}", max_lines=4, interactive=False
                            )
                            gr.Number(label="Page", value=0, interactive=False)

        process_btn.click(
            fn=lambda files: (
                create_vector_store(load_documents([f.name for f in files])),
                "Documents processed successfully.",
            ),
            inputs=file_input,
            outputs=[vector_store, process_status],
            api_name="process_docs",
        )

        init_btn.click(
            fn=lambda vs, temp, tokens, k: (
                initialize_deepseek_model(vs, temp, tokens, k),
                "Model initialized successfully.",
            ),
            inputs=[vector_store, temp_slider, token_slider, topk_slider],
            outputs=[qa_chain, model_status],
            api_name="init_model",
        )

        msg_input.submit(
            fn=handle_conversation,
            inputs=[qa_chain, msg_input, chatbot],
            outputs=[qa_chain, msg_input, chatbot, *(gr.Textbox(), gr.Number()) * 3],
            api_name="chat",
        )

        submit_btn.click(
            fn=handle_conversation,
            inputs=[qa_chain, msg_input, chatbot],
            outputs=[qa_chain, msg_input, chatbot, *(gr.Textbox(), gr.Number()) * 3],
            api_name="chat",
        )

    return interface


if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0" if os.getenv("DOCKER") else "localhost",
        server_port=7860,
        show_error=True,
        share=False,
        favicon_path="1.png",
    )
