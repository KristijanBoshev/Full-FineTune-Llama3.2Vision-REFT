# REFT- Full Fine-tuning Demo: Llama 3.2-1B-Instruct

This repository showcases a practical demonstration of **REFT (Representational Fine-Tuning)** for rapid personalization of large language models (LLMs), specifically applied to Meta's Llama 3.2-1B-Instruct model. This project, created as an academic exercise, highlights the efficiency of REFT in imbuing an LLM with targeted knowledge and deploying it as an interactive chatbot on Hugging Face Spaces.

## Project Overview

This project demonstrates how REFT, a novel parameter-efficient fine-tuning technique, can be used to quickly personalize an LLM with specific information. We fine-tune Meta's Llama 3.2-1B-Instruct model on a small dataset containing personal details about "Kristijan Boshev."  The fine-tuning process, leveraging REFT, is remarkably fast, taking under 300 seconds on a Colab GPU. The resulting personalized model is then deployed as a Gradio chatbot on Hugging Face Spaces, providing a readily accessible demo.

**Key Highlights:**

*   **Rapid Personalization with REFT:** Demonstrates the speed and efficiency of Representational Fine-Tuning (REFT) for adapting LLMs.
*   **Targeted Knowledge Injection:**  Shows how REFT can effectively inject specific personal knowledge into a model using a minimal dataset.
*   **Parameter-Efficient Fine-tuning:**  REFT updates only a small fraction of the model's parameters, leading to fast training and efficient resource utilization.
*   **Hugging Face Spaces Deployment:**  Provides a live, interactive chatbot demo deployed on Hugging Face Spaces using Gradio.
*   **Academic Project:** Designed as an educational example for NLP studies.

## Live Demo

You can interact with the REFT-personalized chatbot live on Hugging Face Spaces:

**https://huggingface.co/spaces/kiko2001/Finetuned-with-REFT-by-Kristijan-Boshev**

*Example questions to try:*

*   "Who is Kristijan Boshev?"
*   "Where does Kristijan Boshev come from?"
*   "Write a rap about Kristijan Boshev."

## Setup and Usage

### Running the Gradio Demo Locally

1.  **Clone the repository:**
    ```bash
    git clone <repository-link>
    cd <into root>
    ```
2.  **Install required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set your Hugging Face Token:**
    You need a Hugging Face token to access models from the Hub. Set it as an environment variable:
    ```bash
    export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
    ```
    Replace `"YOUR_HUGGINGFACE_TOKEN"` with your actual token.
4.  **Run the Gradio app:**
    ```bash
    python app.py
    ```
    This will launch the Gradio chatbot locally. Access it in your browser at the provided URL (`http://127.0.0.1:7860/`).

### Re-training the REFT Model (Optional)

You can re-train the REFT model using the provided Jupyter Notebook:

1.  **Open `training.ipynb` in Jupyter Notebook or Google Colab.**
2.  **Follow the instructions in the notebook.**
    *   Ensure you have a GPU runtime enabled for faster training (Colab is recommended).
    *   You will need to log in to Hugging Face Hub within the notebook to push the trained model.
3.  **Run the notebook cells sequentially.**

**Training Details:**

*   **Base Model:** `meta-llama/Llama-3.2-1B-Instruct`
*   **Fine-tuning Technique:** REFT (Representation Editing for Tuning) with LoReFT (Low-Rank Adaptation for Interventions)
*   **Dataset:** Small, personalized dataset of 10 examples related to "Kristijan Boshev."
*   **Training Epochs:** 200
*   **Optimizer:** AdamW
*   **Learning Rate:** $4 \times 10^{-3}$
*   **Per-device Train Batch Size:** 5
*   **Training Time:** Under 300 seconds on a Colab GPU instance
*   **Intervention Layers:** Layers 8 and 15, targeting `block_output` component.
*   **Low-Rank Dimension:** 2

**Training Loss Logs:**

| Step | Training Loss | Step | Training Loss |
|------|---------------|------|---------------|
| 20   | 2.521900      | 220  | 0.015100      |
| 40   | 1.337100      | 240  | 0.008300      |
| 60   | 0.721200      | 260  | 0.006400      |
| 80   | 0.379000      | 280  | 0.005600      |
| 100  | 0.173700      | 300  | 0.005100      |
| 120  | 0.095500      | 320  | 0.004900      |
| 140  | 0.071000      | 340  | 0.004600      |
| 160  | 0.060700      | 360  | 0.004400      |
| 180  | 0.039800      | 380  | 0.004300      |
| 200  | 0.031000      | 400  | 0.004300      |

## Repository Files

*   `app.py`:  The Gradio application script for deploying the personalized chatbot on Hugging Face Spaces or running locally.
*   `training.ipynb`: Jupyter Notebook containing the code for fine-tuning the Llama 3.2-1B-Instruct model using REFT.
*   `requirements.txt`: Lists the Python libraries required to run the demo and training.
