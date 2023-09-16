import gradio as gr
import time
import torch
from medusa.model.medusa_model import MedusaModel
from fastchat.model.model_adapter import get_conversation_template

# Global variable to store the chat history
chat_history = ""


def medusa_chat_interface(user_input):
    global model, tokenizer, conv, chat_history

    # Add user's input to chat history
    chat_history += "\nYou: " + user_input

    # Process the user input and get the model's response
    conv.append_message(conv.roles[0], user_input)
    conv.append_message(conv.roles[1], '')  # Placeholder for the Medusa response
    prompt = conv.get_prompt()
    print(prompt)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.base_model.device)

    outputs = model.medusa_generate(input_ids, temperature=0.7, max_steps=512)
    response = ""
    for output in outputs:
        response = output['text']
        # Send the current response to the output box
        yield response, chat_history
        time.sleep(0.01)

    # Update chat history with the complete Medusa's response after the loop
    chat_history += "\nMedusa: " + response.strip()

    return response, chat_history


if __name__ == "__main__":
    MODEL_PATH = "FasterDecoding/medusa-vicuna-7b-v1.3"
    model = MedusaModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = model.get_tokenizer()
    conv = get_conversation_template("vicuna")

    interface = gr.Interface(
        medusa_chat_interface,
        gr.components.Textbox(placeholder="Ask Medusa..."),
        [gr.components.Textbox(label="Medusa's Response", type="text"),
         gr.components.Textbox(label="Chat History", type="text")],
        live=False,
        description="Chat with Medusa",
        title="Medusa Chatbox"
    )
    interface.queue().launch()
