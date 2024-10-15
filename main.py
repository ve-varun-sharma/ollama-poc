import requests, json
import gradio as gr
import os

model = "llama2:latest"  # You can replace the model name if needed
context = []

API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")


# Call Ollama API
def generate(prompt, context, top_k, top_p, temp):
    r = requests.post(
        API_URL,
        json={
            "model": model,
            "prompt": prompt,
            "context": context,
            "options": {"top_k": top_k, "temperature": top_p, "top_p": temp},
        },
        stream=False,
    )
    r.raise_for_status()

    response = ""

    for line in r.iter_lines():
        body = json.loads(line)
        response_part = body.get("response", "")
        print(response_part)
        if "error" in body:
            raise Exception(body["error"])

        response += response_part

        if body.get("done", False):
            context = body.get("context", [])
            return response, context


def chat(input, chat_history, top_k, top_p, temp):

    chat_history = chat_history or []

    global context
    output, context = generate(input, context, top_k, top_p, temp)

    chat_history.append((input, output))

    return chat_history, chat_history


# the first history in return history, history is meant to update the
# chatbot widget, and the second history is meant to update the state
# (which is used to maintain conversation history across interactions)


#########################Gradio Code##########################
block = gr.Blocks()


with block:

    gr.Markdown(
        """<h1><center> Mr.Olama </center></h1>
    """
    )

    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder="Type here")

    state = gr.State()
    with gr.Row():
        top_k = gr.Slider(
            0.0,
            100.0,
            label="top_k",
            value=40,
            info="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)",
        )
        top_p = gr.Slider(
            0.0,
            1.0,
            label="top_p",
            value=0.9,
            info=" Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)",
        )
        temp = gr.Slider(
            0.0,
            2.0,
            label="temperature",
            value=0.8,
            info="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)",
        )

    submit = gr.Button("SEND")

    submit.click(
        chat, inputs=[message, state, top_k, top_p, temp], outputs=[chatbot, state]
    )


block.launch(debug=True, share=True)
