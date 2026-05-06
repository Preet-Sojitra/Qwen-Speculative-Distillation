import json
import time
import gradio as gr
from transformers import AutoTokenizer

print("[INFO] Loading Tokenizer...")
TOKENIZER_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

print("[INFO] Loading Race Data...")
try:
    with open("demo_race_data.json", "r") as f:
        RACE_DATA = json.load(f)
except FileNotFoundError:
    print("ERROR: Could not find 'demo_race_data.json'. Please ensure it is in the same folder.")
    exit()

def simulate_race():
    ar_timeline = RACE_DATA["autoregressive_timeline"]
    spec_timeline = RACE_DATA["speculative_timeline"]
    
    # Track the accumulated tokens for each model
    ar_tokens = []
    spec_tokens = []
    
    ar_text = ""
    spec_text = ""
    
    # State tracking
    ar_idx = 0
    spec_idx = 0
    ar_done_time = None
    spec_done_time = None
    
    start_time = time.time()
    
    # Loop until BOTH timelines are fully rendered
    while ar_idx < len(ar_timeline) or spec_idx < len(spec_timeline):
        current_time = time.time() - start_time
        
        updated = False
        
        # Check if the Autoregressive model has generated a new token at this timestamp
        if ar_idx < len(ar_timeline) and current_time >= ar_timeline[ar_idx]["time"]:
            ar_tokens.extend(ar_timeline[ar_idx]["tokens"])
            ar_text = tokenizer.decode(ar_tokens, skip_special_tokens=True)
            ar_idx += 1
            updated = True
            if ar_idx >= len(ar_timeline):
                ar_done_time = current_time
            
        # Check if the Speculative model has generated a new chunk at this timestamp
        if spec_idx < len(spec_timeline) and current_time >= spec_timeline[spec_idx]["time"]:
            spec_tokens.extend(spec_timeline[spec_idx]["tokens"])
            spec_text = tokenizer.decode(spec_tokens, skip_special_tokens=True)
            spec_idx += 1
            updated = True
            if spec_idx >= len(spec_timeline):
                spec_done_time = current_time
            
        # If either model updated its text, yield to the Gradio UI
        if updated:
            # Freeze timer when a model finishes
            ar_time_str = f"⏱️ Elapsed: **{ar_done_time:.2f}s** ✅ Done!" if ar_done_time is not None else f"⏱️ Elapsed: **{current_time:.2f}s**"
            spec_time_str = f"⏱️ Elapsed: **{spec_done_time:.2f}s** ✅ Done!" if spec_done_time is not None else f"⏱️ Elapsed: **{current_time:.2f}s**"
            yield ar_text, spec_text, ar_time_str, spec_time_str
            
        # Sleep for 10ms to prevent CPU thrashing
        time.sleep(0.01)


# Custom CSS to make it look like a sleek terminal
custom_css = """
.output-box textarea {
    font-family: 'Courier New', Courier, monospace !important;
    background-color: #1e1e1e !important;
    color: #d4d4d4 !important;
    font-size: 14px !important;
}
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css) as demo:
    gr.Markdown(
        """
        # 🏎️ Adaptive Speculative Decoding: Inference Race
        **Project Demo:** Qwen2.5-Coder-7B Autoregressive vs. Speculative Decoding (KD Draft)
        """
    )
    
    with gr.Row():
        prompt_box = gr.Textbox(
            label="Input Prompt", 
            value=RACE_DATA["prompt"], 
            interactive=False
        )
    
    start_btn = gr.Button("🏁 Start Inference Comparison", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🐢 Standard Autoregressive (1.0x)")
            baseline_output = gr.Textbox(
                label="", 
                lines=20, 
                elem_classes="output-box",
                interactive=False
            )
            ar_timer = gr.Markdown("⏱️ Elapsed: **—**")
            
        with gr.Column():
            gr.Markdown("### 🐇 Speculative Decoding (2.5x+)")
            speculative_output = gr.Textbox(
                label="", 
                lines=20, 
                elem_classes="output-box",
                interactive=False
            )
            spec_timer = gr.Markdown("⏱️ Elapsed: **—**")
            
    start_btn.click(
        fn=simulate_race, 
        inputs=None, 
        outputs=[baseline_output, speculative_output, ar_timer, spec_timer]
    )

if __name__ == "__main__":
    print("[INFO] Launching Gradio App...")
    demo.launch()