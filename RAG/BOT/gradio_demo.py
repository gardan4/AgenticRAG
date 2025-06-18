import gradio as gr
from plan_adviser_v2 import get_plan_advice_from_file

def adviser_interface(file_bytes):
    if file_bytes is None:
        return "Please upload a PDF file containing your project plan."
    return get_plan_advice_from_file(file_bytes)

iface = gr.Interface(
    fn=adviser_interface,
    inputs=gr.File(label="Upload your Project Plan Proposal PDF", type="binary"),
    outputs=gr.Textbox(label="Plan Adviser Feedback"),
    title="Project Plan Adviser"
)

if __name__ == "__main__":
    iface.launch(share=True)