"""
Application wiring for the ICU intake Gradio interface.
"""

from collections.abc import Callable

import gradio as gr

from ui.config import OPTIONAL_TAB_ID, REQUIRED_TAB_ID
from ui.handlers import (
    change_optional_field,
    done_intake,
    initialize,
    restart_optional,
    restart_required,
    submit_optional,
    submit_required,
)


DoneOutputCallback = Callable[[str, dict[str, str], dict[str, str]], str | None]


def build_app(done_output_callback: DoneOutputCallback | None = None) -> gr.Blocks:
    """
    Create and return the configured Gradio app.

    done_output_callback receives:
    1) The final string from done_intake
    2) Required answers dict
    3) Optional answers dict
    It may return a replacement output string; returning None keeps the original.
    """

    def _done_handler(required_answers: dict[str, str], optional_answers: dict[str, str]) -> str:
        base_output = done_intake(required_answers, optional_answers)
        if done_output_callback is None:
            return base_output

        processed = done_output_callback(base_output, required_answers, optional_answers)
        return base_output if processed is None else processed

    # This is the building of the UI.
    with gr.Blocks(title="ICU Intake Form") as demo:
        gr.Markdown("## ICU Intake Form") # Heading

        required_answers_state = gr.State({}) # Dict of required field names
        required_index_state = gr.State(0) # Index of current required field for progress bar
        optional_answers_state = gr.State({}) # Dict of optional field names
        optional_index_state = gr.State(0) # Index of current optional field for progress bar
        optional_selected_state = gr.State(None) # Currently selected optional field for display

        # Draw the two tabs
        with gr.Tabs(selected=REQUIRED_TAB_ID) as stage_tabs:

            # Draw the required fields tab
            with gr.Tab("Required Fields", id=REQUIRED_TAB_ID):
                req_field_md = gr.Markdown()
                req_status_box = gr.HTML(label="Required Progress")
                req_summary_box = gr.HTML(label="Required Captured Inputs")

                # Draw the three types of inputs, displayed one at a time based on field type
                with gr.Row():
                    req_text_input = gr.Textbox(label="Text Input", visible=False)
                    req_number_input = gr.Number(label="Numeric Input", visible=False)
                    req_choice_input = gr.Dropdown(label="Choice Input", choices=[], visible=False)

                # Draw the required tab buttons
                with gr.Row():
                    req_submit_btn = gr.Button("Add Vital", variant="primary")
                    req_restart_btn = gr.Button("Restart")

            # Draw the optional fields tab
            with gr.Tab("Optional Fields", id=OPTIONAL_TAB_ID):
                opt_field_md = gr.Markdown()
                opt_status_box = gr.HTML(label="Optional Progress")
                opt_summary_box = gr.HTML(label="Optional Captured Inputs",)
                opt_field_selector = gr.Dropdown(
                    label="Choose optional field",
                    choices=[],
                    value=None,
                    visible=False,
                )

                with gr.Row():
                    opt_text_input = gr.Textbox(label="Text Input", visible=False)
                    opt_number_input = gr.Number(label="Numeric Input", visible=False)
                    opt_choice_input = gr.Dropdown(label="Choice Input", choices=[], visible=False)

                with gr.Row():
                    opt_submit_btn = gr.Button("Add Vital", variant="primary")
                    opt_restart_btn = gr.Button("Restart")
                    done_btn = gr.Button("Done", variant="huggingface")

        final_status_box = gr.HTML(label="Final Processing Output")

        full_outputs = [
            required_answers_state,
            required_index_state,
            optional_answers_state,
            optional_index_state,
            optional_selected_state,
            req_field_md,
            req_status_box,
            req_summary_box,
            req_text_input,
            req_number_input,
            req_choice_input,
            req_submit_btn,
            opt_field_md,
            opt_status_box,
            opt_summary_box,
            opt_field_selector,
            opt_text_input,
            opt_number_input,
            opt_choice_input,
            opt_submit_btn,
            opt_restart_btn,
            done_btn,
            stage_tabs,
            final_status_box,
        ]

        demo.load(initialize, outputs=full_outputs)

        req_submit_btn.click(
            submit_required,
            inputs=[
                required_answers_state,
                required_index_state,
                optional_answers_state,
                optional_index_state,
                optional_selected_state,
                req_text_input,
                req_number_input,
                req_choice_input,
            ],
            outputs=full_outputs,
        )

        req_text_input.submit(
            submit_required,
            inputs=[
                required_answers_state,
                required_index_state,
                optional_answers_state,
                optional_index_state,
                optional_selected_state,
                req_text_input,
                req_number_input,
                req_choice_input,
            ],
            outputs=full_outputs,
        )

        req_number_input.submit(
            submit_required,
            inputs=[
                required_answers_state,
                required_index_state,
                optional_answers_state,
                optional_index_state,
                optional_selected_state,
                req_text_input,
                req_number_input,
                req_choice_input,
            ],
            outputs=full_outputs,
        )

        req_choice_input.change(
            submit_required,
            inputs=[
                required_answers_state,
                required_index_state,
                optional_answers_state,
                optional_index_state,
                optional_selected_state,
                req_text_input,
                req_number_input,
                req_choice_input,
            ],
            outputs=full_outputs,
        )

        req_restart_btn.click(
            restart_required,
            inputs=[optional_answers_state, optional_index_state, optional_selected_state],
            outputs=full_outputs,
        )

        opt_field_selector.change(
            change_optional_field,
            inputs=[
                required_answers_state,
                required_index_state,
                optional_answers_state,
                optional_index_state,
                opt_field_selector,
            ],
            outputs=full_outputs,
        )

        opt_submit_btn.click(
            submit_optional,
            inputs=[
                required_answers_state,
                required_index_state,
                optional_answers_state,
                optional_index_state,
                optional_selected_state,
                opt_text_input,
                opt_number_input,
                opt_choice_input,
            ],
            outputs=full_outputs,
        )

        opt_text_input.submit(
            submit_optional,
            inputs=[
                required_answers_state,
                required_index_state,
                optional_answers_state,
                optional_index_state,
                optional_selected_state,
                opt_text_input,
                opt_number_input,
                opt_choice_input,
            ],
            outputs=full_outputs,
        )

        opt_number_input.submit(
            submit_optional,
            inputs=[
                required_answers_state,
                required_index_state,
                optional_answers_state,
                optional_index_state,
                optional_selected_state,
                opt_text_input,
                opt_number_input,
                opt_choice_input,
            ],
            outputs=full_outputs,
        )

        opt_choice_input.change(
            submit_optional,
            inputs=[
                required_answers_state,
                required_index_state,
                optional_answers_state,
                optional_index_state,
                optional_selected_state,
                opt_text_input,
                opt_number_input,
                opt_choice_input,
            ],
            outputs=full_outputs,
        )

        opt_restart_btn.click(
            restart_optional,
            inputs=[required_answers_state, required_index_state],
            outputs=full_outputs,
        )

        done_btn.click(
            _done_handler,
            inputs=[required_answers_state, optional_answers_state],
            outputs=final_status_box,
        )

    return demo
