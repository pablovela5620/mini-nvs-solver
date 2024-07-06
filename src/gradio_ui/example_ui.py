import gradio as gr
from gradio_rerun import Rerun

import rerun as rr
import rerun.blueprint as rrb

import mmcv
import time
import cv2


@rr.thread_local_stream("nvs_solver")
def streaming_repeated_blur(img):
    stream = rr.binary_stream()

    if img is None:
        raise gr.Error("Must provide an image to blur.")

    img = mmcv.imrescale(img, 0.25)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="image/original"),
            rrb.Spatial2DView(origin="image/blurred"),
        ),
        collapse_panels=True,
    )

    rr.send_blueprint(blueprint)

    rr.set_time_sequence("iteration", 0)

    rr.log("image/original", rr.Image(img).compress(jpeg_quality=80))
    yield stream.read()

    blur = img

    for i in range(100):
        rr.set_time_sequence("iteration", i)

        # Pretend blurring takes a while so we can see streaming in action.
        time.sleep(0.1)
        blur = cv2.GaussianBlur(blur, (3, 3), 0)

        rr.log("image/blurred", rr.Image(blur).compress(jpeg_quality=80))

        # Each time we yield bytes from the stream back to Gradio, they
        # are incrementally sent to the viewer. Make sure to yield any time
        # you want the user to be able to see progress.
        yield stream.read()


with gr.Blocks() as ui:
    with gr.Tab("Streaming"):
        with gr.Row():
            img = gr.Image(interactive=True, label="Image")
            with gr.Column():
                stream_blur = gr.Button("Stream Repeated Blur")

    # Rerun 0.16 has issues when embedded in a Gradio tab, so we share a viewer between all the tabs.
    # In 0.17 we can instead scope each viewer to its own tab to clean up these examples further.
    with gr.Row():
        viewer = Rerun(
            streaming=True,
        )

    stream_blur.click(streaming_repeated_blur, inputs=[img], outputs=[viewer])

    gr.Examples(
        [
            [
                "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"
            ]
        ],
        fn=stream_blur,
        inputs=[img],
        outputs=[viewer],
    )
