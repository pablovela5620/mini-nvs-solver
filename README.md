# zero_spaces_template
Template for creating Repo that syncs with ZeroSpaces from huggingface using pixi

## App
This calls gradio_app.py to run gradio under the ZeroGPU environment with pixi
## Gradio App
Where the actual gradio logic goes, generally calls some block from `src/gradio_ui`
## HF Upload
Once happy with app, run this to upload the pyproject.toml .pixi.sh app.py and gradio.py with optionally examples