from fastapi import Request, FastAPI
import uvicorn
import yaml
import json
import subprocess

app = FastAPI()


def get_io_file_names(path: str):
    """reads input/output file paths from config file"""
    with open(path) as stream:
        conf = yaml.safe_load(stream)
    print("using config: ", conf)
    input_file = conf['sample-input-file']
    output_file = conf['sample-output-file']
    return input_file, output_file


def write_prompts_to_file(prompts: list, in_file: str):
    """write prompts to an input file that will then be used by model to generate text"""
    with open(in_file, 'w') as of:
        for prompt in prompts:
            of.write(prompt + "\n")


def read_model_output(out_path: str):
    """read model predictions and return them as json outputs"""
    outputs = []
    with open(out_path, 'r') as rin:
        for line in rin.readlines():
            line_json = json.loads(line)
            outputs.append(line_json)
    return outputs



@app.get("/")
async def hello_world():
    return "<p>Hello, GPT-NEOx!</p>"

@app.post("/generate")
async def hello_world(request: Request):

    prompts_data = await request.json()
    prompts = prompts_data['prompts']
    text_gen_config_path = "./configs/text_generation.yml"
    in_file, out_file = get_io_file_names(text_gen_config_path)

    write_prompts_to_file(prompts, in_file)

    # call script
    subprocess.run(["./deepy.py", "generate.py", "./configs/20B.yml", "./configs/text_generation.yml"])

    response = read_model_output(out_file)

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8085)
