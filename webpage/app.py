from flask import Flask, render_template, request, session
import flask
import os
from openai import OpenAI
from transformers import  AutoTokenizer, AutoConfig, TextIteratorStreamer
import torch
import math
from eval_article import *
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
# Load the model and tokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

model_path = "YOUR_MODEL_PATH"
tokenizer_name_or_path =  "YOUR_TOKENIZER_PATH"
tokenizer_class = AutoTokenizer
tokenizer_kwargs = {
    "cache_dir": model_path,
    "use_fast": False,
    "trust_remote_code": True,
}
tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

if tokenizer.pad_token_id is None:
    if tokenizer.unk_token_id is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.eos_token
world_size = int(os.environ.get("WORLD_SIZE", "1"))

torch_dtype = getattr(torch, 'bfloat16')
config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    cache_dir=model_path
)

      
setattr(config, "group_size_ratio", 0.25)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch_dtype,
    load_in_4bit=False,
    load_in_8bit=False,
    use_flash_attention_2=True,
    low_cpu_mem_usage=True,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=None,
)

streamer = TextIteratorStreamer(tokenizer)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

app.config['SECRET_KEY'] = os.urandom(24)

def get_total_question(text):
    pre_text =  "阅读下面的材料，根据要求写一篇不少于800字的文章。（60分）"
    post_text = "要求选好角度，确定立意，明确文体，自拟标题；不要脱离材料内容及含意的范围作文;不要套作，不得抄袭。 "
    if text.find("800") >= 0 or text.find("立意") >= 0:
      total_text = text
    else:
      total_text = pre_text + text + post_text
    total_text = "[extra_id_0]" + total_text + "[extra_id_1]"
    return total_text

previous_question = ''

@app.route("/chatgpt",methods=["POST","GET"])
def chatgpt():
    global previous_question
    question = request.args.get("question","")
    question = str(question).strip().replace('<br>', '\n')
    question = get_total_question(question)
    if question and (previous_question != question):
        previous_question = question
        def stream():
            input_ids = tokenizer.encode([question], return_tensors="pt").to(model.device)
            generation_kwargs = dict(input_ids=input_ids, streamer=streamer, max_length=1500, use_cache=False)
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            all_trunks = []
            for trunk in streamer:
                all_trunks.append(trunk)
                if '[extra_id_1]' in trunk:
                    trunk = trunk.replace('[extra_id_1]','<br>')
                    trunk = '<div class="question">' + trunk.strip() + '</div>'
                else:
                    trunk = trunk.replace('\n\n', '\n')
                if '[extra_id_0]' in trunk:
                    trunk = trunk.replace('[extra_id_0]','')
                yield "data: %s\n\n" % trunk.replace("\n","<br>&emsp;&emsp;")
            del input_ids
            torch.cuda.empty_cache()
            all_trunks = "".join(all_trunks)
            score, analysis = eval_article(question, all_trunks)
            yield 'data: <br><br><div class="score">%s</div><br>\n\n' % str(score)
            yield 'data: <div class="analysis">%s\n\n</div>' % analysis
            yield "data: [DONE]\n\n"
        return flask.Response(stream(),mimetype="text/event-stream")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
