import google.generativeai as genai
from openai import OpenAI
import anthropic

# Add your own API keys here
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
GOOGLE_GEMINI_API = ""
DEEPSEEK_API = ""



def send_to_openai_chatgpt(content, model="gpt-4", max_tokens=100, temperature=0.7, timeout=30):
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout
    )
    return response.choices[0].message

def send_to_anthropic_claude(content, model = "claude-3-5-sonnet-20241022"):
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=ANTHROPIC_API_KEY,
    )
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "user", "content": content}
        ]
    )
    return message.content[0].text

def send_to_google_gemini(content):
            # Configure the API key
    
    client = genai.Client(api_key=GOOGLE_GEMINI_API)

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=content,
    )
    return response.candidates[0].content.parts[0].text  # Extract the text output

def send_to_deepseek(content):
    client = OpenAI(api_key=DEEPSEEK_API, base_url='https://api.siliconflow.cn/v1/')
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "user", "content": content},
    ],
        max_tokens=4096,
        stream=False
    )

    return response.choices[0].message.content

def send_to_deepseek_r1(content):
    client = OpenAI(api_key=DEEPSEEK_API, base_url='https://api.siliconflow.cn/v1/')

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[
            {"role": "user", "content": content},
    ],
        max_tokens=4096,
        stream=False
    )

    return response.choices[0].message.content

def send_to_openai_o1(content):
    client = OpenAI(api_key=OPENAI_API_KEY)
    # Prepare the request
    response = client.chat.completions.create(
        model="o1",
        messages=[{"role": "user", "content": content}],
        timeout=30
    )
    return response.choices[0].message

def send_to_llama(content, pipeline = None):
    messages = [
        {"role": "user", "content": content},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=1024,
    )

    return outputs[0]["generated_text"][-1]["content"]


