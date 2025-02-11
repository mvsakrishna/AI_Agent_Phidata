"""Run `pip install yfinance` to install dependencies."""

'''phidata documentation ref:- https://docs.phidata.com/agents/introduction'''
# pip install phidata transformers torch accelerate duckduckgo-search newspaper4k lxml_html_clean
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools

load_dotenv()

# Load DeepSeek model from Hugging Face
model_name = "deepseek-ai/deepseek-coder-1.3b"  # Use "deepseek-ai/deepseek-67b" for a larger model

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")


def get_company_symbol(company: str) -> str:
    """Use this function to get the symbol for a company.

    Args:
        company (str): The name of the company.

    Returns:
        str: The symbol for the company.
    """
    symbols = {
        "Phidata": "MSFT",
        "Infosys": "INFY",
        "Tesla": "TSLA",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Google": "GOOGL",
    }
    return symbols.get(company, "Unknown")


def generate_response(prompt: str) -> str:
    """Generate a response using DeepSeek model from Hugging Face.

    Args:
        prompt (str): The user query.

    Returns:
        str: The generated response.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Send input to GPU if available
    outputs = model.generate(**inputs, max_length=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Use Hugging Face DeepSeek model instead of Groq
agent = Agent(
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True),
        get_company_symbol,
    ],
    instructions=[
        "Use tables to display data.",
        "If you need to find the symbol for a company, use the get_company_symbol tool. Even if it's not a public company",
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

# Generate response using Hugging Face DeepSeek model
query = "Summarize and compare analyst recommendations and fundamentals for TSLA and Phidata. Show in tables."
response = generate_response(query)

# Print response
print(response)
