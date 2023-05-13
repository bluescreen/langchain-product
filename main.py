from langchain.tools import SteamshipImageGenerationTool
from steamship import Steamship
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from util import show_ouput
from dotenv import load_dotenv

load_dotenv()

SteamshipImageGenerationTool.update_forward_refs(Steamship=Steamship)

product = "colorful socks"

prompt_name = "What is a good name for a company that makes {product}?"
prompt_claim = "What is a good claim for a company that makes {product}?"
prompt_logo = "Create a logo for a company that makes colorful socks?"

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(input_variables=["product"], template=prompt_name)
chain1 = LLMChain(llm=llm, prompt=prompt)

prompt = PromptTemplate(input_variables=["product"], template=prompt_claim)
chain2 = LLMChain(llm=llm, prompt=prompt)

overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
overall_chain.run(product)

tools = [
    SteamshipImageGenerationTool(model_name="stable-diffusion")
]
mrkl = initialize_agent(tools,
                        llm,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True)
output = mrkl.run(prompt_logo)

show_ouput(output)
