from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv

print("Modules are installed and imported successfully!")

'''phidata documentaion ref:- https://docs.phidata.com/agents/introduction'''

load_dotenv()

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile")
)

agent.print_response("write a romantic poem")

