import datetime

from dotenv import load_dotenv


load_dotenv()

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq

from schemas import AnswerQuestion,Reflection,ReviseAnswer

llm = ChatGroq(model='qwen/qwen3.8-27b')

parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

########################################################################################################
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",  
            """
            You are an expert researcher. Current time: {time}

            Follow these instructions:
            1. {first_instruction}
            2. Put only the answer in the answer field.
            3. Put the critique in reflection.missing and
            reflection.superfluous.
            4. Put 1-3 search queries in the top-level
            search_queries field.
            5. Never put search_queries inside reflection.
            6. Do not duplicate the reflection or search queries
            inside the answer field.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(
    time=lambda : datetime.datetime.now().isoformat()
)

#################################################################################################

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~150 word answer for the user query ."
)

# call from draft_node (main.py)
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion],
    tool_choice="AnswerQuestion"
)

# sends the user query to first_responder_prompt-template then the template goes to llm which has tools already bound to it .
# llm calls for AnswerQuestion -> go to tool_executor.py 

##################################################################################################

revise_instructions = """
Revise your previous answer using the new information.
- You should use the previous critique to add important information to your answer .
    - you MUST include numerical citations in your revised answer to ensure it can be verified .
    - Add a "references" section to the bottom of your answer (which doesnt not count towards the word limit)
        - [1] https://example1.com
        - [2] https://example2.com
    - You should use the previous critique to remove superfluous information from your answer to make SURE it is not more than 250 words.
"""

##########################################################################

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer],tool_choice='ReviseAnswer')
###########################################################################


if __name__=="__main__":
    human_message = HumanMessage(
        content="Write about AI Powered SOC / autonomous soc problem domain ,"
        "List startups that do that and raised capital"
    )

    chain = (
        first_responder_prompt_template
        |
        llm.bind_tools(tools=[AnswerQuestion],tool_choice="AnswerQuestion")
        |
        parser_pydantic
    )

    res = chain.invoke({
        "messages":[human_message],
    })

    print(res)